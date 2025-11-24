import json
import yaml
import os
import subprocess
from typing import Any, Mapping, Optional, TypedDict

import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .base import BaseAgent

class HOSSSourceFiles:
    """Source directory and file names for
    code required to prepare and run HOSS"""
    src_dir = "src_HOSS"
    config_name = "config_HOSS"
    batch_job_creator_name = "batch_job_creator"
    batch_run_name = "run_HOSS"
    run_postprocessing_name = "run_postprocessing"
    load_metrics_name = "inspect_metrics"
    text_to_json = "text_to_json"


class HOSSState(TypedDict, total=False):
    """Describe HOSS State, passed along LangGraph nodes"""
    simulation_task: str
    Lammps_data: None
    Json_data: None
    build_metrics: None
    analyze_metrics: None
    # Hoss simulation keys
    HOSS_run_dir: str
    fracture_idcs: list
    material_idcs: list
    choice_json: str = ""
    # Hoss badge job create keys
    batch_job_creation_attempts: str = 0
    batch_create_returncode: Optional[int]
    batch_create_stdout: str
    batch_create_stderr: str
    # Hoss run keys
    batch_run_returncode: Optional[int]
    batch_run_stdout: str
    batch_run_stderrs: str
    # Metric analysis keys
    postprocessing_dir: str
    analysis_attempts: str = 0
    analysis_returncode: Optional[int]
    analysis_stdout: str
    analysis_stderrs: str


class HOSSAgent(BaseAgent):
    def __init__(
        self,
        llm,
        workspace="./workdir_HOSS",
        hoss_src_dir=None,
        Lammps_to_json_name=None,
        batch_job_creation_attempts_max=8,
        analysis_attempts_max=5,
        max_tokens: int = 200000,
        plot_agent_graph=False,
        **kwargs,
    ):
        # File extension for files modified by HOSS agent
        self.temp_agent_name = "temp_agent_HOSS"

        # Source directory for HOSS batch build files
        if hoss_src_dir is None:
            self.src_dir = HOSSSourceFiles.src_dir
        else:
            self.src_dir = hoss_src_dir

        # Folder name for HOSS config file
        self.config_name = HOSSSourceFiles.config_name
        self.config_name_temp = HOSSSourceFiles.config_name \
            + '_' + self.temp_agent_name
        # Name for json file to convert Lammps data file to
        # Give a default name to json from Lammps file if not set
        if Lammps_to_json_name is None:
            self.Lammps_to_json_name = "Lammps_to_json_file"
        else:
            self.Lammps_to_json_name = Lammps_to_json_name
            
        # Maximum number of attempts to set up HOSS batch jobs, analysis run
        self.bjc_attempts_max = batch_job_creation_attempts_max
        self.analysis_attempts_max = analysis_attempts_max

        self.max_tokens = max_tokens

        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)

        super().__init__(llm, **kwargs)

        # Specify chains here
        self.str_parser = StrOutputParser()

        # Chain: Run Hoss or analyse output
        self.check_input_chain = (
            ChatPromptTemplate.from_template(
                "Consider this task: {simulation_task}. Decide if the task "
                "requires running HOSS using Lammps-based data, or Json "
                "file-based data, or both, or neither. Further, check if the "
                "task requires building and/or analysing post-processing files"
                "that create metrics for existing simulation output."
                "Use this exact schema in your response:\n"
                "{{\n"
                '  "Lammps_data": <bool>\n'
                '  "Json_data": <bool>\n'
                '  "build_metrics": <bool>\n'
                '  "analyze_metrics": <bool>\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        # Chain: (re)set config file for Hoss run
        update_config_str = (
            "Consider this task: {simulation_task}. This task contains " 
            "running the HOSS code, based on a yaml file whose content is this:"
            "### {config_content} ###"
            "Adjust the config content appropriately according to the task, "
            "including in particular any entries related to samples and "
            "Return precisely the adjusted config content in your response,"
            "and nothing else. Also make sure to include a name for "
            "`run_output_dir_name` in the config file."
        )

        repeat_update_config_str = (
            "\nNote that the config file is used by a python file to create "
            "batch run scripts. You have attempted to adjust the config "
            "content before, and based on your config content the batch run "
            "python file led to this error: {batch_creation_error}. Adjust "
            "your config content to avoid this error, ideally while still "
            "satisfying the original task."
        )

        self.update_config_file = (
            ChatPromptTemplate.from_template(
                update_config_str
            )
            | self.llm
            | self.str_parser
        )
        
        self.repeat_update_config_file = (
            ChatPromptTemplate.from_template(
                update_config_str + repeat_update_config_str
            )
            | self.llm
            | self.str_parser
        )

        # Chain: check for output location when only doing postprocessing
        self.check_output_chain = (
            ChatPromptTemplate.from_template(
                "Consider this task: {simulation_task}, which you decided "
                "relates to building and/or analysing post-processing files."
                "According to the task, in which folder is the simulation "
                "output? Use this exact scheme in your response, where <str> "
                "is either the output path in your work directory, or equal "
                "to null if the task does not specify the path."
                "{{\n"
                '  "Post-processing output": <str>\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        # Chain: check for whether post-processing needs to be run
        self.check_postprocessing_chain = (
            ChatPromptTemplate.from_template(
                "Consider this task: {simulation_task}, which you decided "
                "relates to building and/or analysing post-processing files."
                "According to the task, should we run the post-processing "
                "script? Use this exact scheme in your response: "
                "{{\n"
                '  "Need to run post-processing": <bool>\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        # Chain: check for whether metric analysis should be run
        self.check_analysis_chain = (
            ChatPromptTemplate.from_template(
                "Consider this task: {simulation_task}. Does this task include "
                "an analysis of the post-processed (metric-related) data? Use "
                "this exact scheme in your response: "
                "{{\n"
                '  "Need to run post-analysis": <bool>\n'
                "}}\n"
            )
            | self.llm
            | self.str_parser
        )

        # Chain: (re)run analysis
        run_analysis_str = (
            "Consider this task: {simulation_task}. The task contains "
            "metric output, which is given by a dictionary within the folder "
            "`{output_dir}/results/metrics`. A sample script of loading the "
            "dictionary is given by: ### {metrics_script} ###"
            "Write code based on this script that perfoms an analysis on the "
            "metric output, corresponding to the task. Return precisely the "
            "adjusted script in your response, and nothing else."
        )

        repeat_analysis_str = (
            "\nNote that you have attempted to adjust the sample script "
            "before, and the adjusted script led to this error: "
            "{analysis_error}. Adjust your script to avoid this error, "
            "ideally while still satisfying the original task."
        )

        self.build_analysis_script = (
            ChatPromptTemplate.from_template(
                run_analysis_str
            )
            | self.llm
            | self.str_parser
        )

        self.repeat_build_analysis_script = (
            ChatPromptTemplate.from_template(
                run_analysis_str + repeat_analysis_str
            )
            | self.llm
            | self.str_parser
        )

        # Build Langraph here, given nodes and edges
        self._action = self._build_graph()

        # Optionally plot graph
        if plot_agent_graph:
            self.plot_graph()

    def plot_graph(self):
        """Plost agent's graph."""
        self._action.get_graph().draw_mermaid()
        img = self._action.get_graph().draw_mermaid_png()
        open("Hoss_agent_graph.png", "wb").write(img)

    def _check_input_task(self, state: HOSSState) -> HOSSState:
        """Node checking if HOSS simulation should be run and/or post-
        analysis should be done. For fomer, need to load input data: either
        LAMMPS data that needs to be converted, or HOSS-type Json input data."""
        required_data = self.check_input_chain.invoke({
                "simulation_task": state["simulation_task"]})

        if '"Lammps_data": true' in required_data:
            state["Lammps_data"] = None

        if '"Json_data": true' in required_data:
            state["Json_data"] = None

        if '"build_metrics": true' in required_data:
            state["build_metrics"] = None

        if '"analyze_metrics": true' in required_data:
            state["analyze_metrics"] = None

        return state

    def _check_HOSS_pipeline(self) -> str:
        """Search for config file with paths to HOSS and cubit executables;
        check if they exist"""
        missing_config = False
        missing_HOSS_executable = False
        missing_cubit_executable = False

        # check for config file
        config = None
        try:
            with open(f"{self.src_dir}/{self.config_name}.yaml", "r") as file:
                config = yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Config file not found or error parsing: {e}")
            missing_config = True

        if not missing_config:
            # check for Hoss executable
            try:
                cluster_base_path = config["cluster_base_path"]
                HOSS_executable_path = config["HOSS_executable_path"]
                Hoss_path = os.path.join(cluster_base_path, HOSS_executable_path)
                if not os.path.exists(Hoss_path):
                    raise FileNotFoundError(f"HOSS executable not found at: {Hoss_path}")
            except (KeyError, FileNotFoundError) as e:
                print(f"Error attempting to locate HOSS executable: {e}")
                missing_HOSS_executable = True

            # check for cubit executable
            try:
                cluster_base_path = config["cluster_base_path"]
                cubit_executable_path = config["cubit_executable_path"]
                cubit_path = os.path.join(cluster_base_path, cubit_executable_path)
                if not os.path.exists(cubit_path):
                    raise FileNotFoundError(f"cubit executable not found at: {cubit_path}")
            except (KeyError, FileNotFoundError) as e:
                print(f"Error attempting to locate cubit executable: {e}")
                missing_cubit_executable = True

        if any([missing_config, missing_HOSS_executable, missing_cubit_executable]):
            print("Need to have HOSS config and HOSS, cubit executables "
                  "available for HOSS agent to run.\nCheck specified HOSS "
                  "functionality source folder, config file locations, HOSS, "
                  "cubit paths in config file.\n"
                  f"Configuration file found: {not missing_config}\n"
                  f"Cubit executable found:   {not missing_cubit_executable}\n"
                  f"HOSS executable found:    {not missing_cubit_executable}\n")
            return "check_HOSS_pipeline_incomplete"
        return "check_HOSS_pipeline_complete"

    def _simulate_or_analyze(self, state: HOSSState) -> str:
        """Conditional to decide if we want to run Hoss simulation or otherwise
        do metrics analysis."""

        # Order matters here wrt graph conditionals
        if "Lammps_data" in state:
            return "convert_Lammps_data"
        if "Json_data" in state or "Json_from_Lammps_data" in state:
            return self._check_HOSS_pipeline()
        if "build_metrics" in state or "analyze_metrics" in state:
            return "run_post_analysis"
        print("Unclear input task; HOSS Agent can either run HOSS based on "
              "Lammps or Json input data, or analyze output metrics.")
        return "unclear_simulation_task"

    def _convert_LAMMPS_data(self, state: HOSSState) -> HOSSState:
        """Convert LAMMPS output data to Json data as required by HOSS code"""
        text_to_json_path = os.path.join(self.src_dir, "text_to_json.py")
        if not os.path.exists(text_to_json_path):
            raise FileNotFoundError(f"Text to Json conversion script not "
                                    "found in HOSS pipeline source directory.")

        #TODO: finish LAMMPS hook
        LAMMPS_text_files = None
        data_dir = None
        if LAMMPS_text_files is None and data_dir is None:
            raise NotImplementedError

        result = subprocess.run(["python", \
            f"{self.src_dir}/{HOSSSourceFiles.text_to_json}.py", \
            "--Json_name", self.Lammps_to_json_name,
            "--text_files"] + LAMMPS_text_files + ["--data_dir", data_dir])

        if result.returncode == 0:
            print("Transferred Lammps output to json data format.")
        else:
            # TO DO: catch errors from running this file
            raise NotImplementedError("Lammps to Json conversion did not run "
                                      "without error; error handling not yet "
                                      "implemented. Exiting.")

        # Update Lammps flag from state once we converted to Json
        del state["Lammps_data"]
        state["Json_from_Lammps_data"] = None
        # Set json name in state to the one specified for the converted Lammps data
        state["choice_json"] = self.Lammps_to_json_name
        return {**state,
                "Lammps_to_json_returncode": result.returncode,
                "Lammps_to_json_stdout": result.stdout,
                "Lammps_to_json_stderr": result.stderr}

    def _prepare_HOSS_batch_runs(self, state: HOSSState) -> HOSSState:
        """Prepare Hoss batch runs, given details in config file."""

        # Update config file according to task
        if state.get("batch_job_creation_attempts", 0) == 0:
            # Read config file and pass to llm
            with open(f"{self.src_dir}/{self.config_name}.yaml", "r") as f:
                yaml_text = f.read()

            llm_yaml_text = self.update_config_file.invoke({
                "simulation_task": state["simulation_task"],
                "config_content": yaml_text})
        else:
            with open(f"{self.src_dir}/{self.config_name_temp}.yaml", "r") as f:
                yaml_text = f.read()

            llm_yaml_text = self.repeat_update_config_file.invoke({
                "simulation_task": state["simulation_task"],
                "batch_creation_error": state["batch_create_stderr"],
                "config_content": yaml_text})

        # Write content to yaml file
        with open(f"{self.src_dir}/{self.config_name_temp}.yaml", "w") as f:
            f.write(llm_yaml_text)

        # Set up batch runs in serial based on config file
        print("Setting up batch jobs, including meshes, ICs, BCs.")
        result = subprocess.run(["python", \
            f"{self.src_dir}/{HOSSSourceFiles.batch_job_creator_name}.py", \
            "--config_name", self.config_name_temp], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return {**state,
                "batch_job_creation_attempts": state.get("batch_job_creation_attempts", 0) + 1,
                "batch_create_returncode": result.returncode,
                "batch_create_stdout": result.stdout,
                "batch_create_stderr": result.stderr}

    def _HOSS_batch_setup_success(self, state: HOSSState) -> str:
        """Conditional to decide if we need to rerun batch job creation or
        move on to run HOSS."""

        if state["batch_job_creation_attempts"] >= self.bjc_attempts_max:
            print("Batch creation failed, max fix attempts reached. Exiting.")
            return "batch_job_creation_failed"
        if state['batch_create_returncode'] == 0:
            return "batch_job_creation_success"
        return  "batch_job_creation_repeat"

    def _run_HOSS(self, state: HOSSState) -> HOSSState:
        """Run parallel batches of HOSS simulations in run's output directory"""
        with open(f"{self.src_dir}/{self.config_name_temp}.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Display info for batch that is being run with HOSS
        sample_idcs, fracture_idcs = get_sample_frac_idcs(config)
        print(f"Running HOSS run batch jobs using {len(fracture_idcs)} "
              f"fracture files from folder {config['fracture_folder_name']}\n"
              f"and material samples with indeces {sample_idcs} from json "
              f"file {config['materials_json_name']}.")

        # Update json_name in state if not done already (from Lammps conversion)
        if "choice_json" not in state.keys():
            state["choice_json"] = config['materials_json_name']

        # Specify HOSS run directory within workspace
        if config["run_output_dir_name"] is None:
            raise RuntimeError("Need to specify a name for output "
                               "directory in the config file.")
            ###
            # double-check this; probably local vs global path issue
            #elif self.workspace != config["work_dir_name"]:
            #    print(self.workspace)
            #    print(config["work_dir_name"])
            #    exit()
            #    raise RuntimeError("Work directory specified in config file "
            #                    "currently needs to agree with HOSS agent's "
            #                    "workspace.")
            # # #
        else:
            HOSS_run_dir = \
                self.workspace + '/' + config["run_output_dir_name"]
        result = subprocess.run(["bash", HOSSSourceFiles.batch_run_name],
                                cwd=HOSS_run_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        return {**state,
                "HOSS_run_dir": HOSS_run_dir,
                "fracture_idcs": fracture_idcs,
                "material_idcs": sample_idcs,
                "batch_run_returncode": result.returncode,
                "batch_run_stdout": result.stdout,
                "batch_run_stderr": result.stderr}

    def _post_HOSS_steps(self, state: HOSSState) -> str:
        """Decide what to do after HOSS simulation"""
        if state["batch_create_returncode"] != 0:
            return "batch_run_failed"
        if "build_metrics" in state or "analyze_metrics" in state:
            print("Successfully ran HOSS. Moving on to metric building/analysis.")
            return "run_post_analysis"
        print("Successfully ran HOSS. Exiting here.")
        return "task_run_HOSS_only"

    def _set_metric_folder(self, state: HOSSState) -> HOSSState:
        """Locate output folder in which we want to do metrics analysis"""
        # Either have done HOSS run, in which case we consider metrics from
        # that run. Or check for folder specified in simulation task.
        postprocessing_dir = None
        if "HOSS_run_dir" in state:
            postprocessing_dir = state["HOSS_run_dir"]
        else:
            # Assuming postprocessing directory sits in workspace
            pp_invoke = self.check_output_chain.invoke({ \
                "simulation_task": state["simulation_task"]})
            pp_local_dir = json.loads(pp_invoke)["Post-processing output"]
            if pp_local_dir is not None:
                postprocessing_dir = self.workspace + '/' + pp_local_dir
        if postprocessing_dir is not None:
            state["postprocessing_dir"] = postprocessing_dir
        return state

    def _found_metric_folder(self, state: HOSSState) -> str:
        """Conditional for whether we found metric/HOSS run output folder"""
        if "postprocessing_dir" in state:
            return "output_for_metrics_found"
        print("Tasked to work on metrics, but output folder not found.")
        return "output_for_metrics_not_found"

    def _run_metrics(self, state: HOSSState) -> HOSSState:
        """Run postprocessing files to create material metrics"""
        # Check according to task and existing output if we need to run
        # post-processing script.
        pp_invoke = self.check_postprocessing_chain.invoke({ \
            "simulation_task": state["simulation_task"]})
        run_postprocessing = \
            json.loads(pp_invoke)["Need to run post-processing"]

        if not run_postprocessing:
            run_postprocessing = ( \
                not os.path.exists(state['postprocessing_dir'] + "/" + "pp_results")
                or not os.path.exists(state['postprocessing_dir'] + "/" + "results")
            )

        if run_postprocessing:
            print("Running post-processing, including metrics, for "
                  f"folder {state['postprocessing_dir']}")
            result = subprocess.run(["bash", HOSSSourceFiles.run_postprocessing_name],
                                    cwd=state['postprocessing_dir'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

            if result.returncode != 0:
                ###TODO: in-graph exception handling
                raise RuntimeError("Error running post-processing within HOSS "
                                f"agent, with error message: {result.stderr}")
        return state

    def _analyze_metrics_or_stop(self, state: HOSSState) -> str:
        """Conditional on whether to stop after post-processing,
        or writing an analysis."""
        run_analysis = \
                json.loads(self.check_analysis_chain.invoke({ \
                    "simulation_task": state["simulation_task"]})) \
                    ["Need to run post-analysis"]
        if run_analysis:
            return "metric_analysis"
        print("Performed required tasks; skipping metric analysis.")
        return "no_metric_analysis"

    def _analyze_metrics(self, state: HOSSState) -> HOSSState:
        """Take HOSS output metrics and analyze them."""
        # Assuming the post-processing folder has been built using a HOSS run
        # performed with the HOSSSourceFiles, which should copy over the
        # sample metric loading script
        analysis_name = HOSSSourceFiles.load_metrics_name \
            + '_' + self.temp_agent_name + '.py'

        # Build metric analysis file according to task
        if state.get("analysis_attempts", 0) == 0:
            print("Running analysis on post-processing metrics.")
            # Read analysis file and pass to llm
            with open(state['postprocessing_dir'] + "/"
                    + HOSSSourceFiles.load_metrics_name + '.py', "r") as m_file:
                metric_script = m_file.read()

            llm_py_text = self.build_analysis_script.invoke({ \
                "simulation_task": state["simulation_task"], \
                "output_dir": state['postprocessing_dir'], \
                "metrics_script": metric_script})
        else:
            with open(state['postprocessing_dir'] + "/"
                      + analysis_name, "r") as m_file:
                metric_script = m_file.read()

            llm_py_text = self.repeat_build_analysis_script.invoke({ \
                "simulation_task": state["simulation_task"], \
                "output_dir": state['postprocessing_dir'], \
                "metrics_script": metric_script,
                "analysis_error": state["analysis_stderr"]})

        # Write content to yaml file
        with open(state['postprocessing_dir'] + "/" + analysis_name, "w") as f:
            f.write(llm_py_text)

        # Run analysis
        result = subprocess.run(["python3", analysis_name], \
            cwd=state['postprocessing_dir'], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return {**state,
                "analysis_attempts": state.get("analysis_attempts", 0) + 1,
                "analysis_returncode": result.returncode,
                "analysis_stdout": result.stdout,
                "analysis_stderr": result.stderr}

    def _analysis_success(self, state: HOSSState) -> str:
        """Check if analysis ran without errors."""
        if state["analysis_attempts"] >= self.analysis_attempts_max:
            print("Analysis run failed, max fix attempts reached. Exiting.")
            return "analysis_failed"
        if state['analysis_returncode'] == 0:
            print("Successfully performed analysis. "
                  f"Output:\n\n {state['analysis_stdout']}")
            return "analysis_success"
        return  "analysis_repeat"

    def _build_graph(self):
        g = StateGraph(HOSSState)

        # Add graph nodes
        # Input task and data related nodes
        self.add_node(g, self._check_input_task)
        self.add_node(g, self._convert_LAMMPS_data)
        # HOSS run related nodes
        self.add_node(g, self._prepare_HOSS_batch_runs)
        self.add_node(g, self._run_HOSS)
        # Metrics related nodes
        self.add_node(g, self._set_metric_folder)
        self.add_node(g, self._run_metrics)
        self.add_node(g, self._analyze_metrics)

        # Set entry point within graph
        g.set_entry_point("_check_input_task")

        # Edge fork: set decision between running Hoss vs postprocessing
        # Running HOSS decision also includes check on existing Hoss pipeline
        # Also stop here if agent thinks task contains neither Hoss run
        # nor metric analysis
        g.add_conditional_edges("_check_input_task",
                                self._simulate_or_analyze,
            {
                "convert_Lammps_data": "_convert_LAMMPS_data",
                "check_HOSS_pipeline_incomplete": END,
                "check_HOSS_pipeline_complete": "_prepare_HOSS_batch_runs",
                "run_post_analysis": "_set_metric_folder",
                "unclear_simulation_task": END,
            },
        )

        # May first need to convert Lammps data, then go back to input node
        g.add_edge("_convert_LAMMPS_data", "_check_input_task")
      
        # After preparing batch runs, check if there was an issue, and quit
        # quit after a few attempts, or in case of success move on to run Hoss
        g.add_conditional_edges("_prepare_HOSS_batch_runs",
                                self._HOSS_batch_setup_success,
            {
                "batch_job_creation_repeat": "_prepare_HOSS_batch_runs",
                "batch_job_creation_success": "_run_HOSS",
                "batch_job_creation_failed": END,
            },
        )

        # Edge fork: after running HOSS, either stop or do analysis
        g.add_conditional_edges("_run_HOSS",
                                self._post_HOSS_steps,
            {
                "run_post_analysis": "_set_metric_folder",
                "batch_run_failed": END,
                "task_run_HOSS_only": END,
            },
        )

        # Edge fork: when doing analysis, try to locate output folder
        g.add_conditional_edges("_set_metric_folder",
                                self._found_metric_folder,
            {
                "output_for_metrics_found": "_run_metrics",
                "output_for_metrics_not_found": END
            },
        )

        # Edge fork: after finding output folder and (possibly) running
        # postprocessing, optionally do analysis
        g.add_conditional_edges("_run_metrics",
                                self._analyze_metrics_or_stop,
            {
                "metric_analysis": "_analyze_metrics",
                "no_metric_analysis": END,
            },
        )

        g.add_conditional_edges("_analyze_metrics",
                                self._analysis_success,
            {
                "analysis_repeat": "_analyze_metrics",
                "analysis_success": END,
                "analysis_failed": END,
            },
        )

        return g.compile(checkpointer=self.checkpointer)

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        summarize: bool | None = None,
        recursion_limit: int = 1000,
        **_,
    ) -> str:
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        # HOSS state needs to include simulation task
        if "simulation_task" not in inputs:
            raise KeyError(
                "'simulation_task' is required arguments"
            )

        return self._action.invoke(inputs, config)


def get_sample_frac_idcs(config):
    """Helper function to build material sample and fracture indeces for
    correspinding files to be loaded from data folder."""
    sample_idcs = config["sample_idcs"]
    if sample_idcs is None:
        sample_total_num = config["sample_total_num"]
        sample_start_num = config["sample_start_num"]
        sample_idcs = [i for i in range(sample_start_num,
                                        sample_total_num + sample_start_num)]
    else:
        sample_total_num = len(sample_idcs)

    fracture_idcs = config["fracture_idcs"]
    if fracture_idcs is None:
        frac_total_num = config["frac_total_num"]
        frac_start_num = config["frac_start_num"]
        fracture_idcs = [i for i in range(frac_start_num,
                                        frac_start_num + frac_total_num)]
    else:
        frac_total_num = len(fracture_idcs)
    return sample_idcs, fracture_idcs
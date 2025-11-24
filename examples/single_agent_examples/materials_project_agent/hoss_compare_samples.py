"""Use Hoss to compare a set of metal alloys"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

#from ursa.agents import ExecutionAgent, LammpsAgent
from ursa.agents import HOSSAgent

model = "gpt-5"

llm = ChatOpenAI(model=model, timeout=None, max_retries=2)

workspace = "./workdir_HOSS"
# Make sure to have folder including HOSS setup source files
# Also need `data` folder in directory including materials, fracture folders
hoss_src_dir = "./src_HOSS"

hoss_only = False
metrics_only = False
hoss_and_metrics = True
hoss_from_lammps = False

wf = HOSSAgent(llm, workspace=workspace, hoss_src_dir=hoss_src_dir,
               max_tokens=50000, plot_agent_graph=False)

# Simple node testing
simulation_task = ""
if hoss_only:
    simulation_task = ("Carry out a HOSS simulation including two material "
                       "samples from the materials json file and two fracture "
                       "samples.")
elif hoss_and_metrics:
    simulation_task = ("Carry out a HOSS simulation including three material "
                       "samples from the materials json file and three fracture "
                       "samples. Futher, perform post-processing and analyze "
                       "the resulting metrics. Which material is the best?")
elif metrics_only:
    simulation_task = ("A HOSS simulation has been carried out and can be found "
                       "in the folder `Lammps_to_HOSS_output`. Given this "
                       "output, perform postprocessing and an analysis on the "
                       "output metrics.")
elif hoss_from_lammps:

    #simulation_task = ("Carry out a HOSS simulation including two material "
    #                   "samples from the materials json file and two fracture "
    #                   "samples. For this purpose, first convert the lammps "
    #                   "output data.")
    simulation_task = ("Carry out a HOSS simulation including a material "
                       "sample from the Json file Lammps_to_Json.json, with "
                       "6 different fracture patterns. For the folder output, "
                       "Use the name `Lammps_to_HOSS_output`.")

final_HOSS_state = wf.invoke(simulation_task=simulation_task)

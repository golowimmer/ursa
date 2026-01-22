# __init__.py

import importlib
from typing import Any, Dict, Tuple, TYPE_CHECKING

# Map public names to (module, attribute) for lazy loading
_lazy_attrs: Dict[str, Tuple[str, str]] = {
    # acquisition_agents
    "ArxivAgent": (".acquisition_agents", "ArxivAgent"),
    "OSTIAgent": (".acquisition_agents", "OSTIAgent"),
    "WebSearchAgent": (".acquisition_agents", "WebSearchAgent"),

    # arxiv_agent
    "ArxivAgentLegacy": (".arxiv_agent", "ArxivAgentLegacy"),
    "PaperMetadata": (".arxiv_agent", "PaperMetadata"),
    "PaperState": (".arxiv_agent", "PaperState"),

    # base
    "BaseAgent": (".base", "BaseAgent"),
    "BaseChatModel": (".base", "BaseChatModel"),

    # chat_agent
    "ChatAgent": (".chat_agent", "ChatAgent"),
    "ChatState": (".chat_agent", "ChatState"),

    # code_review_agent
    "CodeReviewAgent": (".code_review_agent", "CodeReviewAgent"),
    "CodeReviewState": (".code_review_agent", "CodeReviewState"),

    # execution_agent
    "ExecutionAgent": (".execution_agent", "ExecutionAgent"),
    "ExecutionState": (".execution_agent", "ExecutionState"),

    # hypothesizer_agent
    "HypothesizerAgent": (".hypothesizer_agent", "HypothesizerAgent"),
    "HypothesizerState": (".hypothesizer_agent", "HypothesizerState"),

    # hoss_agent
    "HOSSAgent": (".hoss_agent", "HOSSAgent"),
    "HOSSState": (".hoss_agent", "HOSSState"),

    # lammps_agent
    "LammpsAgent": (".lammps_agent", "LammpsAgent"),
    "LammpsState": (".lammps_agent", "LammpsState"),

    # mp_agent
    "MaterialsProjectAgent": (".mp_agent", "MaterialsProjectAgent"),

    # planning_agent
    "PlanningAgent": (".planning_agent", "PlanningAgent"),
    "PlanningState": (".planning_agent", "PlanningState"),

    # rag_agent
    "RAGAgent": (".rag_agent", "RAGAgent"),
    "RAGState": (".rag_agent", "RAGState"),

    # recall_agent
    "RecallAgent": (".recall_agent", "RecallAgent"),

    # websearch_agent (legacy)
    "WebSearchAgentLegacy": (".websearch_agent", "WebSearchAgentLegacy"),
    "WebSearchState": (".websearch_agent", "WebSearchState"),
}

__all__ = list(_lazy_attrs.keys())


def __getattr__(name: str) -> Any:
    """Dynamically import attributes on first access.

    This avoids importing all agent modules at package import time,
    so a failure in one agent does not prevent using others.
    """
    try:
        module_name, attr_name = _lazy_attrs[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    # Cache the loaded attribute so subsequent access is fast
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Include lazy attributes in dir(package)
    return sorted(list(globals().keys()) + list(__all__))


# Help type checkers and IDEs see the real symbols without importing at runtime
if TYPE_CHECKING:
    from .acquisition_agents import (
        ArxivAgent,
        OSTIAgent,
        WebSearchAgent,
    )
    from .arxiv_agent import (
        ArxivAgentLegacy,
        PaperMetadata,
        PaperState,
    )
    from .base import (
        BaseAgent,
        BaseChatModel,
    )
    from .chat_agent import (
        ChatAgent,
        ChatState,
    )
    from .code_review_agent import (
        CodeReviewAgent,
        CodeReviewState,
    )
    from .execution_agent import (
        ExecutionAgent,
        ExecutionState,
    )
    from .hypothesizer_agent import (
        HypothesizerAgent,
        HypothesizerState,
    )
    from .lammps_agent import (
        LammpsAgent,
        LammpsState,
    )
    from .mp_agent import MaterialsProjectAgent
    from .planning_agent import (
        PlanningAgent,
        PlanningState,
    )
    from .rag_agent import (
        RAGAgent,
        RAGState,
    )
    from .recall_agent import RecallAgent
    from .websearch_agent import (
        WebSearchAgentLegacy,
        WebSearchState,
    )


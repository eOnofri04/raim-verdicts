"""RAIM: Robust Aggregation of Inexpensive Models -- evaluation toolkit.

A small library for running a panel of cheap LLM judges over hallucination
detection benchmarks, with an on-box readout of the panel's cost/reliability
frontier.
"""
from .tasks import TaskSpec, Instance, build_instances, synthetic_instances, TASKS
from .prompts import (build_prompt, parse_verdict, FORMATS, PARAPHRASES,
                      ALL_CONDITIONS, DEF_ON_VARIANTS, DEFINITIONS,
                      FRAME_CONDITIONS, resolve_frame)
from .backends import make_backend
from .panel import PANEL, run_panel, Arm

__all__ = [
    "TaskSpec", "Instance", "build_instances", "synthetic_instances", "TASKS",
    "build_prompt", "parse_verdict", "FORMATS", "PARAPHRASES",
    "ALL_CONDITIONS", "DEF_ON_VARIANTS", "DEFINITIONS",
    "FRAME_CONDITIONS", "resolve_frame",
    "make_backend", "PANEL", "run_panel", "Arm",
]

"""Package marker for `configs` and convenient PROFILES import.

This file makes `configs` importable as a package so code can do
`import configs.llm_profiles` or `from configs import PROFILES`.
"""
from .llm_profiles import PROFILES  # re-export for convenience

__all__ = ["PROFILES"]

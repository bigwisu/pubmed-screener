"""
pubmed_screener/presets.py — Standard PICO prompt templates and screening schemas.
"""

from typing import Dict, Any


def format_pico_prompt(topic: str) -> str:
    """
    Format a standard high-recall PICO screening instruction for PubMed-Screener-420M.

    Args:
        topic: Clinical topic or systematic review research question.

    Returns:
        Formatted PICO instruction prompt.
    """
    return f"Does this study provide potentially relevant evidence on {topic.strip()}?"


def build_noul_question(instruction: str) -> Dict[str, Any]:
    """
    Build single binary proposition question schema for Laya engine.

    Args:
        instruction: Screening question or PICO objective.

    Returns:
        Laya question schema dictionary.
    """
    return {
        "relevance": {
            "type": "noul",
            "instructions": instruction.strip(),
        }
    }

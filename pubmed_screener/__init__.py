"""
pubmed_screener — Calibrated Small Language Model decision architecture
for Stage-1 biomedical systematic review citation screening.
"""

from .screener import PubMedScreener

OPERATIONAL_THRESHOLD = 0.028
DEFAULT_MODEL_ID = "bigwisu/PubMed-Screener-420M"

__all__ = ["PubMedScreener", "OPERATIONAL_THRESHOLD", "DEFAULT_MODEL_ID"]
__version__ = "1.0.0"

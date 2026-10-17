"""
pubmed_screener/screener.py — High-level screening inference interface.
"""

import os
import sys
from typing import Dict, List, Optional, Union, Any

# Ensure local laya submodule is in sys.path if not globally installed
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAYA_SUBMODULE = os.path.join(_PARENT_DIR, "laya")
if os.path.exists(_LAYA_SUBMODULE) and _LAYA_SUBMODULE not in sys.path:
    sys.path.insert(0, _LAYA_SUBMODULE)

try:
    import laya
except ImportError:
    laya = None

OPERATIONAL_THRESHOLD = 0.028
DEFAULT_MODEL_ID = "bigwisu/PubMed-Screener-420M"


class PubMedScreener:
    """
    Calibrated Small Language Model decision interface for systematic review screening.
    """

    def __init__(
        self,
        model_id_or_path: str = DEFAULT_MODEL_ID,
        threshold: float = OPERATIONAL_THRESHOLD,
        device: Optional[str] = None,
        fast: bool = False,
    ):
        """
        Initialize PubMedScreener.

        Args:
            model_id_or_path: Hugging Face repo ID or local checkpoint directory.
            threshold: Decision threshold theta* (default: 0.028).
            device: Compute device ('cuda', 'mps', 'cpu', or None for auto).
            fast: Enable TileLang / GPU fast inference path.
        """
        if laya is None:
            raise ImportError(
                "Laya decision framework is required. Ensure laya submodule is initialized."
            )
        self.threshold = float(threshold)
        self.model_id_or_path = model_id_or_path
        self.agent = laya.load(model_id_or_path, device=device, fast=fast)

    def score(
        self,
        review_objective: str,
        title: str,
        abstract: str,
    ) -> float:
        """
        Compute continuous relevance score for a candidate title and abstract.

        Args:
            review_objective: Natural language PICO eligibility prompt.
            title: Candidate paper title.
            abstract: Candidate paper abstract text.

        Returns:
            Continuous relevance score in range [0.0, 1.0].
        """
        # Format candidate document state
        state = f"Title: {title.strip()}\nAbstract: {abstract.strip()}"

        # Single binary proposition question
        question = {
            "relevance": {
                "type": "noul",
                "instructions": review_objective.strip(),
            }
        }

        response = self.agent.predict(state=state, question=question)

        # Extract predicted probability
        if isinstance(response, dict):
            rel = response.get("relevance", {})
            if isinstance(rel, dict):
                return float(rel.get("probability", rel.get("score", 0.0)))
            if isinstance(rel, (int, float)):
                return float(rel)
        return float(getattr(response, "probability", 0.0))

    def screen(
        self,
        review_objective: str,
        title: str,
        abstract: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classify candidate citation as INCLUDE or EXCLUDE at operational threshold.

        Args:
            review_objective: Natural language PICO eligibility prompt.
            title: Candidate paper title.
            abstract: Candidate paper abstract text.
            threshold: Override threshold (default: self.threshold = 0.028).

        Returns:
            Dictionary with score, decision ('INCLUDE' | 'EXCLUDE'), and threshold.
        """
        thr = self.threshold if threshold is None else float(threshold)
        relevance_score = self.score(review_objective, title, abstract)
        decision = "INCLUDE" if relevance_score >= thr else "EXCLUDE"

        return {
            "score": round(relevance_score, 4),
            "decision": decision,
            "included": relevance_score >= thr,
            "threshold": thr,
        }

    def screen_batch(
        self,
        review_objective: str,
        records: List[Dict[str, str]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Screen multiple candidate records.

        Args:
            review_objective: Natural language PICO eligibility prompt.
            records: List of dictionaries with 'title' and 'abstract' keys (and optional 'pmid').
            threshold: Decision threshold override.

        Returns:
            List of screening result dictionaries.
        """
        thr = self.threshold if threshold is None else float(threshold)
        results = []
        for rec in records:
            res = self.screen(
                review_objective=review_objective,
                title=rec.get("title", ""),
                abstract=rec.get("abstract", ""),
                threshold=thr,
            )
            if "pmid" in rec:
                res["pmid"] = rec["pmid"]
            results.append(res)
        return results

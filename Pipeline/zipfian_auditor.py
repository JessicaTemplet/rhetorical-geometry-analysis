"""
veritas_memoria/analysis/zipfian_auditor.py

Zipfian Efficiency Auditor — drift-from-baseline vocabulary health monitor.

Instead of comparing against the theoretical Zipf power law (rank × freq = const),
this module establishes the zone's *own* Zipfian profile at a known-good state and
flags deviations from that personalized baseline.  A system working in a technical
domain will legitimately skew toward rare terminology; the auditor treats that skew
as the reference point, not a defect.

Key signals
-----------
exponent_delta (Δα):
    positive  →  vocabulary becoming more concentrated (potential rigidity / fixation)
    negative  →  vocabulary dispersing    (potential fragmentation / topic scatter)

vocabulary_turnover_rate:
    high  →  core vocabulary churning rapidly (instability in foundational concepts)
    low   →  concepts stable (flag only if sustained alongside exponent drift)

emerging_terms:
    low-frequency terms that moved into top-k (topic shift, possible injection)

vanishing_terms:
    formerly prominent terms that dropped out of top-k (concept abandonment)

Usage
-----
    auditor = ZipfianAuditor(graph, library)
    baseline = auditor.create_baseline(GraphZone.RATIONALE)
    # ... time passes, new content added ...
    report = auditor.audit(GraphZone.RATIONALE, baseline)
    if report.has_drift():
        print(report.summary())
"""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Counter as CounterType, Dict, List, Optional, Tuple

import numpy as np


# ── Stop-word list (minimal, no external dependency) ──────────────────────────

_DEFAULT_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "what", "which", "who", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "same", "so", "than", "too", "very",
    "just", "can", "also", "as", "if", "then", "because", "while", "get",
    "got", "one", "two", "new", "use", "used", "using", "need", "needs",
    "make", "made", "made", "way", "now", "see", "look", "work", "works",
})

# Drift class boundaries (applied to global_drift_score 0-1)
_DRIFT_STABLE        = 0.15
_DRIFT_MILD          = 0.35
_DRIFT_SIGNIFICANT   = 0.65

# Defaults
DEFAULT_TOP_K                  = 100
DEFAULT_EXPONENT_DRIFT_THRESH  = 0.30
DEFAULT_TURNOVER_THRESH        = 0.40
DEFAULT_MIN_TERM_LENGTH        = 3
DEFAULT_MIN_TOKENS             = 20   # below this, skip exponent fit


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ZipfianProfile:
    """
    Snapshot of a zone's vocabulary rank-frequency distribution.

    Persist this (via to_dict / from_dict) to use as a drift baseline later.
    The exponent α is estimated by OLS on the log-log rank-frequency plot.
    """
    zone: str
    timestamp: str
    node_count: int
    total_tokens: int
    unique_tokens: int
    exponent: float           # α  (positive; pure Zipf ≈ 1.0)
    fit_intercept: float      # log(C) from log(f) = log(C) − α·log(r)
    fit_r2: float             # goodness-of-fit in log-log space (0–1)
    top_k: int                # how many terms are stored in top_terms
    top_terms: List[Tuple[str, int, int]]   # (term, rank_1based, count)
    sparse_fit: bool = False  # True when tokens < min_tokens; exponent unreliable

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "timestamp": self.timestamp,
            "node_count": self.node_count,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "exponent": self.exponent,
            "fit_intercept": self.fit_intercept,
            "fit_r2": self.fit_r2,
            "top_k": self.top_k,
            "top_terms": self.top_terms,
            "sparse_fit": self.sparse_fit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZipfianProfile":
        return cls(
            zone=d["zone"],
            timestamp=d["timestamp"],
            node_count=d["node_count"],
            total_tokens=d["total_tokens"],
            unique_tokens=d["unique_tokens"],
            exponent=d["exponent"],
            fit_intercept=d["fit_intercept"],
            fit_r2=d["fit_r2"],
            top_k=d["top_k"],
            top_terms=[tuple(t) for t in d["top_terms"]],
            sparse_fit=d.get("sparse_fit", False),
        )


@dataclass
class TermShift:
    """
    Movement record for a single term between baseline and current state.

    direction values:
        "emerging"  — term is new (not in baseline vocabulary at all)
        "rising"    — term was present, moved to a lower rank number (more frequent)
        "vanishing" — term has disappeared from current vocabulary
        "falling"   — term was present, moved to a higher rank number (less frequent)
        "stable"    — minor displacement within tolerance
    """
    term: str
    baseline_rank: Optional[int]   # None if the term is new
    current_rank: Optional[int]    # None if the term vanished
    baseline_count: int
    current_count: int
    rank_delta: Optional[int]      # current_rank − baseline_rank; negative = rising

    @property
    def direction(self) -> str:
        if self.baseline_rank is None:
            return "emerging"
        if self.current_rank is None:
            return "vanishing"
        if self.rank_delta is not None and self.rank_delta < -20:
            return "rising"
        if self.rank_delta is not None and self.rank_delta > 20:
            return "falling"
        return "stable"

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "baseline_rank": self.baseline_rank,
            "current_rank": self.current_rank,
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
            "rank_delta": self.rank_delta,
            "direction": self.direction,
        }


@dataclass
class DriftReport:
    """
    Result of comparing a zone's current vocabulary profile against a baseline.
    """
    zone: str
    baseline_timestamp: str
    current_timestamp: str
    baseline_node_count: int
    current_node_count: int

    # Exponent
    exponent_baseline: float
    exponent_current: float
    exponent_delta: float         # current − baseline; + = more concentrated
    exponent_drift_flag: bool

    # Fit quality
    fit_r2_baseline: float
    fit_r2_current: float

    # Vocabulary volume
    total_tokens_baseline: int
    total_tokens_current: int
    unique_tokens_baseline: int
    unique_tokens_current: int

    # Vocabulary stability
    vocabulary_turnover_rate: float   # Jaccard distance on top-k sets (0–1)
    turnover_flag: bool

    # Term movements
    emerging_terms: List[TermShift]
    vanishing_terms: List[TermShift]
    top_rank_displacements: List[TermShift]   # largest |rank_delta|

    # Summary score
    global_drift_score: float    # 0–1
    drift_class: str             # "stable" | "mild_drift" | "significant_drift" | "restructuring"
    recommendation: str

    def has_drift(self) -> bool:
        return self.exponent_drift_flag or self.turnover_flag

    def summary(self) -> str:
        lines = [
            f"Zipfian Drift Report — zone={self.zone}",
            f"  baseline:  {self.baseline_timestamp}  ({self.baseline_node_count} nodes)",
            f"  current:   {self.current_timestamp}  ({self.current_node_count} nodes)",
            f"  exponent:  α_base={self.exponent_baseline:.3f}  α_now={self.exponent_current:.3f}"
            f"  Δ={self.exponent_delta:+.3f}"
            + ("  [FLAGGED]" if self.exponent_drift_flag else ""),
            f"  turnover:  {self.vocabulary_turnover_rate:.1%}"
            + ("  [FLAGGED]" if self.turnover_flag else ""),
            f"  drift score: {self.global_drift_score:.3f}  ({self.drift_class})",
        ]
        if self.emerging_terms:
            lines.append(
                f"  emerging:  {', '.join(t.term for t in self.emerging_terms[:5])}"
            )
        if self.vanishing_terms:
            lines.append(
                f"  vanishing: {', '.join(t.term for t in self.vanishing_terms[:5])}"
            )
        lines.append(f"  → {self.recommendation}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "baseline_node_count": self.baseline_node_count,
            "current_node_count": self.current_node_count,
            "exponent_baseline": self.exponent_baseline,
            "exponent_current": self.exponent_current,
            "exponent_delta": self.exponent_delta,
            "exponent_drift_flag": self.exponent_drift_flag,
            "fit_r2_baseline": self.fit_r2_baseline,
            "fit_r2_current": self.fit_r2_current,
            "total_tokens_baseline": self.total_tokens_baseline,
            "total_tokens_current": self.total_tokens_current,
            "unique_tokens_baseline": self.unique_tokens_baseline,
            "unique_tokens_current": self.unique_tokens_current,
            "vocabulary_turnover_rate": self.vocabulary_turnover_rate,
            "turnover_flag": self.turnover_flag,
            "emerging_terms": [t.to_dict() for t in self.emerging_terms],
            "vanishing_terms": [t.to_dict() for t in self.vanishing_terms],
            "top_rank_displacements": [t.to_dict() for t in self.top_rank_displacements],
            "global_drift_score": self.global_drift_score,
            "drift_class": self.drift_class,
            "recommendation": self.recommendation,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class ZipfianAuditor:
    """
    Establishes a personalized Zipfian baseline for a memory zone and later
    audits deviation from that baseline rather than from the theoretical law.

    Parameters
    ----------
    graph
        VeritasMemoria Graph instance (provides zone → node_id enumeration).
    library
        ProductionMemoryLibrary instance (provides node content via retrieve()).
    top_k
        Number of most-frequent terms to track in profiles and compare.
    stopwords
        Optional custom stop-word set; falls back to the built-in English list.
    exponent_drift_threshold
        Minimum |Δα| that raises exponent_drift_flag (default 0.30).
    turnover_threshold
        Minimum Jaccard distance in top-k vocabulary that raises turnover_flag
        (default 0.40).
    min_term_length
        Minimum character length for a token to be counted (default 3).
    min_tokens
        Minimum total tokens required before fitting an exponent.  Below this
        the profile is marked sparse_fit=True and α is unreliable.
    """

    def __init__(
        self,
        graph,
        library,
        top_k: int = DEFAULT_TOP_K,
        stopwords: Optional[frozenset] = None,
        exponent_drift_threshold: float = DEFAULT_EXPONENT_DRIFT_THRESH,
        turnover_threshold: float = DEFAULT_TURNOVER_THRESH,
        min_term_length: int = DEFAULT_MIN_TERM_LENGTH,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ) -> None:
        self.graph = graph
        self.library = library
        self.top_k = top_k
        self._stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
        self.exponent_drift_threshold = exponent_drift_threshold
        self.turnover_threshold = turnover_threshold
        self.min_term_length = min_term_length
        self.min_tokens = min_tokens

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_baseline(self, zone) -> ZipfianProfile:
        """
        Snapshot the current vocabulary distribution for *zone*.

        Persist the returned ZipfianProfile (via to_dict) and pass it back
        to audit() later.
        """
        gz, zone_str = self._resolve_zone(zone)
        counter, total_tokens, node_count = self._extract_terms(gz)
        alpha, intercept, r2, sparse = self._fit_exponent(counter)
        top_terms = self._top_terms(counter)

        return ZipfianProfile(
            zone=zone_str,
            timestamp=self._now(),
            node_count=node_count,
            total_tokens=total_tokens,
            unique_tokens=len(counter),
            exponent=alpha,
            fit_intercept=intercept,
            fit_r2=r2,
            top_k=len(top_terms),
            top_terms=top_terms,
            sparse_fit=sparse,
        )

    def audit(self, zone, baseline: ZipfianProfile) -> DriftReport:
        """
        Compare the zone's current vocabulary against *baseline*.

        Returns a DriftReport describing what has changed and whether any
        drift thresholds have been crossed.
        """
        gz, zone_str = self._resolve_zone(zone)
        counter, total_tokens, node_count = self._extract_terms(gz)
        alpha, intercept, r2, sparse = self._fit_exponent(counter)
        top_terms_current = self._top_terms(counter)

        return self._build_report(
            zone_str=zone_str,
            baseline=baseline,
            current_counter=counter,
            current_total_tokens=total_tokens,
            current_node_count=node_count,
            current_alpha=alpha,
            current_r2=r2,
            current_top_terms=top_terms_current,
        )

    # ── Term extraction ────────────────────────────────────────────────────────

    def _extract_terms(
        self, gz
    ) -> Tuple[CounterType[str], int, int]:
        """
        Walk every node in *gz* and collect all tokens.

        Returns (counter, total_tokens, node_count).
        """
        from graph_types import GraphZone

        counter: CounterType[str] = Counter()
        total_tokens = 0
        node_count = 0

        zone_nodes: Dict = self.graph._adj.get(gz, {})
        for node_id in zone_nodes:
            record = self.library.retrieve(node_id)
            if record is None:
                continue
            content = record.get("content", "") or ""
            tokens = self._tokenize(content)
            counter.update(tokens)
            total_tokens += len(tokens)
            node_count += 1

        return counter, total_tokens, node_count

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, remove stop-words and short tokens."""
        tokens = re.findall(r"[a-z][a-z']*", text.lower())
        return [
            t for t in tokens
            if len(t) >= self.min_term_length and t not in self._stopwords
        ]

    # ── Zipfian fit ────────────────────────────────────────────────────────────

    def _fit_exponent(
        self, counter: CounterType[str]
    ) -> Tuple[float, float, float, bool]:
        """
        Fit a power law to the rank-frequency distribution via OLS in log-log
        space.

        Returns (alpha, log_intercept, r2, sparse_flag).
        alpha is the absolute value of the slope (positive for a Zipfian
        distribution).  sparse_flag is True when there are too few tokens to
        produce a reliable estimate.
        """
        if not counter:
            return 0.0, 0.0, 0.0, True

        freqs = sorted(counter.values(), reverse=True)
        total_tokens = sum(freqs)

        if total_tokens < self.min_tokens or len(freqs) < 5:
            # Not enough data: return minimum-effort estimate
            alpha = 0.0
            intercept = 0.0
            r2 = 0.0
            return alpha, intercept, r2, True

        ranks = np.arange(1, len(freqs) + 1, dtype=float)
        log_r = np.log(ranks)
        log_f = np.log(np.array(freqs, dtype=float))

        coeffs = np.polyfit(log_r, log_f, 1)
        alpha = -float(coeffs[0])        # slope is negative; α = |slope|
        intercept = float(coeffs[1])

        fitted = np.polyval(coeffs, log_r)
        ss_res = float(np.sum((log_f - fitted) ** 2))
        ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2 = max(0.0, min(1.0, r2))

        return alpha, intercept, r2, False

    # ── Vocabulary helpers ─────────────────────────────────────────────────────

    def _top_terms(
        self, counter: CounterType[str]
    ) -> List[Tuple[str, int, int]]:
        """Return top-k terms as (term, 1-based rank, count) tuples."""
        top = counter.most_common(self.top_k)
        return [(term, rank + 1, count) for rank, (term, count) in enumerate(top)]

    def _top_k_set(
        self, top_terms: List[Tuple[str, int, int]]
    ) -> frozenset:
        return frozenset(t[0] for t in top_terms)

    def _vocabulary_turnover(
        self,
        baseline_top: List[Tuple[str, int, int]],
        current_top: List[Tuple[str, int, int]],
    ) -> float:
        """Jaccard distance on the two top-k term sets (0 = identical, 1 = disjoint)."""
        a = self._top_k_set(baseline_top)
        b = self._top_k_set(current_top)
        if not a and not b:
            return 0.0
        union = len(a | b)
        intersection = len(a & b)
        return 1.0 - intersection / union if union > 0 else 0.0

    # ── Drift report builder ───────────────────────────────────────────────────

    def _build_report(
        self,
        zone_str: str,
        baseline: ZipfianProfile,
        current_counter: CounterType[str],
        current_total_tokens: int,
        current_node_count: int,
        current_alpha: float,
        current_r2: float,
        current_top_terms: List[Tuple[str, int, int]],
    ) -> DriftReport:
        exponent_delta = current_alpha - baseline.exponent
        exponent_drift_flag = abs(exponent_delta) > self.exponent_drift_threshold

        turnover = self._vocabulary_turnover(baseline.top_terms, current_top_terms)
        turnover_flag = turnover > self.turnover_threshold

        emerging, vanishing, displacements = self._compute_term_shifts(
            baseline_top=baseline.top_terms,
            current_top=current_top_terms,
            baseline_counter=Counter({t: c for t, _, c in baseline.top_terms}),
            current_counter=current_counter,
        )

        # Global drift score (0–1)
        exp_component = min(abs(exponent_delta) / max(self.exponent_drift_threshold, 1e-9), 1.0)
        global_drift_score = 0.50 * exp_component + 0.50 * min(turnover, 1.0)

        drift_class = self._classify_drift(global_drift_score)
        recommendation = self._build_recommendation(
            exponent_delta=exponent_delta,
            exponent_drift_flag=exponent_drift_flag,
            turnover=turnover,
            turnover_flag=turnover_flag,
            emerging=emerging,
            vanishing=vanishing,
            drift_class=drift_class,
        )

        return DriftReport(
            zone=zone_str,
            baseline_timestamp=baseline.timestamp,
            current_timestamp=self._now(),
            baseline_node_count=baseline.node_count,
            current_node_count=current_node_count,
            exponent_baseline=baseline.exponent,
            exponent_current=current_alpha,
            exponent_delta=exponent_delta,
            exponent_drift_flag=exponent_drift_flag,
            fit_r2_baseline=baseline.fit_r2,
            fit_r2_current=current_r2,
            total_tokens_baseline=baseline.total_tokens,
            total_tokens_current=current_total_tokens,
            unique_tokens_baseline=baseline.unique_tokens,
            unique_tokens_current=len(current_counter),
            vocabulary_turnover_rate=turnover,
            turnover_flag=turnover_flag,
            emerging_terms=emerging,
            vanishing_terms=vanishing,
            top_rank_displacements=displacements,
            global_drift_score=global_drift_score,
            drift_class=drift_class,
            recommendation=recommendation,
        )

    def _compute_term_shifts(
        self,
        baseline_top: List[Tuple[str, int, int]],
        current_top: List[Tuple[str, int, int]],
        baseline_counter: CounterType[str],
        current_counter: CounterType[str],
    ) -> Tuple[List[TermShift], List[TermShift], List[TermShift]]:
        """
        Categorise movement between two top-k term lists.

        Returns (emerging, vanishing, top_rank_displacements).
        """
        baseline_rank_map: Dict[str, int] = {t: r for t, r, _ in baseline_top}
        current_rank_map: Dict[str, int]  = {t: r for t, r, _ in current_top}

        baseline_set = set(baseline_rank_map)
        current_set  = set(current_rank_map)

        emerging: List[TermShift] = []
        for term in current_set - baseline_set:
            emerging.append(TermShift(
                term=term,
                baseline_rank=None,
                current_rank=current_rank_map[term],
                baseline_count=baseline_counter.get(term, 0),
                current_count=current_counter.get(term, 0),
                rank_delta=None,
            ))
        # Sort by current rank (most prominent first)
        emerging.sort(key=lambda s: s.current_rank or 99999)

        vanishing: List[TermShift] = []
        for term in baseline_set - current_set:
            vanishing.append(TermShift(
                term=term,
                baseline_rank=baseline_rank_map[term],
                current_rank=None,
                baseline_count=baseline_counter.get(term, 0),
                current_count=current_counter.get(term, 0),
                rank_delta=None,
            ))
        # Sort by baseline rank (most prominent first)
        vanishing.sort(key=lambda s: s.baseline_rank or 99999)

        # Terms present in both — sort by |rank_delta|
        displacements: List[TermShift] = []
        for term in baseline_set & current_set:
            b_rank = baseline_rank_map[term]
            c_rank = current_rank_map[term]
            delta  = c_rank - b_rank
            displacements.append(TermShift(
                term=term,
                baseline_rank=b_rank,
                current_rank=c_rank,
                baseline_count=baseline_counter.get(term, 0),
                current_count=current_counter.get(term, 0),
                rank_delta=delta,
            ))
        displacements.sort(key=lambda s: abs(s.rank_delta or 0), reverse=True)
        # Return only the most displaced
        displacements = displacements[:20]

        return emerging, vanishing, displacements

    # ── Classification & recommendation ───────────────────────────────────────

    @staticmethod
    def _classify_drift(score: float) -> str:
        if score < _DRIFT_STABLE:
            return "stable"
        if score < _DRIFT_MILD:
            return "mild_drift"
        if score < _DRIFT_SIGNIFICANT:
            return "significant_drift"
        return "restructuring"

    @staticmethod
    def _build_recommendation(
        exponent_delta: float,
        exponent_drift_flag: bool,
        turnover: float,
        turnover_flag: bool,
        emerging: List[TermShift],
        vanishing: List[TermShift],
        drift_class: str,
    ) -> str:
        if drift_class == "stable":
            return "Vocabulary distribution stable; no action needed."

        parts: List[str] = []

        if exponent_drift_flag:
            if exponent_delta > 0:
                parts.append(
                    f"Exponent rose by {exponent_delta:+.3f}: vocabulary is concentrating; "
                    "review whether topic fixation is intentional."
                )
            else:
                parts.append(
                    f"Exponent fell by {exponent_delta:+.3f}: vocabulary is dispersing; "
                    "consider whether fragmentation reflects scope creep."
                )

        if turnover_flag:
            parts.append(
                f"Top-{DEFAULT_TOP_K} vocabulary turnover is {turnover:.0%}; "
                "foundational concepts are shifting — run EpistaticGate scan to check "
                "for suppressed contradictions."
            )

        if emerging:
            terms = ", ".join(s.term for s in emerging[:5])
            parts.append(f"Emerging terms in top-k: [{terms}] — verify these reflect intended learning.")

        if vanishing:
            terms = ", ".join(s.term for s in vanishing[:5])
            parts.append(f"Vanishing terms from top-k: [{terms}] — confirm intentional deprioritisation.")

        if drift_class == "restructuring":
            parts.append(
                "Global drift score indicates a fundamental vocabulary restructuring; "
                "consider snapshotting a new baseline once the topic shift stabilises."
            )

        return "  ".join(parts) if parts else f"Drift class: {drift_class}; investigate above signals."

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _resolve_zone(zone) -> Tuple:
        """Accept GraphZone enum or plain string; return (GraphZone, str)."""
        from graph_types import GraphZone

        if isinstance(zone, str):
            gz = GraphZone(zone)
            return gz, zone
        return zone, zone.value

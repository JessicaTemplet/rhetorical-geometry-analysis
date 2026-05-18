"""
VeritasMemoria - Diagnostic Annealer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reverse-annealing probe: uses controlled heat injection to expose structural
weaknesses in the belief graph, then triages them for targeted reinforcement.

The standard AdversarialAnnealer searches for "stable lies" — configurations
that look coherent but contain circular reasoning. This module does the
opposite: it runs low-heat diagnostic passes to identify which nodes and edges
are most fragile, then classifies *why* they're fragile so the system can
shore them up appropriately.

Why fragility has two causes
-----------------------------
A node can be fragile for two structurally different reasons:

    Under-evidenced: the node has few or weak connections. It moves easily
        under perturbation because it has little anchoring mass. The belief
        is probably correct — it just hasn't been corroborated yet.
        Prescription: seek more supporting evidence. Do NOT increase edge
        weights directly; that would be confabulation.

    Contested: the node has both positive (supporting) and negative
        (contradicting) edges in roughly equal measure. It moves because it
        is genuinely pulled in two directions. The graph already senses the
        conflict at the edge level without a formal contradiction record.
        Prescription: escalate to contradiction resolution. Shoring up
        would be wrong here — it would bury a real conflict.

A third class is also surfaced:

    Epistasis candidate: a node is stable in isolation but becomes fragile
        when a specific neighbor is present. The neighbor acts as a modifier
        gene — it conditionally suppresses or amplifies the target node's
        anchoring. These are not contradictions and not under-evidenced;
        they are context-dependent relationships that should be explicitly
        encoded rather than treated as simple edges.
        Prescription: flag for conditional relationship modeling.

Algorithm
----------
1.  Build weighted adjacency via SAL's _build_weighted_adjacency().
2.  Compute per-node signed weighted degree vector at baseline.
3.  Run `rounds` perturbation rounds at temperature `heat`:
        For each round, inject multiplicative noise η ~ Uniform(-T, +T)
        to each edge weight symmetrically. Compute perturbed degree vector.
        Record displacement = |degree_perturbed - degree_baseline| per node.
4.  Aggregate displacement across rounds → mean fragility score per node.
5.  Threshold: nodes above fragility_threshold are flagged as fragile.
6.  Triage each fragile node:
        - Compute fraction of total edge mass that is negative.
        - If negative_fraction > contested_threshold → Contested.
        - Else if positive weighted degree < low_mass_threshold → Under-evidenced.
        - Epistasis check: for each fragile node with >= 2 neighbors,
          temporarily remove each neighbor and re-measure local fragility.
          If removing a specific neighbor drops fragility by more than
          epistasis_delta, that neighbor is flagged as a modifier candidate.
7.  Return DiagnosticReport with per-node findings and global stability.

Heat calibration
----------------
Temperature T controls perturbation aggressiveness:
    T = 0.05  gentle diagnostic (recommended for routine checks)
    T = 0.10  moderate stress test (default)
    T = 0.25  aggressive (may surface false positives in sparse graphs)

The global stability score is 1.0 - mean(normalized_displacement) across all
nodes. Above 0.85 = robust. Below 0.60 = systemic structural issues.

Integration
-----------
DiagnosticAnnealer reads from SALCoherenceLayer and is strictly read-only.
It never modifies edge weights, node states, or any persistent data.

    from veritas_memoria.analysis.coherence.diagnostic_annealer import DiagnosticAnnealer

    annealer = DiagnosticAnnealer(sal_layer)
    report = annealer.run_diagnostic(GraphZone.RATIONALE)

    for node in report.fragile_nodes:
        if node.fragility_class == "contested":
            # escalate to contradiction resolution
        elif node.fragility_class == "under_evidenced":
            # queue for evidence-seeking
        elif node.fragility_class == "epistasis_candidate":
            # flag for conditional relationship modeling

Relationship to AdversarialAnnealer
-------------------------------------
AdversarialAnnealer (adversarial_annealer.py) searches for stable-but-circular
configurations — it is an adversarial probe that finds where the system could
deceive itself. DiagnosticAnnealer is a structural integrity probe that finds
where the system is genuinely weak and prescribes targeted reinforcement.
They are complementary: run both before a session commit for full coverage.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Defaults — tunable at construction time
# ─────────────────────────────────────────────────────────────

# Perturbation amplitude: edge weights are multiplied by (1 + η),
# η ~ Uniform(-heat, +heat). Signs are always preserved.
DEFAULT_HEAT: float = 0.10

# Number of perturbation rounds. More rounds = more stable fragility estimate.
# 40 is sufficient for most graphs; increase to 80-100 for large zones (> 300 nodes).
DEFAULT_ROUNDS: int = 40

# Normalized displacement above this → node is flagged as fragile.
DEFAULT_FRAGILITY_THRESHOLD: float = 0.15

# Fraction of a node's total edge mass that must be negative to classify
# it as "contested" rather than "under_evidenced".
DEFAULT_CONTESTED_THRESHOLD: float = 0.30

# Maximum positive weighted degree for the "under_evidenced" base case.
# Nodes below this are considered lightly anchored; above it they have
# adequate mass but may be structurally isolated.
DEFAULT_LOW_MASS_THRESHOLD: float = 0.40

# Minimum fragility reduction when a single neighbor is removed that
# qualifies that neighbor as an epistasis modifier candidate.
DEFAULT_EPISTASIS_DELTA: float = 0.20

# Reduced round count for per-neighbor epistasis checks (speed trade-off).
_EPISTASIS_ROUNDS_DIVISOR: int = 4


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class FragileNode:
    """
    A node identified as structurally fragile by the diagnostic pass.

    fragility_score      Mean normalized L1 displacement across perturbation
                         rounds. Range [0, 1]; higher = more fragile.

    fragility_class      "under_evidenced" | "contested" | "epistasis_candidate"

    recommendation       Human-readable prescription for this node.

    positive_weight_sum  Sum of absolute positive edge weights (anchoring mass).

    negative_weight_sum  Sum of absolute negative edge weights (contest mass).

    neighbor_count       Number of connected neighbors in the adjacency.

    epistasis_modifier   node_id of the neighbor whose removal most reduces
                         fragility, only set when fragility_class is
                         "epistasis_candidate". None otherwise.
    """
    node_id:             str
    fragility_score:     float
    fragility_class:     str
    recommendation:      str
    positive_weight_sum: float
    negative_weight_sum: float
    neighbor_count:      int
    epistasis_modifier:  Optional[str] = None


@dataclass
class DiagnosticReport:
    """
    Output of a full diagnostic annealing pass over one zone.

    zone                      Zone analyzed.
    timestamp                 ISO timestamp of the run.
    heat_applied              Perturbation amplitude T used.
    rounds                    Number of perturbation rounds.
    nodes_scanned             Total nodes in the zone at analysis time.
    fragile_nodes             FragileNode list, sorted by fragility_score descending.
    lambda2_baseline          Fiedler value before any perturbation.
    lambda2_stressed          Mean Fiedler value across perturbed rounds.
    lambda2_stress_delta      lambda2_baseline - lambda2_stressed. Positive means
                              heat hurt coherence, as expected. Large delta at low
                              heat indicates a globally fragile zone.
    global_stability          1.0 - mean(normalized_displacement) across all nodes.
                              > 0.85 = robust; 0.60-0.85 = moderate concerns;
                              < 0.60 = systemic structural issues.
    contested_count           Fragile nodes classified as contested.
    under_evidenced_count     Fragile nodes classified as under-evidenced.
    epistasis_candidate_count Fragile nodes with a detected epistasis modifier.
    """
    zone:                       str
    timestamp:                  str
    heat_applied:               float
    rounds:                     int
    nodes_scanned:              int
    fragile_nodes:              List[FragileNode]
    lambda2_baseline:           float
    lambda2_stressed:           float
    lambda2_stress_delta:       float
    global_stability:           float
    contested_count:            int
    under_evidenced_count:      int
    epistasis_candidate_count:  int

    def summary(self) -> str:
        """One-line human-readable summary for logging and API responses."""
        return (
            f"Zone '{self.zone}': stability={self.global_stability:.3f}, "
            f"fragile={len(self.fragile_nodes)} nodes "
            f"(contested={self.contested_count}, "
            f"under_evidenced={self.under_evidenced_count}, "
            f"epistasis_candidates={self.epistasis_candidate_count}), "
            f"λ₂ stress delta={self.lambda2_stress_delta:.4f}"
        )

    def to_dict(self) -> dict:
        """Serializable representation for API responses."""
        return {
            "zone": self.zone,
            "timestamp": self.timestamp,
            "heat_applied": self.heat_applied,
            "rounds": self.rounds,
            "nodes_scanned": self.nodes_scanned,
            "global_stability": self.global_stability,
            "lambda2_baseline": self.lambda2_baseline,
            "lambda2_stressed": self.lambda2_stressed,
            "lambda2_stress_delta": self.lambda2_stress_delta,
            "contested_count": self.contested_count,
            "under_evidenced_count": self.under_evidenced_count,
            "epistasis_candidate_count": self.epistasis_candidate_count,
            "fragile_nodes": [
                {
                    "node_id": fn.node_id,
                    "fragility_score": fn.fragility_score,
                    "fragility_class": fn.fragility_class,
                    "recommendation": fn.recommendation,
                    "positive_weight_sum": fn.positive_weight_sum,
                    "negative_weight_sum": fn.negative_weight_sum,
                    "neighbor_count": fn.neighbor_count,
                    "epistasis_modifier": fn.epistasis_modifier,
                }
                for fn in self.fragile_nodes
            ],
        }


# ─────────────────────────────────────────────────────────────
# Diagnostic Annealer
# ─────────────────────────────────────────────────────────────

class DiagnosticAnnealer:
    """
    Read-only diagnostic probe for the VeritasMemoria belief graph.

    Uses controlled heat injection to identify structurally fragile nodes
    and classify them for targeted reinforcement. Never modifies the graph.

    Parameters
    ----------
    sal_layer : SALCoherenceLayer
        The coherence layer to read adjacency from. Must have
        _build_weighted_adjacency, _build_normalized_laplacian,
        and _compute_fiedler_value available.

    heat : float
        Perturbation amplitude T. Edge weights are multiplied by
        (1 + η) where η ~ Uniform(-heat, +heat). Signs are always
        preserved — a negative edge stays negative.

    rounds : int
        Number of perturbation rounds. More rounds → more stable fragility
        estimate at the cost of compute time.

    fragility_threshold : float
        Normalized displacement above which a node is considered fragile.

    contested_threshold : float
        Negative weight fraction above which a node is "contested" rather
        than "under_evidenced".

    low_mass_threshold : float
        Maximum positive weighted degree for the "under_evidenced" base
        classification.

    epistasis_delta : float
        Minimum fragility reduction when one neighbor is removed to flag
        that neighbor as an epistasis modifier candidate.

    rng_seed : int, optional
        Seed for the random number generator. Set for reproducible diagnostics.
    """

    def __init__(
        self,
        sal_layer,
        heat: float = DEFAULT_HEAT,
        rounds: int = DEFAULT_ROUNDS,
        fragility_threshold: float = DEFAULT_FRAGILITY_THRESHOLD,
        contested_threshold: float = DEFAULT_CONTESTED_THRESHOLD,
        low_mass_threshold: float = DEFAULT_LOW_MASS_THRESHOLD,
        epistasis_delta: float = DEFAULT_EPISTASIS_DELTA,
        rng_seed: Optional[int] = None,
    ):
        self.sal = sal_layer
        self.heat = heat
        self.rounds = rounds
        self.fragility_threshold = fragility_threshold
        self.contested_threshold = contested_threshold
        self.low_mass_threshold = low_mass_threshold
        self.epistasis_delta = epistasis_delta
        self._rng = np.random.default_rng(rng_seed)

    # ── Public API ───────────────────────────────────────────

    def run_diagnostic(self, zone) -> DiagnosticReport:
        """
        Run a full diagnostic pass over a zone.

        Scans all nodes, identifies fragile ones, triages each fragile node
        into a class with a prescription, and returns a DiagnosticReport.
        Read-only: no graph modifications.

        Parameters
        ----------
        zone : GraphZone
            The zone to analyze.

        Returns
        -------
        DiagnosticReport
            Full fragility analysis with per-node findings and global stability.
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        now = datetime.now(timezone.utc).isoformat()

        adj, _ = self.sal._build_weighted_adjacency(zone)
        nodes = sorted(adj.keys())
        n = len(nodes)

        if n < 2:
            logger.info(
                "DiagnosticAnnealer: zone '%s' has %d node(s) — "
                "insufficient for fragility analysis.",
                zone_val, n,
            )
            return DiagnosticReport(
                zone=zone_val, timestamp=now,
                heat_applied=self.heat, rounds=self.rounds,
                nodes_scanned=n, fragile_nodes=[],
                lambda2_baseline=0.0, lambda2_stressed=0.0,
                lambda2_stress_delta=0.0, global_stability=1.0,
                contested_count=0, under_evidenced_count=0,
                epistasis_candidate_count=0,
            )

        # ── Baseline ─────────────────────────────────────────
        baseline_degrees = self._weighted_degrees(adj, nodes)
        lambda2_baseline = self._compute_lambda2(adj, nodes)

        # ── Perturbation rounds ───────────────────────────────
        accumulated_displacement = np.zeros(n)
        lambda2_stressed_sum = 0.0

        for _ in range(self.rounds):
            perturbed_adj = self._perturb(adj)
            perturbed_degrees = self._weighted_degrees(perturbed_adj, nodes)
            accumulated_displacement += np.abs(perturbed_degrees - baseline_degrees)
            lambda2_stressed_sum += self._compute_lambda2(perturbed_adj, nodes)

        mean_displacement = accumulated_displacement / self.rounds
        lambda2_stressed = lambda2_stressed_sum / self.rounds

        # Normalize displacement to [0, 1] by baseline degree magnitude.
        # Prevents scale sensitivity: a node with degree 10 that moves by 2
        # is less fragile than a node with degree 0.5 that moves by 0.2.
        # Special case: isolated nodes (degree 0) are maximally fragile.
        max_degree = np.maximum(np.abs(baseline_degrees), 1e-8)
        normalized_displacement = np.clip(mean_displacement / max_degree, 0.0, 1.0)

        # Mark isolated nodes as maximally fragile (degree == 0)
        for i in range(n):
            if baseline_degrees[i] == 0:
                normalized_displacement[i] = 1.0

        # ── Identify and triage fragile nodes ─────────────────
        fragile_nodes: List[FragileNode] = []

        for i, node_id in enumerate(nodes):
            score = float(normalized_displacement[i])
            if score < self.fragility_threshold:
                continue

            pos_sum, neg_sum = self._edge_mass(adj, node_id)
            neighbor_count = len(adj.get(node_id, {}))

            fragility_class, recommendation = self._triage(
                score, pos_sum, neg_sum, neighbor_count
            )

            # Epistasis check: does removing one specific neighbor
            # dramatically reduce this node's fragility? If so, the
            # relationship is context-dependent, not a simple contradiction.
            epistasis_modifier: Optional[str] = None
            if neighbor_count >= 2:
                epistasis_modifier = self._check_epistasis(
                    adj, nodes, node_id, score
                )
                if epistasis_modifier is not None:
                    fragility_class = "epistasis_candidate"
                    recommendation = (
                        f"Fragility is strongly coupled to neighbor "
                        f"'{epistasis_modifier}'. The relationship is "
                        "context-dependent rather than a plain contradiction "
                        "or evidence gap. Consider modeling this as an "
                        "epistatic modifier pair rather than a simple edge."
                    )

            fragile_nodes.append(FragileNode(
                node_id=node_id,
                fragility_score=round(score, 4),
                fragility_class=fragility_class,
                recommendation=recommendation,
                positive_weight_sum=round(pos_sum, 4),
                negative_weight_sum=round(neg_sum, 4),
                neighbor_count=neighbor_count,
                epistasis_modifier=epistasis_modifier,
            ))

        fragile_nodes.sort(key=lambda fn: fn.fragility_score, reverse=True)

        # ── Global stability ──────────────────────────────────
        global_stability = float(1.0 - float(np.mean(normalized_displacement)))
        lambda2_stress_delta = lambda2_baseline - lambda2_stressed

        contested_count       = sum(1 for fn in fragile_nodes if fn.fragility_class == "contested")
        under_evidenced_count = sum(1 for fn in fragile_nodes if fn.fragility_class == "under_evidenced")
        epistasis_count       = sum(1 for fn in fragile_nodes if fn.fragility_class == "epistasis_candidate")

        report = DiagnosticReport(
            zone=zone_val, timestamp=now,
            heat_applied=self.heat, rounds=self.rounds,
            nodes_scanned=n,
            fragile_nodes=fragile_nodes,
            lambda2_baseline=round(lambda2_baseline, 6),
            lambda2_stressed=round(lambda2_stressed, 6),
            lambda2_stress_delta=round(lambda2_stress_delta, 6),
            global_stability=round(global_stability, 4),
            contested_count=contested_count,
            under_evidenced_count=under_evidenced_count,
            epistasis_candidate_count=epistasis_count,
        )

        logger.info("DiagnosticAnnealer: %s", report.summary())
        return report

    def run_targeted(self, zone, node_ids: List[str]) -> List[FragileNode]:
        """
        Run a targeted diagnostic on a specific subset of nodes.

        Useful when you already suspect certain nodes are fragile and want
        to characterize them without scanning the entire zone. Returns a
        FragileNode entry for each requested node_id regardless of whether
        it crosses the fragility_threshold.

        Parameters
        ----------
        zone : GraphZone
            The zone containing the nodes.
        node_ids : list of str
            Node IDs to analyze. IDs not found in the zone are silently skipped.

        Returns
        -------
        list of FragileNode
            Sorted by fragility_score descending.
        """
        adj, _ = self.sal._build_weighted_adjacency(zone)
        nodes = sorted(adj.keys())
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        baseline_degrees = self._weighted_degrees(adj, nodes)

        accumulated: Dict[str, float] = defaultdict(float)

        for _ in range(self.rounds):
            perturbed_adj = self._perturb(adj)
            perturbed_degrees = self._weighted_degrees(perturbed_adj, nodes)
            for nid in node_ids:
                if nid not in node_idx:
                    continue
                i = node_idx[nid]
                accumulated[nid] += abs(
                    float(perturbed_degrees[i]) - float(baseline_degrees[i])
                )

        results: List[FragileNode] = []

        for nid in node_ids:
            if nid not in node_idx:
                continue
            i = node_idx[nid]
            mean_disp = accumulated[nid] / self.rounds
            baseline_deg = abs(float(baseline_degrees[i]))
            # Special case: isolated nodes (degree 0) are maximally fragile
            if baseline_deg == 0:
                score = 1.0
            else:
                score = float(np.clip(mean_disp / max(baseline_deg, 1e-8), 0.0, 1.0))

            pos_sum, neg_sum = self._edge_mass(adj, nid)
            neighbor_count = len(adj.get(nid, {}))
            fragility_class, recommendation = self._triage(
                score, pos_sum, neg_sum, neighbor_count
            )

            epistasis_modifier: Optional[str] = None
            if neighbor_count >= 2:
                epistasis_modifier = self._check_epistasis(adj, nodes, nid, score)
                if epistasis_modifier is not None:
                    fragility_class = "epistasis_candidate"
                    recommendation = (
                        f"Fragility strongly coupled to neighbor "
                        f"'{epistasis_modifier}'. Model as epistatic modifier pair."
                    )

            results.append(FragileNode(
                node_id=nid,
                fragility_score=round(score, 4),
                fragility_class=fragility_class,
                recommendation=recommendation,
                positive_weight_sum=round(pos_sum, 4),
                negative_weight_sum=round(neg_sum, 4),
                neighbor_count=neighbor_count,
                epistasis_modifier=epistasis_modifier,
            ))

        results.sort(key=lambda fn: fn.fragility_score, reverse=True)
        return results

    # ── Internal: perturbation ────────────────────────────────

    def _perturb(
        self,
        adj: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Inject multiplicative noise into edge weights.

        Each undirected edge (u, v) gets the same η ~ Uniform(-heat, +heat)
        applied symmetrically to both adj[u][v] and adj[v][u]. This keeps
        the adjacency consistent and prevents asymmetric phantom mass.

        Signs are always preserved: a negative edge stays negative, a
        positive edge stays positive. We're probing structural fragility,
        not simulating adversarial edge-type flips.
        """
        perturbed: Dict[str, Dict[str, float]] = {src: {} for src in adj}
        visited_pairs: Set[Tuple[str, str]] = set()

        for src, neighbors in adj.items():
            for dst, weight in neighbors.items():
                pair: Tuple[str, str] = (
                    (src, dst) if src < dst else (dst, src)
                )
                if pair in visited_pairs:
                    # Copy the already-computed symmetric value
                    mirrored = perturbed.get(dst, {}).get(src)
                    perturbed[src][dst] = mirrored if mirrored is not None else weight
                    continue

                eta = float(self._rng.uniform(-self.heat, self.heat))
                new_w = weight * (1.0 + eta)
                perturbed[src][dst] = new_w
                perturbed.setdefault(dst, {})[src] = new_w
                visited_pairs.add(pair)

        return perturbed

    def _weighted_degrees(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
    ) -> np.ndarray:
        """
        Compute signed weighted degree for each node in `nodes`.

        Degree = sum of all edge weights (positive contributions lift it,
        negative contributions lower it). A node with many strong supporting
        edges has high positive degree and resists perturbation. A node with
        few or weak edges has degree near zero and moves easily.
        """
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        degrees = np.zeros(len(nodes))
        for nid, neighbors in adj.items():
            if nid in node_idx:
                degrees[node_idx[nid]] = sum(neighbors.values())
        return degrees

    def _compute_lambda2(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
    ) -> float:
        """
        Compute Fiedler value (λ₂) of the signed normalized Laplacian.

        Delegates to SAL's internal methods to keep the computation
        exactly consistent with what coherence_state() reports. Returns 0.0
        on any error rather than raising, since perturbation rounds should
        be robust to occasional numerical edge cases.
        """
        if len(nodes) < 3:
            return 0.0
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        try:
            L, _ = self.sal._build_normalized_laplacian(adj, nodes, node_idx)
            return float(self.sal._compute_fiedler_value(L, len(nodes)))
        except Exception as exc:
            logger.debug("DiagnosticAnnealer: lambda2 compute failed: %s", exc)
            return 0.0

    # ── Internal: triage ──────────────────────────────────────

    def _edge_mass(
        self,
        adj: Dict[str, Dict[str, float]],
        node_id: str,
    ) -> Tuple[float, float]:
        """
        Return (positive_weight_sum, negative_weight_sum) for a node.

        Both returned values are non-negative (absolute values).
        """
        neighbors = adj.get(node_id, {})
        pos = sum(w for w in neighbors.values() if w > 0.0)
        neg = sum(abs(w) for w in neighbors.values() if w < 0.0)
        return pos, neg

    def _triage(
        self,
        score: float,
        pos_sum: float,
        neg_sum: float,
        neighbor_count: int,
    ) -> Tuple[str, str]:
        """
        Classify a fragile node and produce a human-readable prescription.

        Decision tree:
            1. If negative weight fraction > contested_threshold → contested.
               The graph already senses this conflict. Escalate; don't shore up.
            2. If positive weighted degree < low_mass_threshold → under_evidenced.
               The belief is lightly anchored. Seek corroborating evidence.
            3. Else → under_evidenced (well-massed but structurally isolated).
               Check cluster membership via SALCoherenceLayer.
        """
        total = pos_sum + neg_sum
        neg_fraction = neg_sum / total if total > 1e-8 else 0.0

        if neg_fraction > self.contested_threshold:
            return (
                "contested",
                (
                    f"Node has significant contradicting edge mass "
                    f"({neg_fraction:.0%} of total). The graph already senses "
                    "this conflict at the edge level. Escalate to contradiction "
                    "resolution — do not reinforce the node directly."
                ),
            )

        if pos_sum < self.low_mass_threshold:
            return (
                "under_evidenced",
                (
                    f"Node is lightly anchored (positive weighted degree "
                    f"{pos_sum:.3f} < threshold {self.low_mass_threshold:.3f}). "
                    "The belief is probably correct but has little corroboration. "
                    "Queue for evidence-seeking; do not increase edge weights directly."
                ),
            )

        # Fragile despite adequate mass — likely structurally isolated
        return (
            "under_evidenced",
            (
                f"Node has adequate positive mass ({pos_sum:.3f}) but still "
                f"shows fragility score {score:.3f}. It may be connected to "
                "a weak subgraph rather than the giant component. Check cluster "
                "membership via SALCoherenceLayer.node_coherence_memberships()."
            ),
        )

    def _check_epistasis(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        node_id: str,
        base_fragility: float,
    ) -> Optional[str]:
        """
        Check whether removing one neighbor dramatically reduces fragility.

        If removing neighbor X drops the node's fragility by more than
        epistasis_delta, X is acting as a modifier that makes the node
        fragile when present. This is epistasis: the node's stability is
        conditional on whether X is in the graph.

        Returns the node_id of the strongest epistasis modifier candidate,
        or None if no single neighbor clears the threshold.

        Uses a reduced round count (rounds // epistasis_rounds_divisor) for
        speed since this runs per-neighbor and compounds quickly.
        """
        neighbors = list(adj.get(node_id, {}).keys())
        if not neighbors:
            return None

        epistasis_rounds = max(8, self.rounds // _EPISTASIS_ROUNDS_DIVISOR)
        best_modifier: Optional[str] = None
        best_reduction: float = 0.0

        for candidate in neighbors:
            # Build adjacency with the candidate neighbor removed
            stripped_adj: Dict[str, Dict[str, float]] = {}
            for src, nbrs in adj.items():
                if src == candidate:
                    stripped_adj[src] = {}
                    continue
                stripped_adj[src] = {
                    dst: w for dst, w in nbrs.items() if dst != candidate
                }

            # Measure local fragility for node_id in the stripped graph
            stripped_baseline = sum(stripped_adj.get(node_id, {}).values())
            accumulated_disp = 0.0

            for _ in range(epistasis_rounds):
                perturbed = self._perturb(stripped_adj)
                perturbed_deg = sum(perturbed.get(node_id, {}).values())
                accumulated_disp += abs(perturbed_deg - stripped_baseline)

            mean_disp = accumulated_disp / epistasis_rounds
            stripped_deg = abs(stripped_baseline)
            stripped_score = float(
                np.clip(mean_disp / max(stripped_deg, 1e-8), 0.0, 1.0)
            )

            reduction = base_fragility - stripped_score
            if reduction > best_reduction:
                best_reduction = reduction
                best_modifier = candidate

        if best_reduction >= self.epistasis_delta:
            return best_modifier
        return None

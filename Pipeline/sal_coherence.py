"""
VeritasMemoria - SRL Coherence Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements the Spectral Reasoning Layer (SRL) stopping criterion
from the phase transition framework. Gives VM a principled answer
to the question it currently cannot answer:

    "Has the agent reasoned enough to commit, or is it still in
     the subcritical regime where committing would be premature?"

Architecture
------------
The SRL framework models the belief graph as a random graph
undergoing an Erdős–Rényi phase transition. Below the critical
edge density p_c = 1/n, the graph is fragmented — beliefs don't
cohere into a reliable basis for decision. Above p_c, a giant
connected component emerges — the reasoning has converged.

Two computable signals mark this transition:

    λ₂  (Fiedler value) — the second-smallest eigenvalue of the
         normalized graph Laplacian. Near zero = fragmented graph.
         Opening gap = connectivity is emerging. Stable above
         λ_min = coherence has converged.

    Sheaf consistency — for each node and its neighborhood, are
         the belief states locally compatible? Measures how well
         the graph's local structure agrees globally.

Stopping criterion (from Section 4 of the framework):
    COMMIT when:  dλ₂/dt < ε_spectral  AND  λ₂ > λ_min
                  AND mean sheaf consistency > τ_sheaf
    CONTINUE when: any condition above is not met
    BLOCKED when: unresolved contradictions exist in zone

Integration with VM
-------------------
- Reads directly from GraphEngine._adj (in-memory adjacency cache)
- Respects GateLevel — BLOCK_UNTIL_RESOLVED edges are excluded
- Integrates with IlluminationResult — coherence state enriches
  the nagging signal (nodes in subcritical clusters get higher
  nagging scores because they haven't joined the giant component)
- Plugs into VeritasMemoriaSystem.illuminate() as an additional
  signal layer
- The Archivist can call coherence_state() before committing a
  session to determine whether reasoning has converged
- Optionally accepts a semantic_index (ProductionSemanticIndex) to
  enable restriction-map sheaf consistency instead of edge census
- Cross-zone bridge dependency checking: coherence_state() surfaces
  whether the committing zone depends on unresolved work in other zones

Relationship to VM's typed edges
---------------------------------
VM's graph is not a pure ER random graph — it has typed edges
with gate levels, and BLOCK_UNTIL_RESOLVED edges act as inhibitory
constraints that prevent traversal. This implementation accounts
for that by:
  1. Excluding blocked edges from the Laplacian (they reduce
     effective n and alter connectivity statistics)
  2. Weighting edges by kind — SUPPORTS/EVIDENCE edges carry
     positive weight, CONTRADICTS edges are excluded entirely
     (weight=0.0 in EDGE_KIND_WEIGHTS, filtered at adjacency build time)
  3. Using the weighted Laplacian so that high-confidence edges
     contribute more to the Fiedler value

Spectral approach — signed Laplacian
-------------------------------------
CONTRADICTS edges are included in the adjacency with negative weights
(-1.0 for contradicts, -0.5 for evidence_contradicts_preference).
The Laplacian used is the signed normalized Laplacian:

    L_s = D_|A|^{-1/2} (D_|A| - A) D_|A|^{-1/2}

where D_|A| is built from absolute-value row sums.

This means λ₂ directly reflects net coherence:
  - A zone with 10 SUPPORTS and 0 CONTRADICTS → high λ₂
  - A zone with 10 SUPPORTS and 5 CONTRADICTS → suppressed λ₂
  - A zone with equal supports and contradicts → λ₂ near or below 0

λ₂ can be negative for the signed Laplacian (eigenvalues in [-1, 2]).
Negative λ₂ means contradiction structure dominates; the zone is
incoherent and commit should be blocked by the lambda2_sufficient check.

CoherenceState.contradiction_edges reports how many contradiction edges
were present and actively suppressed lambda2 via negative weights.

Usage
-----
    from sal_coherence import SALCoherenceLayer, CoherenceState

    # Attach to an existing GraphEngine
    srl = SALCoherenceLayer(graph_engine)

    # Check if reasoning has converged before committing
    state = srl.coherence_state(GraphZone.WORK_KNOWLEDGE)
    if state.should_commit:
        archivist.commit_session(...)
    else:
        # Continue reasoning; state.explanation tells you why

    # Get per-node coherence membership (enriches IlluminationResult)
    memberships = srl.node_coherence_memberships(GraphZone.WORK_KNOWLEDGE)

    # Full diagnostic for the Calibrator
    report = srl.diagnostic_report(GraphZone.WORK_KNOWLEDGE)

Dependencies
------------
numpy  — Laplacian construction and eigenvalue computation
scipy  — sparse eigensolver (much faster than numpy for large graphs)

Both are already in your requirements. No new dependencies needed.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    # Imported here for type hints only — avoids a circular import at runtime
    # since field_theory_engine imports hsh_geometry, which sal_coherence also
    # imports lazily.  At runtime the engine is passed in as an opaque object.
    from .field_theory_engine import (
        FieldTheoryEngine,
        RegimeState,
    )

import numpy as np

from kalman_belief import ZoneKalmanRegistry, KalmanEstimate
from hsh_geometry import HSHGeometry

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants (tunable via SALCoherenceLayer constructor)
# ─────────────────────────────────────────────────────────────

# λ₂ must exceed this to be considered "connected enough"
DEFAULT_LAMBDA_MIN: float = 0.05

# |Δλ₂| must be below this to be considered "stable"
DEFAULT_EPSILON_SPECTRAL: float = 0.01

# Mean sheaf consistency must exceed this
DEFAULT_TAU_SHEAF: float = 0.60

# Edge kind weights — signed. Positive weights contribute to coherent
# connectivity and raise lambda2. Negative weights represent repulsive
# structure (contradictions) and push lambda2 down, correctly reducing
# measured coherence when unresolved disagreements exist.
#
# These feed into the signed normalized Laplacian:
#   L_s = D_|A|^{-1/2} (D_|A| - A) D_|A|^{-1/2}
# where D_|A| is built from absolute-value row sums. This keeps the
# matrix well-defined with eigenvalues in [-1, 2].
EDGE_KIND_WEIGHTS: Dict[str, float] = {
    "supports":                        1.0,
    "evidence_supports_decision":      1.0,
    "fact_updates_belief":             0.9,
    "implements":                      0.85,
    "depends_on":                      0.8,
    "refines":                         0.8,
    "decision_updates_preference":     0.75,
    "decision_refines_policy":         0.75,
    "about":                           0.6,
    "temporal_next":                   0.5,
    "semantic_similarity":             0.5,   # temp edges
    "decision_requires_evidence":      0.4,
    "duplicate_of":                    0.3,
    "evidence_contradicts_preference": -0.5,  # weak repulsion
    "contradicts":                     -1.0,  # full repulsion -- reduces lambda2
}


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SpectralSnapshot:
    """
    A single measurement of the graph's spectral state.
    Stored in the history buffer to compute dλ₂/dt.
    """
    timestamp: str
    lambda2: float           # Fiedler value
    node_count: int
    edge_count: int          # unblocked, weighted edges only
    zone: str


@dataclass
class SheavConsistencyScore:
    """
    Local consistency score for a single node and its neighborhood.

    Implements an approximation of sheaf consistency (Robinson 2017).
    A true sheaf assigns restriction maps between node belief states —
    the consistency radius measures how much perturbation is needed to
    globalize the local sections. We approximate this in two modes:

    Restriction-map mode (when semantic_index is available):
        For each edge (u, v), compute cosine similarity between the
        embeddings of u and v. Compare to the expected similarity
        profile for that edge kind: SUPPORTS edges expect high
        similarity (the beliefs should agree), CONTRADICTS edges
        expect low or negative similarity (they should disagree).
        Score = how well actual pairwise similarity matches expectation.
        This catches transitive incompatibilities that have no explicit
        CONTRADICTS edge — if A SUPPORTS B and B SUPPORTS C but
        embed(A) and embed(C) are near-orthogonal, the chain is
        inconsistent even without a direct A-C contradiction edge.

    Census mode (fallback, no semantic_index):
        Edge-type census with penalty formula. Captures only explicitly
        typed contradictions. Faster but blind to implicit incoherence.

    score is 0.0 (inconsistent) to 1.0 (fully consistent) in both modes.
    restriction_map_mode=True indicates which mode was used.
    """
    node_id: str
    score: float                    # 0.0 (inconsistent) to 1.0 (fully consistent)
    support_count: int
    contradiction_count: int
    dependency_count: int
    neighbor_count: int
    restriction_map_mode: bool = False   # True if computed from embeddings
    mean_neighbor_similarity: float = 0.0  # mean cosine sim to neighbors (restriction mode)


@dataclass
class CoherenceState:
    """
    The output of SALCoherenceLayer.coherence_state().

    This is the stopping criterion result — the answer to
    "should the agent commit, or keep reasoning?"
    """
    zone: str
    timestamp: str

    # Spectral signals
    lambda2: float                   # Current Fiedler value
    lambda2_delta: float             # |Δλ₂| since last measurement
    lambda2_stable: bool             # delta < epsilon_spectral
    lambda2_sufficient: bool         # lambda2 > lambda_min

    # Sheaf consistency
    mean_sheaf_consistency: float
    sheaf_sufficient: bool           # mean > tau_sheaf
    nodes_inconsistent: List[str]    # node_ids with low consistency

    # Contradiction state (hard gate)
    unresolved_contradictions: int
    contradictions_blocking: bool    # any BLOCK_UNTIL_RESOLVED exist

    # Final verdict
    should_commit: bool
    regime: str                      # "subcritical" | "critical" | "supercritical"
    explanation: str

    # Graph statistics
    node_count: int
    edge_count: int                  # effective (unblocked, positive-weight) edges
    giant_component_fraction: float  # fraction of nodes in largest component

    # Number of contradiction edges present in the zone's adjacency.
    # These are included in the signed Laplacian with negative weights,
    # so they actively suppress lambda2. Non-zero means lambda2 already
    # reflects the repulsive structure -- no separate backstop needed
    # for the spectral signal.
    contradiction_edges: int = 0

    # Cross-zone bridge dependencies found during commit check.
    # Non-empty means this zone has outbound bridge edges to zones that
    # are not themselves committable. Commit may be premature if this
    # zone's beliefs depend on unresolved work in another zone.
    bridge_dependencies: List = field(default_factory=list)  # List[BridgeDependency]

    # Per-node coherence cluster membership (node_id -> cluster_id)
    # Nodes in the giant component share cluster 0
    cluster_membership: Dict[str, int] = field(default_factory=dict)

    # Kalman-smoothed lambda2 signal (runs alongside raw delta, not yet
    # used in stopping criterion -- observe both signals before switching)
    lambda2_kalman: Optional[float] = None          # smoothed estimate
    lambda2_kalman_variance: Optional[float] = None # filter confidence
    lambda2_kalman_residual: Optional[float] = None # innovation (raw - smoothed)

    # Pre-contradiction fragile bridge nodes from the D(u) displacement signal.
    # Populated when HSH coherence field is active; empty when falling back to the
    # flat Laplacian.  Each entry is (node_id, D_value) sorted descending.
    #
    # When non-empty AND phi_H is only marginally above lambda_min, these nodes
    # indicate zone-seam tension that the Fiedler value has not yet degraded
    # enough to catch.  should_commit is set False in this case (fragile_bridge_blocking).
    fragile_bridge_nodes: List = field(default_factory=list)  # List[Tuple[str, float]]

    # ── Field theory integration (populated when FieldTheoryEngine is wired in) ─

    # Three-regime state machine output (Coherent/Stressed/Surgical).
    # None when no FieldTheoryEngine is attached or in Phase 0 dormancy.
    regime_state: Optional["RegimeState"] = None

    # True when dλ₂/dt < −δ_c — early warning of Regime 2 entry.
    # Fires before λ₂ drops below threshold, giving human intervention time.
    # Always False when no FieldTheoryEngine is attached or in Phase 0.
    derivative_warning: bool = False


@dataclass
class BridgeDependency:
    """
    Represents a cross-zone dependency found during bridge checking.

    When zone A has a bridge edge to zone B, committing A may be unsafe
    if B has unresolved work that A depends on. This dataclass captures
    one such dependency for surfacing in CoherenceState.
    """
    from_zone: str
    from_node_id: str
    to_zone: str
    to_node_id: str
    edge_weight: float
    to_zone_committable: bool    # whether the target zone passes its own criterion
    to_zone_blocked: bool        # whether the target zone has hard blocks
    to_zone_lambda2: float       # spectral state of the target zone at check time


# ─────────────────────────────────────────────────────────────
# SRL Coherence Layer
# ─────────────────────────────────────────────────────────────

class SALCoherenceLayer:
    """
    Spectral Reasoning Layer coherence monitor for VeritasMemoria.

    Attaches to a GraphEngine instance and computes spectral and
    sheaf-theoretic coherence signals over a zone's belief graph.

    The core insight: VM's graph is already a belief dependency graph.
    The SRL framework gives that graph a computable stopping criterion.
    This class is the bridge between the two.
    """

    def __init__(
        self,
        graph_engine,                            # GraphEngine instance
        lambda_min: float = DEFAULT_LAMBDA_MIN,
        epsilon_spectral: float = DEFAULT_EPSILON_SPECTRAL,
        tau_sheaf: float = DEFAULT_TAU_SHEAF,
        history_size: int = 10,                  # snapshots to keep per zone
        semantic_index=None,                     # optional ProductionSemanticIndex
        memory_library=None,                     # optional MemoryLibrary for content lookup
        hsh_mode: bool = True,                   # use HSH Curvature-Adaptive Laplacian
        vg=None,                                 # optional VeritasGeometria for ORC kappa
        field_theory_engine: Optional["FieldTheoryEngine"] = None,  # scalar-tensor FTE
    ):
        self.graph = graph_engine
        self.lambda_min = lambda_min
        self.epsilon_spectral = epsilon_spectral
        self.tau_sheaf = tau_sheaf
        self.history_size = history_size
        self.semantic_index = semantic_index     # enables restriction-map sheaf mode
        self.memory_library = memory_library     # enables content-based diagnostics
        self.hsh_mode = hsh_mode                 # HSH: use Curvature-Adaptive Laplacian
        self._vg = vg                            # VeritasGeometria — ORC kappa source

        # Field theory engine — optional.  When present:
        #   • every λ₂ computation is logged to the engine's time-series buffer
        #   • the three-regime state machine (Coherent/Stressed/Surgical) is updated
        #   • the dλ₂/dt derivative warning is surfaced in CoherenceState
        self._fte: Optional["FieldTheoryEngine"] = field_theory_engine

        # HSH geometry engine — lazy init to avoid import cost when hsh_mode=False
        self._hsh: Optional["HSHGeometry"] = None

        # Kalman filter registry — one tracker per zone, runs alongside raw delta
        self._kalman = ZoneKalmanRegistry()

        # Spectral history per zone for dλ₂/dt computation
        # zone_value -> deque[SpectralSnapshot] (maxlen caps memory automatically)
        from collections import deque as _deque
        from typing import Deque as _Deque
        self._history: Dict[str, _Deque[SpectralSnapshot]] = defaultdict(
            lambda: _deque(maxlen=self.history_size)
        )

    # ── Public API ───────────────────────────────────────────

    def coherence_state(self, zone) -> CoherenceState:
        """
        Compute the full coherence state for a zone.

        This is the main entry point. Call this before committing
        a reasoning session to determine if the agent has reasoned
        enough to commit reliably.

        Parameters
        ----------
        zone : GraphZone
            The zone to evaluate.

        Returns
        -------
        CoherenceState
            Full stopping criterion result with explanation.
        """
        zone_val = zone.value if hasattr(zone, 'value') else str(zone)
        now = datetime.now(timezone.utc).isoformat()

        # Build the weighted adjacency from GraphEngine's in-memory cache
        adj, contradiction_edges = self._build_weighted_adjacency(zone)
        nodes = sorted(adj.keys())
        n = len(nodes)

        if n < 3:
            # Too few nodes to compute meaningful spectral properties
            return CoherenceState(
                zone=zone_val, timestamp=now,
                lambda2=0.0, lambda2_delta=0.0,
                lambda2_stable=False, lambda2_sufficient=False,
                mean_sheaf_consistency=0.0, sheaf_sufficient=False,
                nodes_inconsistent=[],
                unresolved_contradictions=len(self.graph._blocked_nodes),
                contradictions_blocking=len(self.graph._blocked_nodes) > 0,
                should_commit=False,
                regime="subcritical",
                explanation=(
                    f"Zone '{zone_val}' has only {n} nodes — "
                    "insufficient graph density for coherence measurement. "
                    "Add more memories before committing."
                ),
                node_count=n, edge_count=0,
                giant_component_fraction=0.0,
            )

        # ── Step 1: Compute Fiedler value (λ₂) ──────────────
        # HSH mode: use the Curvature-Adaptive Laplacian L_H instead of the
        # flat signed normalized Laplacian L_s. The coherence field Phi_H
        # combines lambda_2(L_H) with the Gaussian Curvature integral,
        # giving a geometric penalty for content that lacks hierarchical depth.
        node_idx = {nid: i for i, nid in enumerate(nodes)}

        hsh_field = None
        if self.hsh_mode:
            if self._hsh is None:
                from hsh_geometry import HSHGeometry
                self._hsh = HSHGeometry(phi_min=self.lambda_min)
            # Build simple weight adjacency for HSH (flatten tuple format)
            hsh_adj: Dict[str, Dict[str, float]] = {}
            for nid, neighbors in adj.items():
                hsh_adj[nid] = {dst: w for dst, w in neighbors.items()}
            # Per-node zone map: every node in this call belongs to `zone`.
            # Previously never threaded through, causing all nodes to default
            # to WORK_KNOWLEDGE regardless of actual zone — now corrected.
            node_zones = {nid: zone for nid in nodes}
            # ORC kappa from VeritasGeometria gives topology-dependent curvature.
            # Falls back to Gaussian curvature (zone-constant) when VG is absent.
            orc_kappa = None
            if self._vg is not None:
                try:
                    cmap = self._vg.curvature_map(zone)
                    orc_kappa = cmap.node_mean_kappa
                except Exception as e:
                    logger.debug(
                        "VG curvature_map failed for zone %s, using Gaussian fallback: %s",
                        zone_val, e,
                    )
            try:
                hsh_field = self._hsh.coherence_field(
                    hsh_adj, nodes, zone=zone,
                    node_zones=node_zones,
                    orc_kappa=orc_kappa,
                )
                lambda2 = hsh_field.phi_H   # Phi_H replaces raw lambda2 as signal
            except Exception as e:
                logger.warning("HSH coherence field failed, falling back to flat: %s", e)
                hsh_field = None

        # Compute GOV→RAT seam fragile bridges using a dedicated cross-zone adjacency.
        # Cannot rely on hsh_field.fragile_bridge_nodes here: coherence_field() receives
        # a single-zone adjacency where node_zones maps every node to the same zone, so
        # displacement_signal() finds no cross-zone edges and always returns zeros.
        # Instead, build the seam adjacency from the raw graph and call fragile_bridges()
        # directly with the correct per-node zone assignments.
        fragile_bridge_nodes: List = []
        if self.hsh_mode and self._hsh is not None:
            try:
                from graph_types import GraphZone as _GZ
                seam_adj, seam_nodes, seam_node_zones = self._build_seam_adjacency(
                    _GZ.GOVERNANCE, _GZ.RATIONALE
                )
                if seam_nodes:
                    fragile_bridge_nodes = self._hsh.fragile_bridges(
                        seam_adj, seam_nodes,
                        node_zones=seam_node_zones,
                        seam_zones=(_GZ.GOVERNANCE, _GZ.RATIONALE),
                    )
            except Exception as _fb_exc:
                logger.debug(
                    "seam fragile bridge computation failed (non-fatal): %s", _fb_exc
                )

        if hsh_field is None:
            # Flat signed Laplacian fallback
            L, edge_count = self._build_normalized_laplacian(adj, nodes, node_idx)
            lambda2 = self._compute_fiedler_value(L, n)
        else:
            # Edge count from flat adjacency for reporting
            _, edge_count = self._build_normalized_laplacian(adj, nodes, node_idx)

        # Kalman-smoothed estimate runs alongside raw signal
        k_estimate = self._kalman.update(zone_val, lambda2)

        # ── Field theory integration ──────────────────────────
        # When a FieldTheoryEngine is attached, log this λ₂ measurement and
        # compute the regime/derivative-warning signals.  Both run even in
        # Phase 0 so that the engine builds a stable baseline — only the
        # state-machine output is suppressed by the engine in Phase 0.
        ft_regime: Optional["RegimeState"] = None
        ft_derivative_warning: bool = False
        if self._fte is not None:
            self._fte.log_lambda2(zone_val, lambda2, timestamp=now)
            ft_regime = self._fte.compute_regime(
                zone_val=zone_val,
                lambda2=lambda2,
                lambda_min=self.lambda_min,
                surgery_triggered=self._fte.any_surgery_triggered,
            )
            ft_derivative_warning = self._fte.derivative_warning_fires(zone_val)

        # ── Step 2: Measure spectral stability (dλ₂/dt) ─────
        history = self._history[zone_val]
        lambda2_delta = 0.0
        if history:
            lambda2_delta = abs(lambda2 - history[-1].lambda2)
        # Require at least 2 snapshots before stability can be declared.
        # On the first call, history is empty so has_history=False and
        # lambda2_delta=0.0 — without this guard, the first call would
        # trivially satisfy stability (delta=0) before any trend data exists,
        # risking a false COMMIT on the very first evaluation.
        # After this snapshot is appended, len(history) >= 2 on the next call.
        has_history = len(history) >= 1  # True on 2nd+ call; False on 1st

        # Record snapshot — deque handles maxlen trimming in O(1)
        snap = SpectralSnapshot(
            timestamp=now, lambda2=lambda2,
            node_count=n, edge_count=edge_count, zone=zone_val
        )
        history.append(snap)

        # ── Step 3: Sheaf consistency ────────────────────────
        # Use restriction-map mode if semantic_index is available,
        # fall back to edge census otherwise.
        if self.semantic_index is not None:
            sheaf_scores = self._compute_sheaf_consistency_restriction(zone, adj)
        else:
            sheaf_scores = self._compute_sheaf_consistency(zone, adj)
        mean_sheaf = (
            sum(s.score for s in sheaf_scores) / len(sheaf_scores)
            if sheaf_scores else 0.0
        )
        nodes_inconsistent = [
            s.node_id for s in sheaf_scores
            if s.score < self.tau_sheaf and s.contradiction_count > 0
        ]

        # ── Step 4: Contradiction gate check ────────────────
        unresolved = len(self.graph._blocked_nodes)
        blocking = unresolved > 0

        # ── Step 5: Giant component fraction ─────────────────
        giant_fraction, clusters = self._find_connected_components(adj, nodes)

        # ── Step 6: Determine regime ─────────────────────────
        # Erdős–Rényi critical point: λ₂ ≈ 0 below p_c, opens above p_c
        # We use the ratio to node count as a proxy for the critical density
        effective_density = (2 * edge_count) / max(n * (n - 1), 1)
        critical_density = 1.0 / max(n, 1)

        if effective_density < critical_density * 0.5:
            regime = "subcritical"
        elif lambda2 > self.lambda_min and giant_fraction > 0.5:
            regime = "supercritical"
        else:
            regime = "critical"

        # ── Step 7: Cross-zone bridge dependency check ───────
        bridge_deps = self._check_bridge_dependencies(zone)

        # ── Step 8: Stopping criterion ───────────────────────
        lambda2_stable = has_history and lambda2_delta < self.epsilon_spectral
        lambda2_sufficient = lambda2 > self.lambda_min
        sheaf_sufficient = mean_sheaf > self.tau_sheaf
        # Bridge warning: unresolved dependencies in other zones don't hard-block
        # commit (the other zone may not be ready yet for unrelated reasons) but
        # they are surfaced prominently so the Archivist can decide.
        bridge_warning = any(
            not dep.to_zone_committable for dep in bridge_deps
        )

        # Fragile bridge gate: pre-contradiction seam tension that the Fiedler
        # value has not yet degraded enough to catch.  Only fires when fragile
        # bridges exist AND phi_H is only marginally above lambda_min (i.e. the
        # coherence field is weak).  phi_H has replaced lambda2 as the signal
        # when HSH is active, so the comparison is against the same variable.
        fragile_bridge_blocking = (
            bool(fragile_bridge_nodes)
            and lambda2 < 2.0 * self.lambda_min
        )

        should_commit = (
            lambda2_stable
            and lambda2_sufficient
            and sheaf_sufficient
            and not blocking
            and not fragile_bridge_blocking
        )

        explanation = self._build_explanation(
            zone_val, should_commit, regime,
            lambda2, lambda2_delta, lambda2_stable, lambda2_sufficient,
            mean_sheaf, sheaf_sufficient,
            unresolved, blocking,
            giant_fraction, n, edge_count,
            contradiction_edges,
            bridge_deps,
            fragile_bridge_nodes=fragile_bridge_nodes,
            ft_regime=ft_regime,
            ft_derivative_warning=ft_derivative_warning,
        )

        return CoherenceState(
            zone=zone_val, timestamp=now,
            lambda2=round(lambda2, 6),
            lambda2_delta=round(lambda2_delta, 6),
            lambda2_stable=lambda2_stable,
            lambda2_sufficient=lambda2_sufficient,
            mean_sheaf_consistency=round(mean_sheaf, 4),
            sheaf_sufficient=sheaf_sufficient,
            nodes_inconsistent=nodes_inconsistent,
            unresolved_contradictions=unresolved,
            contradictions_blocking=blocking,
            should_commit=should_commit,
            regime=regime,
            explanation=explanation,
            node_count=n,
            edge_count=edge_count,
            giant_component_fraction=round(giant_fraction, 4),
            contradiction_edges=contradiction_edges,
            bridge_dependencies=bridge_deps,
            cluster_membership=clusters,
            lambda2_kalman=k_estimate.value,
            lambda2_kalman_variance=k_estimate.variance,
            lambda2_kalman_residual=k_estimate.residual,
            fragile_bridge_nodes=fragile_bridge_nodes,
            regime_state=ft_regime,
            derivative_warning=ft_derivative_warning,
        )

    def node_coherence_memberships(self, zone) -> Dict[str, int]:
        """
        Return cluster membership for each node in a zone.

        Nodes in the giant component (cluster 0) have joined the
        coherent subgraph. Nodes in other clusters are isolated
        belief fragments — good candidates for the nagging signal.

        Returns dict of node_id -> cluster_id.
        Cluster 0 = giant component (coherent).
        Cluster 1+ = isolated fragments (incoherent).
        """
        adj, _ = self._build_weighted_adjacency(zone)
        nodes = sorted(adj.keys())
        _, clusters = self._find_connected_components(adj, nodes)
        return clusters

    def enrich_illumination(
        self,
        illumination_results: list,
        zone,
        subcritical_boost: float = 1.5,
    ) -> list:
        """
        Enrich IlluminationResult objects with coherence membership.

        Nodes in subcritical clusters (not in the giant component)
        get a boosted nagging_score — they represent isolated belief
        fragments that haven't been connected to the coherent reasoning
        graph yet. This is the formal grounding for the "nagging feeling":
        the system knows something exists but it hasn't been integrated
        into the coherent whole.

        Parameters
        ----------
        illumination_results : list[IlluminationResult]
            Output from GraphEngine.illuminate()
        zone : GraphZone
            Same zone the illumination was run on
        subcritical_boost : float
            Multiplier applied to nagging_score for subcritical nodes

        Returns
        -------
        Enriched list, sorted by adjusted nagging_score descending.
        """
        memberships = self.node_coherence_memberships(zone)

        for result in illumination_results:
            cluster = memberships.get(result.node_id, -1)
            if cluster > 0:
                # Not in giant component — boost the nagging signal
                result.nagging_score *= subcritical_boost
                # Tag it so callers know why the score changed
                result.__dict__['coherence_cluster'] = cluster
                result.__dict__['coherence_note'] = (
                    "Isolated belief fragment — not yet integrated "
                    "into the coherent reasoning graph."
                )
            else:
                result.__dict__['coherence_cluster'] = 0
                result.__dict__['coherence_note'] = (
                    "Member of coherent reasoning graph."
                )

        illumination_results.sort(key=lambda r: r.nagging_score, reverse=True)
        return illumination_results

    def diagnostic_report(self, zone) -> Dict:
        """
        Full diagnostic for the Calibrator.

        Returns a JSON-serializable dict suitable for logging
        or display in the VM dashboard.
        """
        state = self.coherence_state(zone)
        zone_val = zone.value if hasattr(zone, 'value') else str(zone)

        # Per-node sheaf scores
        adj, contradiction_edges = self._build_weighted_adjacency(zone)
        sheaf_scores = self._compute_sheaf_consistency(zone, adj)

        return {
            "zone": zone_val,
            "timestamp": state.timestamp,
            "verdict": {
                "should_commit": state.should_commit,
                "regime": state.regime,
                "explanation": state.explanation,
            },
            "spectral": {
                "lambda2": state.lambda2,
                "lambda2_delta": state.lambda2_delta,
                "lambda2_stable": state.lambda2_stable,
                "lambda2_sufficient": state.lambda2_sufficient,
                "lambda_min_threshold": self.lambda_min,
                "epsilon_threshold": self.epsilon_spectral,
            },
            "sheaf": {
                "mean_consistency": state.mean_sheaf_consistency,
                "tau_threshold": self.tau_sheaf,
                "sufficient": state.sheaf_sufficient,
                "inconsistent_nodes": state.nodes_inconsistent,
                "mode": "restriction_map" if (sheaf_scores and sheaf_scores[0].restriction_map_mode) else "census",
                "per_node": [
                    {
                        "node_id": s.node_id,
                        "score": round(s.score, 4),
                        "supports": s.support_count,
                        "contradictions": s.contradiction_count,
                        "neighbors": s.neighbor_count,
                        "restriction_map_mode": s.restriction_map_mode,
                        "mean_neighbor_similarity": round(s.mean_neighbor_similarity, 4),
                    }
                    for s in sorted(sheaf_scores, key=lambda x: x.score)[:20]
                ],
            },
            "topology": {
                "node_count": state.node_count,
                "edge_count": state.edge_count,
                "giant_component_fraction": state.giant_component_fraction,
                "unresolved_contradictions": state.unresolved_contradictions,
                "contradictions_blocking": state.contradictions_blocking,
                "contradiction_edges": state.contradiction_edges,
                "lambda2_includes_contradiction_repulsion": True,
            },
            "bridge_dependencies": [
                {
                    "from_node": dep.from_node_id,
                    "to_zone": dep.to_zone,
                    "to_node": dep.to_node_id,
                    "to_zone_committable": dep.to_zone_committable,
                    "to_zone_blocked": dep.to_zone_blocked,
                    "to_zone_lambda2": dep.to_zone_lambda2,
                    "edge_weight": dep.edge_weight,
                }
                for dep in state.bridge_dependencies
            ],
            "fragile_bridges": [
                {"node_id": node_id, "d_value": round(d_val, 4)}
                for node_id, d_val in state.fragile_bridge_nodes
            ],
            "history": [
                {
                    "timestamp": s.timestamp,
                    "lambda2": round(s.lambda2, 6),
                    "nodes": s.node_count,
                    "edges": s.edge_count,
                }
                for s in self._history.get(zone_val, [])
            ],
            "field_theory": (
                {
                    "regime":            state.regime_state.regime.name,
                    "lambda2":           state.regime_state.lambda2,
                    "dlambda2_dt":       state.regime_state.dlambda2_dt,
                    "surgery_triggered": state.regime_state.surgery_triggered,
                    "derivative_warning": state.derivative_warning,
                    "explanation":       state.regime_state.explanation,
                    "fte_status":        self._fte.status() if self._fte is not None else None,
                }
                if state.regime_state is not None
                else {"attached": False}
            ),
        }

    def spectral_history(self, zone) -> List[SpectralSnapshot]:
        """Return the λ₂ history for a zone (for plotting / trend analysis)."""
        zone_val = zone.value if hasattr(zone, 'value') else str(zone)
        return list(self._history.get(zone_val, []))

    # ── Internal: adjacency construction ────────────────────

    def _build_weighted_adjacency(self, zone) -> Tuple[Dict[str, Dict[str, float]], int]:
        """
        Build a signed weighted adjacency dict from GraphEngine's in-memory cache.

        Respects GateLevel — BLOCK_UNTIL_RESOLVED edges are excluded.
        Weights edges by kind using EDGE_KIND_WEIGHTS, which now includes
        negative values for contradiction edges. This feeds the signed
        normalized Laplacian so that contradictions actively reduce lambda2
        rather than being invisible to the spectral computation.

        Returns: (adj, contradiction_edge_count)
            adj — node_id -> {neighbor_id: signed_weight}
                positive weights = supporting relationships
                negative weights = contradicting relationships
            contradiction_edge_count — number of contradiction edges present
                (for diagnostic reporting)
        """
        adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        zone_adj = self.graph._adj.get(zone, {})

        blocked = self.graph._blocked_nodes
        contradiction_count = 0

        # Collect all nodes in this zone from the memory library.
        # This ensures isolated nodes are included, not just those with edges.
        all_zone_nodes: Set[str] = set()
        if self.memory_library is not None:
            try:
                all_mems = self.memory_library.retrieve_all()
                if all_mems:
                    for mem in all_mems:
                        if mem.get("metadata", {}).get("zone") == zone.value or \
                           mem.get("zone") == zone.value:
                            all_zone_nodes.add(mem.get("memory_id") or mem.get("id", ""))
            except Exception:
                # Fallback if memory library access fails
                pass

        # Initialize adjacency dict for all nodes in the zone, including isolated ones
        for node_id in zone_adj.keys():
            if node_id not in blocked:
                all_zone_nodes.add(node_id)

        for node_id in all_zone_nodes:
            if node_id and node_id not in blocked:
                adj.setdefault(node_id, defaultdict(float))

        for src_id, neighbors in zone_adj.items():
            if src_id in blocked:
                continue
            adj.setdefault(src_id, defaultdict(float))

            for dst_id, kind, weight, gate in neighbors:
                if dst_id in blocked:
                    continue
                gate_val = gate.value if hasattr(gate, 'value') else str(gate)
                if gate_val == "block_until_resolved":
                    continue

                kind_val = kind.value if hasattr(kind, 'value') else str(kind)
                kind_weight = EDGE_KIND_WEIGHTS.get(kind_val, 0.5)

                # Skip edges with no defined relationship to coherence
                if kind_weight == 0.0:
                    continue

                # Edge weight = structural confidence × kind sign+magnitude
                effective_weight = float(weight) * kind_weight

                if kind_weight < 0.0:
                    contradiction_count += 1
                    # For contradictions: take the most negative (strongest repulsion)
                    current = adj[src_id].get(dst_id, 0.0)
                    adj[src_id][dst_id] = min(current, effective_weight)
                    adj.setdefault(dst_id, defaultdict(float))
                    adj[dst_id][src_id] = min(adj[dst_id].get(src_id, 0.0), effective_weight)
                else:
                    # For supports: take the strongest positive weight
                    current = adj[src_id].get(dst_id, 0.0)
                    adj[src_id][dst_id] = max(current, effective_weight)
                    adj.setdefault(dst_id, defaultdict(float))
                    adj[dst_id][src_id] = max(adj[dst_id].get(src_id, 0.0), effective_weight)

        # Convert inner defaultdicts to plain dicts
        # Divide by 2 — each undirected edge appears from both src and dst
        return (
            {k: dict(v) for k, v in adj.items()},
            contradiction_count // 2,
        )

    # ── Internal: cross-zone seam adjacency ─────────────────

    def _build_seam_adjacency(
        self,
        zone_a,
        zone_b,
    ) -> Tuple[Dict[str, Dict[str, float]], List[str], Dict]:
        """
        Build a cross-zone adjacency spanning zone_a and zone_b.

        Used to compute D(u) fragile bridge signals at zone seams.
        Unlike _build_weighted_adjacency(), which is limited to a single
        zone, this method collects edges that cross the seam between
        zone_a and zone_b so displacement_signal() has real cross-zone
        edges to evaluate.

        Algorithm
        ---------
        1. Build a source-node → zone mapping from _adj keys.  Source
           nodes are authoritative for zone assignment; destination nodes
           that only appear as targets (not sources) inherit the zone of
           the adjacency they were reached from.
        2. For each zone in {zone_a, zone_b}, traverse its adjacency and
           collect only edges whose destination belongs to the OTHER zone.
        3. Return the merged adjacency, sorted node list, and node_zones map.

        Returns
        -------
        seam_adj : Dict[str, Dict[str, float]]
            Weighted adjacency containing only cross-seam edges.
        seam_nodes : List[str]
            All nodes in zone_a ∪ zone_b (sorted), including isolated ones.
        node_zones : Dict[str, GraphZone]
            Correct zone assignment for every node in seam_nodes.
        """
        blocked = self.graph._blocked_nodes

        # Step 1: build authoritative source-node → zone mapping
        src_to_zone: Dict[str, object] = {}
        for gz in type(zone_a).__members__.values():          # iterate all GraphZone members
            for nid in self.graph._adj.get(gz, {}).keys():
                if nid not in blocked:
                    src_to_zone[nid] = gz

        seam_adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Step 2: collect cross-seam edges from both sides
        for src_zone, dst_zone in ((zone_a, zone_b), (zone_b, zone_a)):
            for src_id, neighbors in self.graph._adj.get(src_zone, {}).items():
                if src_id in blocked:
                    continue
                for dst_id, kind, weight, gate in neighbors:
                    if dst_id in blocked:
                        continue
                    # Only include edges where dst is confirmed in the other zone
                    if src_to_zone.get(dst_id) != dst_zone:
                        continue
                    gate_val = gate.value if hasattr(gate, 'value') else str(gate)
                    if gate_val == "block_until_resolved":
                        continue
                    kind_val = kind.value if hasattr(kind, 'value') else str(kind)
                    kind_weight = EDGE_KIND_WEIGHTS.get(kind_val, 0.5)
                    if kind_weight == 0.0:
                        continue
                    effective_weight = float(weight) * kind_weight
                    if kind_weight < 0.0:
                        seam_adj[src_id][dst_id] = min(
                            seam_adj[src_id].get(dst_id, 0.0), effective_weight
                        )
                        seam_adj[dst_id][src_id] = min(
                            seam_adj[dst_id].get(src_id, 0.0), effective_weight
                        )
                    else:
                        seam_adj[src_id][dst_id] = max(
                            seam_adj[src_id].get(dst_id, 0.0), effective_weight
                        )
                        seam_adj[dst_id][src_id] = max(
                            seam_adj[dst_id].get(src_id, 0.0), effective_weight
                        )

        # Step 3: ensure every node in both zones appears (even if isolated)
        for gz in (zone_a, zone_b):
            for nid in self.graph._adj.get(gz, {}).keys():
                if nid not in blocked:
                    seam_adj.setdefault(nid, defaultdict(float))

        seam_nodes = sorted(seam_adj.keys())
        node_zones = {nid: src_to_zone.get(nid, zone_a) for nid in seam_nodes}

        return (
            {k: dict(v) for k, v in seam_adj.items()},
            seam_nodes,
            node_zones,
        )

    # ── Internal: Laplacian construction ────────────────────

    def _build_normalized_laplacian(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        node_idx: Dict[str, int],
    ) -> Tuple:
        """
        Build the signed normalized Laplacian L_s = D_|A|^{-1/2} (D_|A| - A) D_|A|^{-1/2}.

        Unlike the standard normalized Laplacian which requires non-negative
        edge weights, the signed version handles both positive (supporting) and
        negative (contradicting) weights correctly:

        - D_|A| is the degree matrix built from absolute-value row sums.
          This ensures D is positive definite regardless of sign.
        - A contains the raw signed weights (positive=support, negative=contradiction).
        - The resulting L_s has eigenvalues in [-1, 2].

        Effect on lambda2:
          - A zone with only SUPPORTS has high lambda2 (coherent, well-connected).
          - A zone with equal SUPPORTS and CONTRADICTS has lambda2 near or below 0
            (net incoherence, contradictions cancel the support structure).
          - Contradictions actively suppress lambda2 rather than being invisible.

        This is the correct spectral representation for VM's belief graph, where
        contradictions are structural facts about disagreement, not absences.

        Returns (L_matrix, edge_count) where edge_count counts all edges
        (both positive and negative) and L_matrix is a numpy array.
        """
        n = len(nodes)
        A = np.zeros((n, n), dtype=np.float64)
        edge_count = 0

        for src_id, neighbors in adj.items():
            i = node_idx.get(src_id)
            if i is None:
                continue
            for dst_id, w in neighbors.items():
                j = node_idx.get(dst_id)
                if j is None or j <= i:  # upper triangle only, then symmetrize
                    continue
                A[i, j] = w
                A[j, i] = w
                edge_count += 1

        # Degree matrix from absolute-value row sums.
        # Using |w| ensures D is positive definite even with negative edges.
        abs_degrees = np.abs(A).sum(axis=1)

        D_inv_sqrt = np.zeros(n, dtype=np.float64)
        nonzero = abs_degrees > 0
        D_inv_sqrt[nonzero] = 1.0 / np.sqrt(abs_degrees[nonzero])

        # Signed normalized Laplacian: L_s = D_|A|^{-1/2} (D_|A| - A) D_|A|^{-1/2}
        D_abs = np.diag(abs_degrees)
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        L = D_inv_sqrt_mat @ (D_abs - A) @ D_inv_sqrt_mat

        return L, edge_count

    # ── Internal: Fiedler value ──────────────────────────────

    def _compute_fiedler_value(self, L: np.ndarray, n: int) -> float:
        """
        Compute lambda2 — the second-smallest eigenvalue of the signed Laplacian.

        For the signed normalized Laplacian, eigenvalues lie in [-1, 2].
        Unlike the standard Laplacian, lambda1 is NOT guaranteed to be 0 —
        it can be negative when contradictions dominate the graph structure.

        Interpretation:
          lambda2 > lambda_min  — coherent support structure dominates
          lambda2 near 0        — support and contradiction roughly balanced
          lambda2 < 0           — contradiction structure dominates; incoherent

        The stopping criterion (lambda2 > lambda_min AND stable) still works
        correctly — a zone heavy with contradictions will fail lambda2_sufficient
        and the system will continue rather than commit.

        Uses scipy sparse eigensolver when available (faster for n > 20),
        falls back to numpy for small graphs or missing scipy.
        """
        try:
            if _SCIPY_AVAILABLE and n > 20:
                L_sparse = csr_matrix(L)
                # which='SA' = smallest algebraic (not magnitude) — correct for
                # signed Laplacian where eigenvalues can be negative.
                # which='SM' would return values near 0 in absolute terms,
                # missing negative eigenvalues that are the most important signal.
                eigenvalues = eigsh(
                    L_sparse, k=min(3, n - 1),
                    which='SA', tol=1e-6, maxiter=1000,  # type: ignore[arg-type]
                    return_eigenvectors=False,
                )
                eigenvalues = np.sort(np.real(eigenvalues))
                lambda2 = float(eigenvalues[1]) if len(eigenvalues) >= 2 else 0.0
            else:
                # Dense numpy path for small graphs
                eigenvalues = np.linalg.eigvalsh(L)
                eigenvalues = np.sort(eigenvalues)
                lambda2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

            # No floor clamp — negative lambda2 is meaningful signal,
            # not numerical noise. It indicates contradiction-dominated structure.
            return lambda2

        except Exception as e:
            logger.warning(f"Fiedler value computation failed: {e}")
            return 0.0

    # ── Internal: Sheaf consistency ──────────────────────────

    def _compute_sheaf_consistency(
        self, zone, adj: Dict[str, Dict[str, float]]
    ) -> List[SheavConsistencyScore]:
        """
        Compute local sheaf consistency for each node.

        For each node, examine the edge kinds incident to it
        in the original (unweighted) adjacency:
          - SUPPORTS-type edges: evidence of local coherence
          - CONTRADICTS edges: evidence of local incoherence
          - Other edges: neutral

        Consistency score = support_fraction penalized by contradiction_fraction.

        A node with 10 SUPPORTS and 0 CONTRADICTS = score 1.0.
        A node with 5 SUPPORTS and 5 CONTRADICTS = score ~0.3.
        A node with 0 edges = score 0.5 (unknown, neither coherent nor not).
        """
        zone_adj = self.graph._adj.get(zone, {})
        blocked = self.graph._blocked_nodes
        scores = []

        for node_id in adj:
            if node_id in blocked:
                continue

            neighbors = zone_adj.get(node_id, [])
            support_count = 0
            contradiction_count = 0
            dependency_count = 0
            neighbor_count = 0

            for dst_id, kind, weight, gate in neighbors:
                gate_val = gate.value if hasattr(gate, 'value') else str(gate)
                if gate_val == "block_until_resolved":
                    continue
                if dst_id in blocked:
                    continue

                neighbor_count += 1
                kind_val = kind.value if hasattr(kind, 'value') else str(kind)

                if kind_val in ("supports", "evidence_supports_decision",
                                "fact_updates_belief", "implements",
                                "decision_updates_preference"):
                    support_count += 1
                elif kind_val in ("contradicts", "evidence_contradicts_preference"):
                    contradiction_count += 1
                elif kind_val in ("depends_on", "decision_requires_evidence"):
                    dependency_count += 1

            # Score: base on support fraction, penalize contradictions
            if neighbor_count == 0:
                score = 0.5  # no information
            else:
                support_frac = support_count / neighbor_count
                contradiction_frac = contradiction_count / neighbor_count
                # Consistency degrades sharply with contradictions
                score = support_frac * (1.0 - 2.0 * contradiction_frac)
                score = max(0.0, min(1.0, score))

            scores.append(SheavConsistencyScore(
                node_id=node_id,
                score=score,
                support_count=support_count,
                contradiction_count=contradiction_count,
                dependency_count=dependency_count,
                neighbor_count=neighbor_count,
            ))

        return scores

    # ── Internal: connected components ──────────────────────

    def _find_connected_components(
        self, adj: Dict[str, Dict[str, float]], nodes: List[str]
    ) -> Tuple[float, Dict[str, int]]:
        """
        BFS to find connected components in the weighted adjacency graph.

        Returns:
            giant_fraction: fraction of nodes in the largest component
            cluster_membership: node_id -> cluster_id
                cluster 0 = giant component (or first found if tied)
        """
        visited: Dict[str, int] = {}
        cluster_id = 0
        cluster_sizes: Dict[int, int] = {}

        for node in nodes:
            if node in visited:
                continue
            # BFS from this node
            queue = [node]
            visited[node] = cluster_id
            size = 1
            while queue:
                current = queue.pop()
                for neighbor in adj.get(current, {}):
                    if neighbor not in visited:
                        visited[neighbor] = cluster_id
                        queue.append(neighbor)
                        size += 1
            cluster_sizes[cluster_id] = size
            cluster_id += 1

        if not cluster_sizes:
            return 0.0, {}

        # Find giant component
        max_cluster = max(cluster_sizes, key=lambda k: cluster_sizes[k])
        max_size = cluster_sizes[max_cluster]
        giant_fraction = max_size / len(nodes) if nodes else 0.0

        # Remap so giant component = cluster 0
        remap = {max_cluster: 0}
        next_id = 1
        for cid in cluster_sizes:
            if cid != max_cluster:
                remap[cid] = next_id
                next_id += 1

        cluster_membership = {nid: remap[cid] for nid, cid in visited.items()}
        return giant_fraction, cluster_membership

    # ── Internal: restriction-map sheaf consistency ─────────

    def _compute_sheaf_consistency_restriction(
        self, zone, adj: Dict[str, Dict[str, float]]
    ) -> List["SheavConsistencyScore"]:
        """
        Restriction-map sheaf consistency using pairwise embedding similarity.

        For each node u, examines each edge (u, v) and asks: does the
        similarity between embed(u) and embed(v) match what the edge
        type predicts?

          SUPPORTS/IMPLEMENTS/REFINES  → expects high similarity (>0.6)
          CONTRADICTS                  → expects low or negative similarity (<0.2)
          DEPENDS_ON                   → neutral expectation (0.3–0.7)
          Other                        → neutral

        Score for node u = mean over all incident edges of:
            1.0 - |actual_similarity - expected_similarity|

        This catches transitive incoherence that has no explicit
        CONTRADICTS edge: if A SUPPORTS B and B SUPPORTS C but
        embed(A) and embed(C) are orthogonal, A's score is penalized
        even though no direct contradiction edge exists between A and C.

        Requires self.semantic_index to have embeddings pre-computed.
        Falls back to census mode if embeddings unavailable for a node.
        """
        si = self.semantic_index
        blocked = self.graph._blocked_nodes
        zone_adj = self.graph._adj.get(zone, {})
        scores = []

        # Expected similarity by edge kind
        expected_sim: Dict[str, float] = {
            "supports":                        0.75,
            "evidence_supports_decision":      0.70,
            "fact_updates_belief":             0.65,
            "implements":                      0.70,
            "refines":                         0.65,
            "decision_updates_preference":     0.60,
            "decision_refines_policy":         0.60,
            "about":                           0.50,
            "duplicate_of":                    0.90,
            "temporal_next":                   0.40,
            "semantic_similarity":             0.60,
            "depends_on":                      0.45,
            "decision_requires_evidence":      0.45,
            "contradicts":                     0.10,
            "evidence_contradicts_preference": 0.15,
        }

        for node_id in adj:
            if node_id in blocked:
                continue

            neighbors_raw = zone_adj.get(node_id, [])
            support_count = 0
            contradiction_count = 0
            dependency_count = 0
            neighbor_count = 0
            edge_scores = []

            src_emb = si.embeddings.get(node_id) if si else None

            for dst_id, kind, weight, gate in neighbors_raw:
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if gate_val == "block_until_resolved":
                    continue
                if dst_id in blocked:
                    continue

                neighbor_count += 1
                kind_val = kind.value if hasattr(kind, "value") else str(kind)

                if kind_val in ("supports", "evidence_supports_decision",
                                "fact_updates_belief", "implements",
                                "decision_updates_preference"):
                    support_count += 1
                elif kind_val in ("contradicts", "evidence_contradicts_preference"):
                    contradiction_count += 1
                elif kind_val in ("depends_on", "decision_requires_evidence"):
                    dependency_count += 1

                # Restriction map: compare actual similarity to expected
                if src_emb is not None and si is not None:
                    dst_emb = si.embeddings.get(dst_id)
                    if dst_emb is not None:
                        # Cosine similarity
                        norm_src = float(np.linalg.norm(src_emb))
                        norm_dst = float(np.linalg.norm(dst_emb))
                        if norm_src > 0 and norm_dst > 0:
                            cos_sim = float(
                                np.dot(src_emb, dst_emb) / (norm_src * norm_dst)
                            )
                            # Remap [-1,1] -> [0,1] for comparison
                            actual_sim = (cos_sim + 1.0) / 2.0
                            exp_sim = expected_sim.get(kind_val, 0.5)
                            # Score = 1 - deviation from expectation
                            edge_score = 1.0 - abs(actual_sim - exp_sim)
                            edge_scores.append(edge_score)

            if not edge_scores:
                # No embeddings available for this node's edges: fall back to census
                if neighbor_count == 0:
                    score = 0.5
                else:
                    support_frac = support_count / neighbor_count
                    contradiction_frac = contradiction_count / neighbor_count
                    score = support_frac * (1.0 - 2.0 * contradiction_frac)
                    score = max(0.0, min(1.0, score))
                mean_sim = 0.0
                restriction_mode = False
            else:
                score = max(0.0, min(1.0, sum(edge_scores) / len(edge_scores)))
                mean_sim = score
                restriction_mode = True

            scores.append(SheavConsistencyScore(
                node_id=node_id,
                score=score,
                support_count=support_count,
                contradiction_count=contradiction_count,
                dependency_count=dependency_count,
                neighbor_count=neighbor_count,
                restriction_map_mode=restriction_mode,
                mean_neighbor_similarity=mean_sim,
            ))

        return scores

    # ── Internal: cross-zone bridge dependency check ─────────

    def _check_bridge_dependencies(self, zone) -> List["BridgeDependency"]:
        """
        Check whether committing this zone is safe given cross-zone dependencies.

        Examines all outbound bridge edges from the zone being evaluated.
        For each bridge edge (this_zone, node_A) -> (other_zone, node_B):

        1. Build the other zone's adjacency (lightweight — no Laplacian needed).
        2. Check whether the other zone has hard-blocked nodes.
        3. Compute a lightweight lambda2 estimate for the other zone.
        4. Return a BridgeDependency record for each outbound bridge.

        The caller surfaces unresolved dependencies as a WARNING (not a hard
        block) in the explanation — the Archivist decides whether cross-zone
        unreadiness is a blocker for this specific commit.

        Returns an empty list if no bridge edges exist from this zone,
        or if _bridges is unavailable on the graph engine.
        """
        deps = []
        bridges = getattr(self.graph, "_bridges", {})
        zone_val = zone.value if hasattr(zone, "value") else str(zone)

        # Find all bridge edges originating in this zone
        for (from_zone, from_node_id), targets in bridges.items():
            fz = from_zone.value if hasattr(from_zone, "value") else str(from_zone)
            if fz != zone_val:
                continue

            for to_zone, to_node_id, weight, gate in targets:
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if gate_val == "block_until_resolved":
                    # Hard-blocked bridge edge — already caught by contradiction gate
                    continue

                tz = to_zone.value if hasattr(to_zone, "value") else str(to_zone)

                # Check target zone's blocked nodes
                to_zone_blocked = any(
                    rec.zone == to_zone
                    for rec in self.graph._blocked_nodes.values()
                    if hasattr(rec, "zone")
                )

                # Quick spectral estimate for the target zone
                try:
                    target_adj, _ = self._build_weighted_adjacency(to_zone)
                    target_nodes = sorted(target_adj.keys())
                    target_n = len(target_nodes)
                    if target_n >= 3:
                        target_idx = {nid: i for i, nid in enumerate(target_nodes)}
                        target_L, _ = self._build_normalized_laplacian(
                            target_adj, target_nodes, target_idx
                        )
                        target_lambda2 = self._compute_fiedler_value(target_L, target_n)
                    else:
                        target_lambda2 = 0.0
                except Exception:
                    target_lambda2 = 0.0

                to_zone_committable = (
                    target_lambda2 > self.lambda_min
                    and not to_zone_blocked
                )

                deps.append(BridgeDependency(
                    from_zone=zone_val,
                    from_node_id=from_node_id,
                    to_zone=tz,
                    to_node_id=to_node_id,
                    edge_weight=float(weight),
                    to_zone_committable=to_zone_committable,
                    to_zone_blocked=to_zone_blocked,
                    to_zone_lambda2=round(target_lambda2, 6),
                ))

        return deps

    # ── Internal: explanation builder ───────────────────────

    def _build_explanation(
        self,
        zone: str, should_commit: bool, regime: str,
        lambda2: float, lambda2_delta: float,
        lambda2_stable: bool, lambda2_sufficient: bool,
        mean_sheaf: float, sheaf_sufficient: bool,
        unresolved: int, blocking: bool,
        giant_fraction: float, n: int, edge_count: int,
        contradiction_edges: int = 0,
        bridge_deps: Optional[List] = None,
        fragile_bridge_nodes: Optional[List] = None,
        ft_regime: Optional["RegimeState"] = None,
        ft_derivative_warning: bool = False,
    ) -> str:

        bridge_deps = bridge_deps or []
        fragile_bridge_nodes = fragile_bridge_nodes or []
        contradiction_note = (
            f" ({contradiction_edges} contradiction edge(s) included with negative weight — "
            f"these suppressed λ₂ directly.)"
            if contradiction_edges > 0 else ""
        )

        unresolved_bridge_deps = [d for d in bridge_deps if not d.to_zone_committable]
        bridge_note = ""
        if unresolved_bridge_deps:
            zones = ", ".join(sorted({d.to_zone for d in unresolved_bridge_deps}))
            bridge_note = (
                f" WARNING: {len(unresolved_bridge_deps)} bridge edge(s) point to "
                f"zone(s) [{zones}] that are not yet committable. "
                "Committing this zone may be premature if it depends on their unresolved work."
            )

        fragile_bridge_note = ""
        if fragile_bridge_nodes:
            top_node, top_d = fragile_bridge_nodes[0]
            fragile_bridge_note = (
                f" FRAGILE BRIDGES: {len(fragile_bridge_nodes)} node(s) with elevated "
                f"D(u) displacement at the governance->rationale seam "
                f"(highest: {top_node!r} D={top_d:.3f}). "
                "Pre-contradiction tension detected — commit blocked until seam stabilises."
            )

        # Field-theory regime note — appended when FieldTheoryEngine is active.
        # Derivative warning fires BEFORE λ₂ drops below threshold, giving the
        # human intervention time ahead of Regime 3 entry.
        ft_note = ""
        if ft_regime is not None:
            ft_note = f" [FTE regime: {ft_regime.regime.name}"
            if ft_derivative_warning:
                ft_note += (
                    f"; DERIVATIVE WARNING: dλ₂/dt < −δ_c ({ft_regime.dlambda2_dt:.6f}) "
                    "— pre-emptive review queue active; human sign-offs requested"
                )
            ft_note += "]"
        elif ft_derivative_warning:
            # Derivative warning without a full regime object (defensive path)
            ft_note = " [DERIVATIVE WARNING: dλ₂/dt < −δ_c — pre-emptive review queue active]"

        if blocking:
            return (
                f"BLOCKED — {unresolved} unresolved contradiction(s) in zone '{zone}'. "
                "Reasoning cannot commit until a human resolves these. "
                f"Call graph.get_unresolved_contradictions() to inspect.{ft_note}"
            )

        if should_commit:
            return (
                f"COMMIT — Zone '{zone}' has entered the supercritical regime. "
                f"λ₂={lambda2:.4f} (>{self.lambda_min}, stable Δ={lambda2_delta:.4f}), "
                f"sheaf consistency={mean_sheaf:.2f} (>{self.tau_sheaf}). "
                f"{round(giant_fraction*100)}% of nodes in coherent component. "
                f"Reasoning has converged — safe to commit."
                f"{contradiction_note}{bridge_note}{ft_note}"
            )

        reasons = []
        if not lambda2_sufficient:
            reasons.append(
                f"λ₂={lambda2:.4f} below threshold {self.lambda_min} — "
                "belief graph is fragmented (subcritical)"
            )
        if not lambda2_stable:
            reasons.append(
                f"λ₂ still changing (Δ={lambda2_delta:.4f} > ε={self.epsilon_spectral}) — "
                "coherence has not stabilized"
            )
        if not sheaf_sufficient:
            reasons.append(
                f"Sheaf consistency={mean_sheaf:.2f} below threshold {self.tau_sheaf} — "
                "local belief neighborhoods are still inconsistent"
            )

        return (
            f"CONTINUE — Zone '{zone}' is in the {regime} regime. "
            + "; ".join(reasons) + ". "
            f"({n} nodes, {edge_count} coherence edges, "
            f"{round(giant_fraction*100)}% in giant component)"
            f"{contradiction_note}{bridge_note}{fragile_bridge_note}{ft_note}"
        )
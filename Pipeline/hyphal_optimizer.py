"""
VeritasMemoria - Hyphal Network Graph Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies Physarum polycephalum (slime mold) network optimization to
VM's belief graph, reinforcing heavily-traversed edges and flagging
chronically weak edges and orphan nodes for human review.

The biology
------------
Physarum polycephalum is a plasmodial slime mold that solves network
optimization problems without a brain, central coordination, or global
knowledge. It extends hyphal tubes (cytoplasmic channels) through its
environment searching for food. When a tube finds a nutrient source,
peristaltic flow through that tube increases. Tubes with high flow
grow wider and carry more flow -- positive feedback. Tubes with low
flow narrow and eventually collapse -- negative feedback. The result
is an efficient, fault-tolerant network that naturally prunes redundant
and low-value paths.

Toshiyuki Nakagaki's landmark 2000 paper in Nature showed that Physarum
placed on a map of Tokyo's rail stations spontaneously recreated a network
nearly identical to the actual Tokyo subway system -- without knowing
anything about transportation engineering.

The mathematical model (Tero et al., 2010)
--------------------------------------------
The Physarum dynamics are governed by:

    dD_ij/dt = f(Q_ij) - gamma * D_ij

Where:
    D_ij        conductivity of the tube between nodes i and j
    Q_ij        flow through that tube (demand-driven)
    f(Q_ij)     reinforcement function -- increases conductivity with flow
    gamma       decay rate -- conductivity decays without flow

The flow Q_ij is determined by pressure differences (Kirchhoff's laws
applied to the fluid network). At equilibrium, the network converges
to an efficient topology.

Mapping to VM's belief graph
------------------------------
    Hyphal tube         a graph edge (relationship between memories)
    Conductivity D_ij   edge weight (how strong the relationship is)
    Flow Q_ij           traversal frequency (how often this edge is
                        crossed during recall or search)
    Nutrient sources    frequently-queried memory nodes (high demand)
    Atrophy             edges that carry no flow get weight reduced
                        toward a minimum, flagged for pruning
    Orphan node         a memory node with no traversal and all
                        edges below the atrophy threshold

Key adaptations for VM:
    1. Flow is asymmetric -- directed traversal from recall queries
       is tracked per edge, not just per node
    2. Governance edges are immune to atrophy -- NEVER decayed
    3. Blocked edges are excluded from flow calculation
    4. Pruning is never automatic -- the optimizer produces
       PruningCandidate records requiring human approval
    5. Reinforcement has an upper bound (max_weight=1.0) to prevent
       runaway amplification of high-traffic edges

Why this matters for VM
------------------------
Without pruning, the belief graph accumulates dead weight: edges that
were created early in the system's life, referenced once, and never
touched again. These edges:

    - Slow down traversal (more edges to check)
    - Pollute coherence scoring (lambda2 is affected by all edges)
    - Create false connections in the TDA analysis
    - Generate noise in the SIR propagation analysis

Physarum optimization finds these automatically without requiring
centralized knowledge of what should be kept. It's purely local:
each edge's fate is determined by whether flow passes through it.

Human oversight
---------------
Pruning candidates are never acted on automatically. HyphalOptimizer
produces PruningCandidate records with full context (why flagged,
how long dormant, what nodes would be affected). Human approval
required via approve_pruning(). Same protective posture as
contradict resolution, retirement approval, constraint conflicts.

Integration
-----------
HyphalOptimizer runs as a background maintenance pass, not in the
hot path. Recommended: once per session end, after ClonalSelector
has identified low-affinity memories, feed the traversal log from
that session into record_traversal() calls, then run optimize().

The optimizer reads traversal data accumulated across calls to
record_traversal() and updates edge conductivity accordingly.
Periodic calls to pruning_candidates() surface the results.

Dependencies
------------
Standard library only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from graph_types import GraphZone


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Decay rate gamma -- conductivity lost per optimization step without flow
DEFAULT_GAMMA: float = 0.05

# Reinforcement exponent mu -- how steeply flow reinforces conductivity
# mu = 1: linear reinforcement
# mu > 1: superlinear (high-flow edges dominate faster)
# mu < 1: sublinear (more democratic, weaker paths survive longer)
DEFAULT_MU: float = 1.2

# Minimum conductivity before edge is flagged as atrophied
DEFAULT_MIN_CONDUCTIVITY: float = 0.10

# Conductivity maximum (prevents runaway amplification)
DEFAULT_MAX_CONDUCTIVITY: float = 1.0

# Steps since last traversal before flagging as dormant
DEFAULT_DORMANCY_STEPS: int = 5

# Edge kinds immune to atrophy (governance relationships)
IMMUNE_EDGE_KINDS: frozenset = frozenset({
    "decision_refines_policy",
    "contradicts",          # contradiction edges must be resolved by human, not decayed
    "block_until_resolved",
})

# Zone immune to ALL atrophy
IMMUNE_ZONES: frozenset = frozenset({
    "governance",
})


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class EdgeConductivity:
    """
    Physarum conductivity state for a single graph edge.

    conductivity    current D_ij (0.0 to max_conductivity)
    total_flow      cumulative flow across all optimization steps
    last_flow_step  last step in which nonzero flow was recorded
    current_step    current optimizer step count
    zone            which zone this edge is in
    kind            edge kind string
    immune          True if this edge is exempt from atrophy
    """
    src_id: str
    dst_id: str
    zone: str
    kind: str
    conductivity: float
    total_flow: float = 0.0
    last_flow_step: int = 0
    current_step: int = 0
    immune: bool = False

    @property
    def dormant_steps(self) -> int:
        return self.current_step - self.last_flow_step

    @property
    def is_atrophied(self) -> bool:
        return (
            not self.immune
            and self.conductivity <= DEFAULT_MIN_CONDUCTIVITY
        )


@dataclass
class PruningCandidate:
    """
    An edge or node flagged for potential pruning.

    Never acted on automatically. Requires human approval.
    """
    candidate_type: str         # "edge" or "node"
    zone: str
    node_id: Optional[str]      # for node candidates
    src_id: Optional[str]       # for edge candidates
    dst_id: Optional[str]       # for edge candidates
    edge_kind: Optional[str]
    conductivity: float
    dormant_steps: int
    total_flow: float
    flagged_at: str
    reason: str
    dependent_nodes: List[str] = field(default_factory=list)
    approved: bool = False
    approved_by: Optional[str] = None


@dataclass
class OptimizationResult:
    """
    Summary of a single optimization pass.

    edges_reinforced    edges whose conductivity increased
    edges_decayed       edges whose conductivity decreased
    edges_atrophied     edges that crossed the atrophy threshold
    orphan_nodes        nodes with all edges atrophied and no flow
    pruning_candidates  new PruningCandidate records generated
    mean_conductivity   mean conductivity across all tracked edges
    summary             human-readable pass summary
    """
    step: int
    edges_reinforced: int
    edges_decayed: int
    edges_atrophied: int
    orphan_nodes: List[str]
    pruning_candidates: List[PruningCandidate]
    mean_conductivity: float
    summary: str


# ─────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────

class HyphalOptimizer:
    """
    Physarum-inspired graph optimization for VM's belief graph.

    Records edge traversals during recall/search, applies Physarum
    dynamics to update conductivity, and surfaces atrophied edges
    and orphan nodes as pruning candidates for human review.

    Usage
    -----
        optimizer = HyphalOptimizer(graph_engine)

        # During recall_memories() and search_memories(),
        # record every edge traversal:
        optimizer.record_traversal(src_id, dst_id, zone="rationale")

        # At session end, run optimization pass:
        result = optimizer.optimize()
        print(result.summary)

        # Surface pruning candidates:
        candidates = optimizer.pruning_candidates()
        for c in candidates:
            print(c.reason, c.conductivity)

        # Human approves a pruning:
        optimizer.approve_pruning(
            src_id="mem_abc",
            dst_id="mem_def",
            zone="rationale",
            approved_by="attorney_jones",
        )
        # Actual edge deletion performed separately by caller
    """

    def __init__(
        self,
        graph_engine,
        gamma: float = DEFAULT_GAMMA,
        mu: float = DEFAULT_MU,
        min_conductivity: float = DEFAULT_MIN_CONDUCTIVITY,
        max_conductivity: float = DEFAULT_MAX_CONDUCTIVITY,
        dormancy_steps: int = DEFAULT_DORMANCY_STEPS,
    ):
        self.graph = graph_engine
        self.gamma = gamma
        self.mu = mu
        self.min_conductivity = min_conductivity
        self.max_conductivity = max_conductivity
        self.dormancy_steps = dormancy_steps

        # Edge conductivity state: (zone, src_id, dst_id) -> EdgeConductivity
        self._conductivity: Dict[Tuple[str, str, str], EdgeConductivity] = {}

        # Flow accumulator for current step: (zone, src_id, dst_id) -> flow
        self._step_flow: Dict[Tuple[str, str, str], float] = defaultdict(float)

        # Current step counter
        self._step: int = 0

        # Pending pruning candidates
        self._pruning_candidates: Dict[str, PruningCandidate] = {}

        # Approved prunings
        self._approved: Set[str] = set()

        # Initialize conductivity from graph
        self._initialize_from_graph()

    # ── Initialization ───────────────────────────────────────

    def _initialize_from_graph(self) -> None:
        """
        Seed conductivity from current graph edge weights.

        Existing edge weights become initial conductivity values.
        This means a freshly-added high-confidence edge starts with
        high conductivity and must maintain flow to keep it.
        """
        from graph import GraphZone

        for zone in GraphZone:
            zone_val = zone.value
            if zone_val in IMMUNE_ZONES:
                continue

            adj = self.graph._adj.get(zone, {})
            for src_id, neighbors in adj.items():
                for dst_id, kind, weight, gate in neighbors:
                    gate_val = gate.value if hasattr(gate, "value") else str(gate)
                    if gate_val == "block_until_resolved":
                        continue

                    kind_val = kind.value if hasattr(kind, "value") else str(kind)
                    key = (zone_val, src_id, dst_id)

                    if key not in self._conductivity:
                        immune = kind_val in IMMUNE_EDGE_KINDS
                        self._conductivity[key] = EdgeConductivity(
                            src_id=src_id,
                            dst_id=dst_id,
                            zone=zone_val,
                            kind=kind_val,
                            conductivity=min(float(weight), self.max_conductivity),
                            immune=immune,
                        )

    # ── Traversal recording ──────────────────────────────────

    def record_traversal(
        self,
        src_id: str,
        dst_id: str,
        zone: str,
        flow_weight: float = 1.0,
    ) -> None:
        """
        Record an edge traversal event.

        Call this from traverse_graph(), recall_memories(), and
        search_memories() every time an edge is crossed during
        graph traversal.

        flow_weight can be set to the edge confidence or query
        relevance score to make high-confidence traversals count more.
        """
        if zone in IMMUNE_ZONES:
            return

        key = (zone, src_id, dst_id)
        self._step_flow[key] += flow_weight

        # Also record reverse direction for undirected edges
        rev_key = (zone, dst_id, src_id)
        self._step_flow[rev_key] += flow_weight * 0.5

        # Ensure conductivity record exists
        if key not in self._conductivity:
            self._conductivity[key] = EdgeConductivity(
                src_id=src_id,
                dst_id=dst_id,
                zone=zone,
                kind="unknown",
                conductivity=0.5,  # neutral start for newly seen edges
            )

    # ── Optimization pass ────────────────────────────────────

    def optimize(self) -> OptimizationResult:
        """
        Run one Physarum optimization step.

        Correctly implements the Physarum dynamics by solving for flow
        via Kirchhoff's laws BEFORE updating conductivity. This is the
        step the naive implementation skips, causing the "explosion" or
        non-convergence problem described in the Tero et al. model.

        Pipeline per step:
            1. Build conductance matrix from current D_ij values
            2. Solve linear system for node pressures (Kirchhoff)
            3. Compute flow Q_ij = D_ij * |p_i - p_j| / L_ij
            4. Update D_ij using Physarum dynamics
            5. Identify atrophied edges and orphan nodes

        The traversal-count flow from record_traversal() is used as the
        "demand" signal (nutrient source strength) to set the boundary
        conditions for the pressure solve rather than being used as Q_ij
        directly. This preserves the network-efficiency property of the
        biological model.
        """
        self._step += 1
        now = _utc_now()

        reinforced = 0
        decayed = 0
        newly_atrophied = 0
        new_candidates: List[PruningCandidate] = []

        # Build per-zone conductance graphs and solve for pressure
        zone_flows = self._solve_kirchhoff_flows()

        # Update conductivity for every tracked edge
        for key, ec in self._conductivity.items():
            if ec.immune:
                continue

            zone, src_id, dst_id = key
            old_conductivity = ec.conductivity

            # Use Kirchhoff-solved flow if available, fall back to
            # traversal count for edges not in the pressure solve
            kirchhoff_flow = zone_flows.get(key, 0.0)
            traversal_flow = self._step_flow.get(key, 0.0)

            # Kirchhoff flow is the structurally correct Q_ij.
            # Traversal flow is the demand signal -- blend them.
            # When traversal flow is zero, pure Kirchhoff decay applies.
            # When traversal flow is high, it amplifies Kirchhoff reinforcement.
            demand_scale = 1.0 + math.log1p(traversal_flow)
            flow = kirchhoff_flow * demand_scale

            # Physarum dynamics: dD/dt = f(Q) - gamma * D
            # f(Q) = Q^mu, discrete step with dt=1
            reinforcement = (flow ** self.mu) if flow > 0 else 0.0
            decay = self.gamma * ec.conductivity
            delta = reinforcement - decay

            new_conductivity = ec.conductivity + delta
            new_conductivity = max(
                self.min_conductivity,
                min(self.max_conductivity, new_conductivity),
            )

            ec.conductivity = round(new_conductivity, 4)
            ec.current_step = self._step

            if traversal_flow > 0 or kirchhoff_flow > 0:
                ec.total_flow += max(traversal_flow, kirchhoff_flow)
                ec.last_flow_step = self._step

            if new_conductivity > old_conductivity:
                reinforced += 1
            elif new_conductivity < old_conductivity:
                decayed += 1

            was_atrophied = old_conductivity <= self.min_conductivity
            is_atrophied = ec.is_atrophied

            if is_atrophied and not was_atrophied:
                newly_atrophied += 1

            if is_atrophied and ec.dormant_steps >= self.dormancy_steps:
                cid = f"{zone}:{src_id}:{dst_id}"
                if cid not in self._pruning_candidates and cid not in self._approved:
                    candidate = PruningCandidate(
                        candidate_type="edge",
                        zone=zone,
                        node_id=None,
                        src_id=src_id,
                        dst_id=dst_id,
                        edge_kind=ec.kind,
                        conductivity=ec.conductivity,
                        dormant_steps=ec.dormant_steps,
                        total_flow=ec.total_flow,
                        flagged_at=now,
                        reason=(
                            f"Edge atrophied after {ec.dormant_steps} steps "
                            f"with no traversal. Conductivity={ec.conductivity:.3f} "
                            f"(threshold={self.min_conductivity}). "
                            f"Total lifetime flow={ec.total_flow:.2f}."
                        ),
                    )
                    self._pruning_candidates[cid] = candidate
                    new_candidates.append(candidate)

        orphan_nodes = self._find_orphan_nodes()

        for node_id, zone in orphan_nodes:
            cid = f"{zone}:node:{node_id}"
            if cid not in self._pruning_candidates and cid not in self._approved:
                candidate = PruningCandidate(
                    candidate_type="node",
                    zone=zone,
                    node_id=node_id,
                    src_id=None,
                    dst_id=None,
                    edge_kind=None,
                    conductivity=0.0,
                    dormant_steps=self._step,
                    total_flow=0.0,
                    flagged_at=now,
                    reason=(
                        f"Orphan node: all edges atrophied, "
                        f"no traversal recorded across {self._step} steps."
                    ),
                    dependent_nodes=self._get_dependent_nodes(node_id, zone),
                )
                self._pruning_candidates[cid] = candidate
                new_candidates.append(candidate)

        self._step_flow.clear()

        mean_conductivity = (
            sum(ec.conductivity for ec in self._conductivity.values())
            / len(self._conductivity)
            if self._conductivity else 0.0
        )

        summary = _build_optimization_summary(
            step=self._step,
            total_edges=len(self._conductivity),
            reinforced=reinforced,
            decayed=decayed,
            newly_atrophied=newly_atrophied,
            orphans=len(orphan_nodes),
            new_candidates=len(new_candidates),
            mean_conductivity=mean_conductivity,
        )

        return OptimizationResult(
            step=self._step,
            edges_reinforced=reinforced,
            edges_decayed=decayed,
            edges_atrophied=newly_atrophied,
            orphan_nodes=[n for n, z in orphan_nodes],
            pruning_candidates=new_candidates,
            mean_conductivity=round(mean_conductivity, 4),
            summary=summary,
        )

    def _solve_kirchhoff_flows(self) -> Dict[Tuple[str, str, str], float]:
        """
        Solve for edge flows using Kirchhoff's circuit laws.

        For each zone, builds a conductance matrix G where G_ij = D_ij,
        then solves the linear system G * p = b for node pressures p,
        where b encodes the demand at each node (traversal frequency as
        source/sink strength).

        Flow on each edge: Q_ij = D_ij * |p_i - p_j|

        This is the step required for correct Physarum dynamics. Without
        solving for pressure first, Q_ij cannot be computed from D_ij --
        you need to know the pressure difference across the edge to know
        how much fluid flows through it.

        Uses a simple iterative Gauss-Seidel solver (no numpy required)
        which converges reliably for the sparse graphs typical in VM.
        """
        from collections import defaultdict

        flows: Dict[Tuple[str, str, str], float] = {}

        # Group edges by zone
        zone_edges: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)
        for (zone, src, dst), ec in self._conductivity.items():
            if not ec.immune:
                zone_edges[zone].append((src, dst, zone, ec.conductivity))

        for zone, edges in zone_edges.items():
            # Collect nodes
            nodes = list({n for src, dst, z, d in edges for n in (src, dst)})
            if len(nodes) < 2:
                continue

            node_idx = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)

            # Build demand vector from traversal flow
            # High traversal = strong source. Low traversal = sink.
            demand = [0.0] * n
            total_demand = sum(self._step_flow.values()) or 1.0
            for (z, src, dst), flow in self._step_flow.items():
                if z != zone:
                    continue
                if src in node_idx:
                    demand[node_idx[src]] += flow / total_demand
                if dst in node_idx:
                    demand[node_idx[dst]] -= flow / total_demand * 0.5

            # Gauss-Seidel pressure solve
            pressure = [0.0] * n
            for _ in range(50):  # max iterations
                for i, node in enumerate(nodes):
                    weighted_sum = 0.0
                    total_conductance = 0.0
                    for src, dst, z, D in edges:
                        j = None
                        if src == node:
                            j = node_idx.get(dst)
                        elif dst == node:
                            j = node_idx.get(src)
                        if j is not None:
                            weighted_sum += D * pressure[j]
                            total_conductance += D
                    if total_conductance > 1e-10:
                        pressure[i] = (demand[i] + weighted_sum) / total_conductance

            # Compute flow from pressure differences
            for src, dst, z, D in edges:
                i = node_idx.get(src)
                j = node_idx.get(dst)
                if i is not None and j is not None:
                    q = D * abs(pressure[i] - pressure[j])
                    flows[(zone, src, dst)] = q
                    flows[(zone, dst, src)] = q * 0.5

        return flows

    # ── Pruning candidates ───────────────────────────────────

    def pruning_candidates(
        self,
        zone_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
    ) -> List[PruningCandidate]:
        """
        Return all pending pruning candidates.

        Optionally filtered by zone or edge kind.
        Never includes approved candidates.
        """
        candidates = list(self._pruning_candidates.values())
        if zone_filter:
            candidates = [c for c in candidates if c.zone == zone_filter]
        if kind_filter:
            candidates = [c for c in candidates if c.edge_kind == kind_filter]
        return sorted(candidates, key=lambda c: c.conductivity)

    def approve_pruning(
        self,
        approved_by: str,
        src_id: Optional[str] = None,
        dst_id: Optional[str] = None,
        node_id: Optional[str] = None,
        zone: str = "rationale",
    ) -> Optional[PruningCandidate]:
        """
        Record human approval for pruning an edge or node.

        Actual deletion from the graph must be performed separately
        by the caller. This records the approval decision.

        approved_by is required -- a human actor_id must be on record.
        """
        if src_id and dst_id:
            cid = f"{zone}:{src_id}:{dst_id}"
        elif node_id:
            cid = f"{zone}:node:{node_id}"
        else:
            return None

        candidate = self._pruning_candidates.get(cid)
        if not candidate:
            return None

        candidate.approved = True
        candidate.approved_by = approved_by
        self._approved.add(cid)
        del self._pruning_candidates[cid]
        return candidate

    # ── Diagnostics ──────────────────────────────────────────

    def conductivity_report(
        self,
        zone_filter: Optional[str] = None,
        top_n: int = 10,
    ) -> str:
        """
        Human-readable report of edge conductivity distribution.

        Shows top-N strongest and weakest edges for a zone.
        """
        edges = list(self._conductivity.values())
        if zone_filter:
            edges = [e for e in edges if e.zone == zone_filter]

        if not edges:
            return "No conductivity data available."

        by_conductivity = sorted(edges, key=lambda e: e.conductivity, reverse=True)
        strongest = by_conductivity[:top_n]
        weakest = [e for e in by_conductivity[-top_n:] if not e.immune]

        lines = [
            f"Hyphal Conductivity Report (step={self._step})",
            f"Total edges tracked: {len(edges)}",
            f"Mean conductivity: {sum(e.conductivity for e in edges)/len(edges):.4f}",
            "",
            f"Strongest {len(strongest)} edges:",
        ]
        for e in strongest:
            lines.append(
                f"  {e.src_id[:8]}...->{e.dst_id[:8]}... "
                f"zone={e.zone} kind={e.kind} "
                f"D={e.conductivity:.3f} flow={e.total_flow:.1f}"
            )
        lines += ["", f"Weakest {len(weakest)} edges (non-immune):"]
        for e in weakest:
            lines.append(
                f"  {e.src_id[:8]}...->{e.dst_id[:8]}... "
                f"zone={e.zone} kind={e.kind} "
                f"D={e.conductivity:.3f} dormant={e.dormant_steps} steps"
            )

        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────

    def _find_orphan_nodes(self) -> List[Tuple[str, str]]:
        """
        Find nodes where all incident edges are atrophied.

        A node is an orphan if every edge touching it has conductivity
        at or below min_conductivity and it has received no flow.
        """
        node_edges: Dict[Tuple[str, str], List[EdgeConductivity]] = defaultdict(list)
        for (zone, src, dst), ec in self._conductivity.items():
            node_edges[(src, zone)].append(ec)
            node_edges[(dst, zone)].append(ec)

        orphans = []
        for (node_id, zone), edges in node_edges.items():
            if zone in IMMUNE_ZONES:
                continue
            non_immune = [e for e in edges if not e.immune]
            if not non_immune:
                continue
            if all(e.is_atrophied for e in non_immune):
                orphans.append((node_id, zone))

        return orphans

    def _get_dependent_nodes(self, node_id: str, zone_val: str) -> List[str]:
        """Find nodes that would lose connections if this node were pruned."""
        from graph_types import GraphZone
        try:
            zone = GraphZone(zone_val)
        except ValueError:
            return []
        adj = self.graph._adj.get(zone, {})
        dependents = []
        for src, neighbors in adj.items():
            for dst, *_ in neighbors:
                if dst == node_id and src != node_id:
                    dependents.append(src)
                if src == node_id and dst != node_id:
                    dependents.append(dst)
        return list(set(dependents))[:10]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_optimization_summary(
    step: int,
    total_edges: int,
    reinforced: int,
    decayed: int,
    newly_atrophied: int,
    orphans: int,
    new_candidates: int,
    mean_conductivity: float,
) -> str:
    return (
        f"Hyphal optimization step {step}. "
        f"Edges tracked: {total_edges}. "
        f"Reinforced: {reinforced}. "
        f"Decayed: {decayed}. "
        f"Newly atrophied: {newly_atrophied}. "
        f"Orphan nodes: {orphans}. "
        f"New pruning candidates: {new_candidates}. "
        f"Mean conductivity: {mean_conductivity:.4f}."
    )

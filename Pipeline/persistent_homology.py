"""
VeritasMemoria - Persistent Homology Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects reasoning gaps and structural weaknesses in the belief graph
using persistent homology from topological data analysis (TDA).

What persistent homology adds that spectral analysis cannot
------------------------------------------------------------
SALCoherenceLayer uses the Fiedler value (lambda2) to measure whether
the belief graph is connected enough to commit. Lambda2 tells you about
global connectivity at the current edge weight threshold. It does not
tell you:

    - Which specific connections are structurally weak vs robust
    - Whether a connection persists across a range of thresholds or
      dies the moment you raise the bar slightly
    - Where the reasoning chain has a genuine gap vs a thin bridge

Persistent homology answers these questions by analyzing the graph
across ALL possible edge weight thresholds simultaneously and tracking
which topological features (connected components, loops, voids) are
born and which die as the threshold increases.

A feature with high persistence (born early, dies late) is real
structure. A feature with low persistence (born and dies at nearly
the same threshold) is noise or a weak link.

For VM's belief graph:
    Long-persistent component  a robust, well-supported belief cluster
    Short-persistent component a fragile belief that barely holds together
    Persistence gap            reasoning chain relies on a single weak edge
    Topological hole (H1)      circular reasoning or belief loop with no
                               independent resolution path

The mathematical framework
----------------------------
We compute Vietoris-Rips filtration over the belief graph:

1. Start with threshold t = 0 (no edges, all nodes isolated)
2. Increase t from 0 to 1 (max edge weight)
3. At each t, add all edges with weight >= t to the graph
4. Track when connected components merge (H0 -- dimension 0 homology)
5. Track when loops form and fill (H1 -- dimension 1 homology)

Each topological event is recorded as a (birth, death) pair called a
persistence pair. The persistence = death - birth. High persistence = 
robust feature.

The persistence diagram plots all (birth, death) pairs. Points far
from the diagonal (death >> birth) are signal. Points near the
diagonal are noise.

Implementation note
--------------------
True Vietoris-Rips filtration over arbitrary point clouds requires
computing pairwise distances in high-dimensional space. For VM's
graph, we have a simpler structure: we already have edge weights
that encode relationship strength. We use these directly as the
filtration parameter rather than geometric distances.

This is a graph filtration (also called a weighted graph filtration
or a Rips filtration on the adjacency matrix). It is a standard
approach in TDA applied to network data and is mathematically sound.

H0 (connected components) is computed exactly using union-find.
H1 (loops) is approximated using cycle detection in the filtration
-- exact H1 computation requires boundary matrix reduction which
is O(n^3) and impractical for real-time use. The approximation
identifies loops that form and tracks when they become triangulated
(filled) by a third supporting edge.

Integration with SAL
---------------------
PersistentHomologyLayer attaches to the same GraphEngine as
SALCoherenceLayer. It should be called alongside coherence_state()
and its output enriches the commit decision:

    - High mean H0 persistence: good, belief clusters are robust
    - Many short-lived H0 pairs: fragile connectivity, raise concern
    - H1 features present: circular reasoning detected, investigate
    - Persistence gap (large jump in birth values): reasoning chain
      has a structural discontinuity

PersistenceState is added to CoherenceState as an optional field.
The Archivist checks it before commit alongside lambda2 and sheaf.

Dependencies
------------
numpy -- already in requirements for Laplacian computation
Standard library for union-find and filtration management
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Persistence below this is treated as noise (near-diagonal points)
DEFAULT_PERSISTENCE_NOISE_FLOOR: float = 0.05

# Number of filtration steps (higher = more precise, slower)
DEFAULT_FILTRATION_STEPS: int = 20

# H1 persistence threshold -- loops must persist this long to be flagged
DEFAULT_H1_THRESHOLD: float = 0.10


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class PersistencePair:
    """
    A (birth, death) pair from the persistence diagram.

    birth       threshold at which this feature appeared
    death       threshold at which this feature disappeared
                (math.inf means it persists to the end -- never dies)
    persistence death - birth (inf for essential features)
    dimension   0 = connected component, 1 = loop
    node_ids    nodes involved in this feature
    """
    birth: float
    death: float
    persistence: float
    dimension: int              # 0 = H0, 1 = H1
    node_ids: List[str]
    feature_type: str           # "component", "loop", "essential"


@dataclass
class PersistenceGap:
    """
    A structural discontinuity in the filtration.

    A large gap between consecutive birth values means that a significant
    portion of the belief graph requires a much weaker edge to connect --
    the reasoning chain has a thin bridge at that threshold.
    """
    gap_start: float        # threshold just before the gap
    gap_end: float          # threshold just after the gap
    gap_size: float         # gap_end - gap_start
    nodes_before: int       # nodes connected at gap_start
    nodes_after: int        # nodes connected at gap_end
    weak_edge: Optional[Tuple[str, str, float]]   # the edge that bridges the gap


@dataclass
class PersistenceState:
    """
    Full TDA output for a zone.

    h0_pairs            all H0 persistence pairs (component merges)
    h1_pairs            all H1 persistence pairs (loops detected)
    essential_count     H0 features that never die (connected components
                        at threshold 0 -- indicates disconnected zone)
    mean_h0_persistence mean persistence of H0 pairs (higher = more robust)
    max_h0_persistence  most persistent H0 pair (the strongest connection)
    persistence_gaps    structural discontinuities in the filtration
    circular_reasoning  True if significant H1 features detected
    betti_0             number of connected components at threshold 0
                        (1 = fully connected, n = n isolated nodes)
    betti_1             number of independent loops at threshold 0
    noise_pairs         H0 pairs below persistence_noise_floor
    signal_pairs        H0 pairs above persistence_noise_floor
    zone: str
    analyzed_at: str
    node_count: int
    edge_count: int
    summary: str
    """
    zone: str
    analyzed_at: str
    node_count: int
    edge_count: int
    h0_pairs: List[PersistencePair]
    h1_pairs: List[PersistencePair]
    essential_count: int
    mean_h0_persistence: float
    max_h0_persistence: float
    persistence_gaps: List[PersistenceGap]
    circular_reasoning: bool
    betti_0: int
    betti_1: int
    noise_pairs: int
    signal_pairs: int
    summary: str


# ─────────────────────────────────────────────────────────────
# Union-Find for H0 computation
# ─────────────────────────────────────────────────────────────

class UnionFind:
    """
    Path-compressed union-find for efficient component tracking
    during filtration.
    """

    def __init__(self, nodes: List[str]):
        self._parent: Dict[str, str] = {n: n for n in nodes}
        self._rank: Dict[str, int] = {n: 0 for n in nodes}
        self._component_nodes: Dict[str, Set[str]] = {n: {n} for n in nodes}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> Optional[Tuple[str, str]]:
        """
        Merge components containing x and y.
        Returns (root_absorbed, root_surviving) if merge happened,
        None if already in same component.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None

        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx

        self._parent[ry] = rx
        self._component_nodes[rx] |= self._component_nodes[ry]
        del self._component_nodes[ry]

        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

        return (ry, rx)  # ry was absorbed into rx

    def component_nodes(self, x: str) -> Set[str]:
        return self._component_nodes.get(self.find(x), {x})

    def num_components(self) -> int:
        return len(self._component_nodes)


# ─────────────────────────────────────────────────────────────
# Persistent homology layer
# ─────────────────────────────────────────────────────────────

class PersistentHomologyLayer:
    """
    Computes persistent homology over VM's weighted belief graph.

    Attaches to a GraphEngine instance. Read-only -- does not modify
    the graph or block any nodes.

    Usage
    -----
        tda = PersistentHomologyLayer(graph_engine)

        # Analyze a zone before commit:
        state = tda.analyze_zone(GraphZone.RATIONALE)
        print(state.summary)

        # Check for circular reasoning:
        if state.circular_reasoning:
            # Surface H1 pairs to human for review

        # Check structural robustness:
        if state.mean_h0_persistence < 0.2:
            # Belief clusters are fragile -- connections are weak

        # Check for reasoning gaps:
        for gap in state.persistence_gaps:
            if gap.gap_size > 0.3:
                # Significant structural discontinuity
    """

    def __init__(
        self,
        graph_engine,
        noise_floor: float = DEFAULT_PERSISTENCE_NOISE_FLOOR,
        filtration_steps: int = DEFAULT_FILTRATION_STEPS,
        h1_threshold: float = DEFAULT_H1_THRESHOLD,
    ):
        self.graph = graph_engine
        self.noise_floor = noise_floor
        self.filtration_steps = filtration_steps
        self.h1_threshold = h1_threshold

    # ── Public API ───────────────────────────────────────────

    def analyze_zone(self, zone) -> PersistenceState:
        """
        Run full persistent homology analysis for a zone.

        Computes H0 (component) and H1 (loop) persistence pairs,
        identifies structural gaps, and assesses circular reasoning risk.
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        now = _utc_now()

        # Build edge list from SAL-style weighted adjacency
        edges, nodes = self._extract_edges(zone_val)

        if len(nodes) < 3:
            return PersistenceState(
                zone=zone_val,
                analyzed_at=now,
                node_count=len(nodes),
                edge_count=len(edges),
                h0_pairs=[],
                h1_pairs=[],
                essential_count=len(nodes),
                mean_h0_persistence=0.0,
                max_h0_persistence=0.0,
                persistence_gaps=[],
                circular_reasoning=False,
                betti_0=len(nodes),
                betti_1=0,
                noise_pairs=0,
                signal_pairs=0,
                summary=(
                    f"Zone '{zone_val}' has only {len(nodes)} nodes -- "
                    "insufficient for persistent homology analysis."
                ),
            )

        # Sort edges by weight descending for filtration
        # (we add strongest edges first -- high weight = born early)
        edges_sorted = sorted(edges, key=lambda e: e[2], reverse=True)

        # Compute filtration thresholds
        weights = [e[2] for e in edges_sorted]
        thresholds = self._build_thresholds(weights)

        # H0: compute persistence pairs via union-find over filtration
        h0_pairs, essential_count, birth_events = self._compute_h0(
            nodes=list(nodes),
            edges=edges_sorted,
            thresholds=thresholds,
        )

        # H1: detect loops in the filtration (approximate)
        h1_pairs = self._compute_h1_approx(
            nodes=list(nodes),
            edges=edges_sorted,
            thresholds=thresholds,
        )

        # Persistence gaps
        gaps = self._find_persistence_gaps(
            birth_events=birth_events,
            edges=edges_sorted,
            nodes=list(nodes),
        )

        # Metrics
        finite_h0 = [p for p in h0_pairs if p.persistence != math.inf]
        mean_h0 = (
            sum(p.persistence for p in finite_h0) / len(finite_h0)
            if finite_h0 else 0.0
        )
        max_h0 = max((p.persistence for p in finite_h0), default=0.0)

        noise_pairs = sum(
            1 for p in h0_pairs
            if p.persistence != math.inf and p.persistence < self.noise_floor
        )
        signal_pairs = sum(
            1 for p in h0_pairs
            if p.persistence != math.inf and p.persistence >= self.noise_floor
        )

        circular = any(p.persistence >= self.h1_threshold for p in h1_pairs)

        # Betti numbers at threshold 0 (full graph)
        betti_0 = essential_count
        betti_1 = len([p for p in h1_pairs if p.persistence >= self.h1_threshold])

        summary = _build_tda_summary(
            zone=zone_val,
            node_count=len(nodes),
            edge_count=len(edges),
            betti_0=betti_0,
            betti_1=betti_1,
            mean_h0=mean_h0,
            max_h0=max_h0,
            signal_pairs=signal_pairs,
            noise_pairs=noise_pairs,
            gap_count=len(gaps),
            circular=circular,
        )

        return PersistenceState(
            zone=zone_val,
            analyzed_at=now,
            node_count=len(nodes),
            edge_count=len(edges),
            h0_pairs=h0_pairs,
            h1_pairs=h1_pairs,
            essential_count=essential_count,
            mean_h0_persistence=round(mean_h0, 4),
            max_h0_persistence=round(max_h0, 4),
            persistence_gaps=gaps,
            circular_reasoning=circular,
            betti_0=betti_0,
            betti_1=betti_1,
            noise_pairs=noise_pairs,
            signal_pairs=signal_pairs,
            summary=summary,
        )

    # ── H0 computation ───────────────────────────────────────

    def _compute_h0(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        thresholds: List[float],
    ) -> Tuple[List[PersistencePair], int, List[Tuple[float, str]]]:
        """
        Compute H0 persistence pairs via union-find filtration.

        As we add edges in decreasing weight order, each merge event
        records the death of a component that gets absorbed. The born
        threshold is the weight at which a component first appears
        (all nodes are born at threshold = max_weight since at that
        threshold we have only the strongest edges).

        Returns (h0_pairs, essential_count, birth_events).
        birth_events is a list of (threshold, node_id) for gap detection.
        """
        uf = UnionFind(nodes)
        pairs: List[PersistencePair] = []
        birth_events: List[Tuple[float, str]] = []

        # All nodes start as isolated components (born at weight 1.0)
        node_birth: Dict[str, float] = {n: 1.0 for n in nodes}

        for src, dst, weight in edges:
            if src not in node_birth or dst not in node_birth:
                continue

            result = uf.union(src, dst)
            if result is None:
                continue  # already connected

            absorbed_root, surviving_root = result

            # The absorbed component dies at this weight
            # It was born when the younger of the two roots was born
            absorbed_birth = node_birth[absorbed_root]
            surviving_birth = node_birth[surviving_root]

            # Older birth = was born earlier (higher threshold) = stronger
            # By convention, the younger component (lower birth threshold) dies
            dying_birth = min(absorbed_birth, surviving_birth)
            death = weight
            persistence = dying_birth - death

            dying_nodes = list(uf.component_nodes(absorbed_root))

            pairs.append(PersistencePair(
                birth=round(dying_birth, 4),
                death=round(death, 4),
                persistence=round(persistence, 4),
                dimension=0,
                node_ids=dying_nodes[:5],   # sample for readability
                feature_type="component",
            ))
            birth_events.append((dying_birth, absorbed_root))

            # Surviving root inherits the older birth
            node_birth[surviving_root] = max(absorbed_birth, surviving_birth)

        # Essential features: components that never merge (betti_0)
        essential_count = uf.num_components()

        # Add essential pairs (birth, inf)
        for root, component in uf._component_nodes.items():
            pairs.append(PersistencePair(
                birth=round(node_birth.get(root, 1.0), 4),
                death=math.inf,
                persistence=math.inf,
                dimension=0,
                node_ids=list(component)[:5],
                feature_type="essential",
            ))

        return pairs, essential_count, birth_events

    # ── H1 computation (approximate) ─────────────────────────

    def _compute_h1_approx(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        thresholds: List[float],
    ) -> List[PersistencePair]:
        """
        Approximate H1 (loop) detection via cycle tracking.

        A loop is born when an edge is added between two nodes that are
        already in the same connected component (creating a cycle). It
        dies when a third edge triangulates the loop (creating a 2-simplex
        that fills the hole).

        This is an approximation -- exact H1 requires boundary matrix
        reduction. For VM's use case (detecting circular reasoning),
        the approximation correctly identifies the presence and rough
        persistence of loops without the O(n^3) cost.
        """
        uf_h1 = UnionFind(nodes)
        loops: Dict[str, Tuple[float, List[str]]] = {}  # loop_id -> (birth, nodes)
        pairs: List[PersistencePair] = []

        # Track adjacency for triangulation detection
        adjacency: Dict[str, Set[str]] = defaultdict(set)

        for src, dst, weight in edges:
            if src not in {n for n in nodes} or dst not in {n for n in nodes}:
                continue

            src_root = uf_h1.find(src)
            dst_root = uf_h1.find(dst)

            if src_root == dst_root:
                # Adding this edge creates a cycle -- H1 feature born
                loop_id = f"{min(src,dst)}-{max(src,dst)}"
                if loop_id not in loops:
                    loops[loop_id] = (weight, [src, dst])
            else:
                # Check if this edge triangulates an existing loop
                common_neighbors = adjacency[src] & adjacency[dst]
                if common_neighbors:
                    # This edge completes a triangle -- fills a loop
                    for loop_id, (birth, loop_nodes) in list(loops.items()):
                        # If either endpoint was in a loop, that loop may be filled
                        if src in loop_nodes or dst in loop_nodes:
                            death = weight
                            persistence = birth - death
                            if persistence > 0:
                                pairs.append(PersistencePair(
                                    birth=round(birth, 4),
                                    death=round(death, 4),
                                    persistence=round(persistence, 4),
                                    dimension=1,
                                    node_ids=loop_nodes,
                                    feature_type="loop",
                                ))
                            del loops[loop_id]
                            break

                uf_h1.union(src, dst)

            adjacency[src].add(dst)
            adjacency[dst].add(src)

        # Remaining loops never died -- essential H1
        for loop_id, (birth, loop_nodes) in loops.items():
            pairs.append(PersistencePair(
                birth=round(birth, 4),
                death=math.inf,
                persistence=math.inf,
                dimension=1,
                node_ids=loop_nodes,
                feature_type="essential",
            ))

        return pairs

    # ── Persistence gap detection ─────────────────────────────

    def _find_persistence_gaps(
        self,
        birth_events: List[Tuple[float, str]],
        edges: List[Tuple[str, str, float]],
        nodes: List[str],
    ) -> List[PersistenceGap]:
        """
        Find large gaps between consecutive birth thresholds.

        A gap means that between threshold t1 and t2, no new components
        merge -- the graph is in a stable state and then suddenly many
        connections appear. Large gaps indicate structural discontinuity:
        the reasoning chain relies on edges clustered at a narrow weight
        range and would fragment if those edges were removed.
        """
        if len(birth_events) < 2:
            return []

        sorted_births = sorted(birth_events, key=lambda x: x[0], reverse=True)
        gaps: List[PersistenceGap] = []

        edge_by_weight = {(s, d): w for s, d, w in edges}

        for i in range(len(sorted_births) - 1):
            t1, node1 = sorted_births[i]
            t2, node2 = sorted_births[i + 1]
            gap_size = t1 - t2

            if gap_size < 0.15:
                continue

            # Find the edge that bridges this gap
            # (the weakest edge that connects across the gap)
            bridge_edge = None
            for src, dst, w in edges:
                if t2 <= w <= t1:
                    if bridge_edge is None or w < bridge_edge[2]:
                        bridge_edge = (src, dst, w)

            gaps.append(PersistenceGap(
                gap_start=round(t2, 4),
                gap_end=round(t1, 4),
                gap_size=round(gap_size, 4),
                nodes_before=i + 1,
                nodes_after=i + 2,
                weak_edge=bridge_edge,
            ))

        return sorted(gaps, key=lambda g: g.gap_size, reverse=True)

    # ── Helpers ──────────────────────────────────────────────

    def _extract_edges(
        self,
        zone_val: str,
    ) -> Tuple[List[Tuple[str, str, float]], Set[str]]:
        """
        Extract positive-weight edges from the zone adjacency.

        Uses absolute values -- both supporting and contradicting edges
        contribute to topological structure. Blocked edges excluded.
        """
        from veritas_memoria.core.graph.graph import GraphZone

        try:
            zone = GraphZone(zone_val)
        except ValueError:
            return [], set()

        adj = self.graph._adj.get(zone, {})
        blocked = set(self.graph._blocked_nodes.keys())

        edges: List[Tuple[str, str, float]] = []
        nodes: Set[str] = set()
        seen: Set[frozenset] = set()

        for src_id, neighbors in adj.items():
            if src_id in blocked:
                continue
            nodes.add(src_id)
            for dst_id, kind, weight, gate in neighbors:
                if dst_id in blocked:
                    continue
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if gate_val == "block_until_resolved":
                    continue

                pair = frozenset({src_id, dst_id})
                if pair in seen:
                    continue
                seen.add(pair)

                nodes.add(dst_id)
                # Use absolute weight -- topology doesn't care about sign,
                # only about whether a connection exists and how strong it is
                edges.append((src_id, dst_id, abs(float(weight))))

        return edges, nodes

    def _build_thresholds(self, weights: List[float]) -> List[float]:
        """Build evenly-spaced filtration thresholds across weight range."""
        if not weights:
            return []
        min_w = min(weights)
        max_w = max(weights)
        if min_w == max_w:
            return [max_w]
        step = (max_w - min_w) / self.filtration_steps
        return [max_w - i * step for i in range(self.filtration_steps + 1)]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_tda_summary(
    zone: str,
    node_count: int,
    edge_count: int,
    betti_0: int,
    betti_1: int,
    mean_h0: float,
    max_h0: float,
    signal_pairs: int,
    noise_pairs: int,
    gap_count: int,
    circular: bool,
) -> str:
    robustness = (
        "ROBUST" if mean_h0 > 0.3
        else "MODERATE" if mean_h0 > 0.15
        else "FRAGILE"
    )
    circular_note = " CIRCULAR REASONING DETECTED." if circular else ""
    gap_note = f" {gap_count} structural gap(s) detected." if gap_count else ""

    return (
        f"Persistent Homology -- zone '{zone}'\n"
        f"  Nodes: {node_count}  Edges: {edge_count}\n"
        f"  Connected components (beta_0): {betti_0} "
        f"({'fully connected' if betti_0 == 1 else 'FRAGMENTED'})\n"
        f"  Independent loops (beta_1): {betti_1}{circular_note}\n"
        f"  Mean H0 persistence: {mean_h0:.4f} [{robustness}]\n"
        f"  Max H0 persistence:  {max_h0:.4f}\n"
        f"  Signal pairs: {signal_pairs}  Noise pairs: {noise_pairs}"
        f"{gap_note}"
    )

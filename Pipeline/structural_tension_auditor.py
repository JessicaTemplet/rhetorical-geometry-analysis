"""
VeritasMemoria - Structural Tension Auditor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scans belief graph zones for nodes that are implicitly close to high-mass
attractor nodes (GOVERNANCE / IDENTITY) without a declared bridge edge.
These are structural tension points — places where someone probably should
have declared a cross-zone relationship but didn't.

Use this as a periodic audit probe, not a retrieval filter. The output
is a list of nodes worth inspecting and potentially formalizing via
explicit bridge edges.

Why this is a probe not an enforcer
-------------------------------------
VeritasMemoria's explicit design rule is that cross-zone relationships
require a declared bridge edge. The epistatic gate handles declared
suppressions. This module surfaces undeclared implicit proximity — it
does not apply constraints from those relationships. It reports them.

Physics
--------
Gravitational influence decays with hop distance:

    influence(d) = M / (1 + α·d²)

Node mass is zone-derived and amplified by connectivity:

    mass = zone_base_mass × (1 + log(1 + weighted_degree))

Tidal tension measures how hard undeclared attractors are pulling
relative to the node's own grounding:

    tidal_tension = total_attractor_influence / (1 + local_connectivity)

A node with high tidal tension is being dominated by nearby massive
nodes it has no declared relationship with. Flag it for review.

Zone base masses
-----------------
    GOVERNANCE  1.00
    IDENTITY    0.75
    KNOWLEDGE   0.45
    WORK        0.25
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Zone mass table
# ─────────────────────────────────────────────────────────────

# Zone base mass mirrors HSH Lambda values — higher Lambda = more gravitational
# authority = higher mass = stronger structural tension propagation.
ZONE_BASE_MASS: Dict[str, float] = {
    "governance":         1.00,   # lambda=1.0 — anchor of the hyperbolic disk
    "rationale":   0.65,   # lambda=0.65 — stable professional corpus
    "work_knowledge":     0.55,   # lambda=0.55 — matter-scoped persistent facts
    "temporal_knowledge": 0.35,   # lambda=0.35 — ephemeral, near the disk boundary
}

# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────

DEFAULT_ALPHA: float           = 1.0
DEFAULT_MIN_INFLUENCE: float   = 0.05
DEFAULT_MAX_HOPS: int          = 6
DEFAULT_CRISIS_THRESHOLD: float = 0.65
DEFAULT_MIN_MASS: float        = 0.10


# ─────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────

@dataclass
class CrisisRecord:
    """
    A node under significant undeclared gravitational tension.

    node_id                   The node being pulled.
    zone                      Its zone.
    tidal_tension             total_attractor_influence / (1 + local_connectivity).
    local_connectivity        Sum of absolute edge weights in the node's own zone.
    total_attractor_influence Combined pull from all in-range attractors.
    dominant_attractor_id     Source ID of the strongest attractor.
    dominant_attractor_zone   Zone of the dominant attractor.
    dominant_hop_distance     Hop distance to the dominant attractor.
    dominant_influence        Influence from the dominant attractor alone.
    attractor_count           Total attractors within range.
    warning                   Human-readable note for the auditor.
    """
    node_id:                   str
    zone:                      str
    tidal_tension:             float
    local_connectivity:        float
    total_attractor_influence: float
    dominant_attractor_id:     Optional[str]
    dominant_attractor_zone:   Optional[str]
    dominant_hop_distance:     int
    dominant_influence:        float
    attractor_count:           int
    warning:                   str


# ─────────────────────────────────────────────────────────────
# Structural Tension Auditor
# ─────────────────────────────────────────────────────────────

class StructuralTensionAuditor:
    """
    Periodic audit probe for implicit gravitational tension.

    Finds nodes that are topologically close to high-mass GOVERNANCE or
    IDENTITY nodes without a declared bridge relationship. These are
    candidates for explicit bridge edge formalization.

    Read-only. Never modifies graph state.

    Parameters
    ----------
    graph_engine : GraphEngine
    alpha : float
        Decay constant for influence falloff. Default 1.0.
    min_influence : float
        Floor below which attractor influence is ignored. Default 0.05.
    max_hops : int
        BFS depth limit. Default 6.
    crisis_threshold : float
        Tidal tension above which a node is flagged. Default 0.65.
    min_mass : float
        Minimum mass for a node to be treated as an attractor. Default 0.10.
    """

    def __init__(
        self,
        graph_engine,
        alpha: float            = DEFAULT_ALPHA,
        min_influence: float    = DEFAULT_MIN_INFLUENCE,
        max_hops: int           = DEFAULT_MAX_HOPS,
        crisis_threshold: float = DEFAULT_CRISIS_THRESHOLD,
        min_mass: float         = DEFAULT_MIN_MASS,
    ):
        self.graph            = graph_engine
        self.alpha            = alpha
        self.min_influence    = min_influence
        self.max_hops         = max_hops
        self.crisis_threshold = crisis_threshold
        self.min_mass         = min_mass

    # ── Public API ────────────────────────────────────────────

    def scan_for_crises(self, zone) -> List[CrisisRecord]:
        """
        Scan all nodes in a zone for high tidal tension.

        Returns CrisisRecord for every node whose tidal_tension exceeds
        crisis_threshold, sorted by tidal_tension descending (worst first).

        Use the results to identify cross-zone relationships worth
        formalizing as explicit bridge edges.

        Parameters
        ----------
        zone : GraphZone

        Returns
        -------
        list of CrisisRecord, sorted by tidal_tension descending.
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        from graph_types import GraphZone as GZ
        try:
            gz = GZ(zone_val)
        except ValueError:
            return []

        all_nodes = set(self.graph._adj.get(gz, {}).keys())
        blocked   = self.graph._blocked_nodes
        crises: List[CrisisRecord] = []

        for node_id in all_nodes:
            if node_id in blocked:
                continue
            record = self._evaluate_node(node_id, gz, zone_val)
            if record is not None:
                crises.append(record)

        crises.sort(key=lambda r: r.tidal_tension, reverse=True)
        logger.info(
            "StructuralTensionAuditor.scan_for_crises('%s'): %d crisis nodes.",
            zone_val, len(crises),
        )
        return crises

    def node_tension(self, node_id: str, zone) -> float:
        """
        Compute tidal tension for a single node.

        Convenience method for spot-checks without a full zone scan.

        Returns
        -------
        float
            Tidal tension in [0, ∞). 0.0 = no attractors nearby.
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        from graph_types import GraphZone as GZ
        try:
            gz = GZ(zone_val)
        except ValueError:
            return 0.0
        local = self._local_connectivity(node_id, gz)
        total_inf, _, _, _ = self._find_attractors(node_id, gz)
        return total_inf / (1.0 + local)

    # ── Internal: per-node evaluation ────────────────────────

    def _evaluate_node(self, node_id: str, gz, zone_val: str) -> Optional[CrisisRecord]:
        """Return a CrisisRecord if the node's tension exceeds the threshold, else None."""
        local = self._local_connectivity(node_id, gz)
        total_inf, dominant_id, dominant_zone, dominant_hop, dominant_inf, count = \
            self._find_attractors_full(node_id, gz)

        tension = total_inf / (1.0 + local)
        if tension < self.crisis_threshold:
            return None

        if dominant_id:
            warning = (
                f"Node '{node_id}' has tidal tension {tension:.3f} "
                f"(threshold {self.crisis_threshold}). "
                f"Dominant undeclared attractor: '{dominant_id}' "
                f"({dominant_zone}, d={dominant_hop}, "
                f"influence={dominant_inf:.3f}). "
                f"Consider formalizing this relationship as a bridge edge."
            )
        else:
            warning = (
                f"Node '{node_id}' has tidal tension {tension:.3f} "
                f"from {count} attractor(s). Review bridge edge coverage."
            )

        return CrisisRecord(
            node_id=node_id,
            zone=zone_val,
            tidal_tension=round(tension, 4),
            local_connectivity=round(local, 4),
            total_attractor_influence=round(total_inf, 4),
            dominant_attractor_id=dominant_id,
            dominant_attractor_zone=dominant_zone,
            dominant_hop_distance=dominant_hop,
            dominant_influence=round(dominant_inf, 4),
            attractor_count=count,
            warning=warning,
        )

    def _find_attractors(self, node_id: str, gz) -> Tuple[float, Optional[str], Optional[str], int]:
        """Return (total_influence, dominant_id, dominant_zone, count) — compact form."""
        total, dom_id, dom_zone, dom_hop, dom_inf, count = \
            self._find_attractors_full(node_id, gz)
        return total, dom_id, dom_zone, count

    def _find_attractors_full(
        self, node_id: str, gz
    ) -> Tuple[float, Optional[str], Optional[str], int, float, int]:
        """
        BFS from node_id, collect all attractor nodes within range.

        Returns (total_influence, dominant_id, dominant_zone_val,
                 dominant_hop, dominant_influence, attractor_count).
        """
        reachable = self._bfs_hop_distances(node_id, gz)
        blocked   = self.graph._blocked_nodes

        total_inf    = 0.0
        dominant_id:   Optional[str] = None
        dominant_zone: Optional[str] = None
        dominant_hop:  int            = 0
        dominant_inf:  float          = 0.0
        count = 0

        for (reached_id, reached_zone_val), hop_dist in reachable.items():
            if reached_id == node_id:
                continue
            if reached_id in blocked:
                continue
            # Only count nodes that are more massive zones than the target
            mass = self._node_mass(reached_id, reached_zone_val)
            if mass < self.min_mass:
                continue
            inf = self._influence(mass, hop_dist)
            if inf < self.min_influence:
                continue

            total_inf += inf
            count += 1
            if inf > dominant_inf:
                dominant_inf  = inf
                dominant_id   = reached_id
                dominant_zone = reached_zone_val
                dominant_hop  = hop_dist

        return total_inf, dominant_id, dominant_zone, dominant_hop, dominant_inf, count

    # ── Internal: physics ─────────────────────────────────────

    def _influence(self, mass: float, hop_distance: int) -> float:
        return mass / (1.0 + self.alpha * hop_distance ** 2)

    def _node_mass(self, node_id: str, zone_val: str) -> float:
        base = ZONE_BASE_MASS.get(zone_val, 0.25)
        try:
            from graph_types import GraphZone as GZ
            gz = GZ(zone_val)
            neighbors = self.graph._adj.get(gz, {}).get(node_id, [])
            weighted_degree = sum(abs(float(w)) for (_, _, w, _) in neighbors)
        except Exception:
            weighted_degree = 0.0
        return base * (1.0 + math.log1p(weighted_degree))

    def _local_connectivity(self, node_id: str, gz) -> float:
        try:
            neighbors = self.graph._adj.get(gz, {}).get(node_id, [])
            return sum(abs(float(w)) for (_, _, w, _) in neighbors)
        except Exception:
            return 0.0

    # ── Internal: BFS ─────────────────────────────────────────

    def _bfs_hop_distances(
        self, start_id: str, start_gz
    ) -> Dict[Tuple[str, str], int]:
        """
        Multi-zone BFS. Traverses _adj (within-zone) and _bridges (cross-zone).
        Skips BLOCK_UNTIL_RESOLVED edges and blocked nodes.

        Returns dict of (node_id, zone_val) -> hop_distance.
        """
        start_zone_val = (
            start_gz.value if hasattr(start_gz, "value") else str(start_gz)
        )
        from graph_types import GraphZone as GZ

        blocked  = self.graph._blocked_nodes
        visited: Dict[Tuple[str, str], int] = {}
        queue:   deque = deque()

        start_key = (start_id, start_zone_val)
        visited[start_key] = 0
        queue.append((start_id, start_zone_val, 0))

        while queue:
            cur_id, cur_zone_val, depth = queue.popleft()
            if depth >= self.max_hops:
                continue

            # Within-zone edges
            try:
                gz = GZ(cur_zone_val)
                for (dst_id, _, _w, gate) in self.graph._adj.get(gz, {}).get(cur_id, []):
                    if dst_id in blocked:
                        continue
                    g = gate.value if hasattr(gate, "value") else str(gate)
                    if g == "block_until_resolved":
                        continue
                    key = (dst_id, cur_zone_val)
                    if key not in visited:
                        visited[key] = depth + 1
                        queue.append((dst_id, cur_zone_val, depth + 1))
            except ValueError:
                pass

            # Bridge edges
            try:
                gz = GZ(cur_zone_val)
                for (to_zone, to_id, _w, gate) in self.graph._bridges.get((gz, cur_id), []):
                    if to_id in blocked:
                        continue
                    g = gate.value if hasattr(gate, "value") else str(gate)
                    if g == "block_until_resolved":
                        continue
                    to_zone_val = to_zone.value if hasattr(to_zone, "value") else str(to_zone)
                    key = (to_id, to_zone_val)
                    if key not in visited:
                        visited[key] = depth + 1
                        queue.append((to_id, to_zone_val, depth + 1))
            except Exception:
                pass

        return visited

"""
VeritasMemoria - Veritas Geometria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Self-organizing conceptual galaxy consolidation layer for VeritasMemoria.

Background
-----------
This module emerged from a simulation that tested whether a Riemannian
Belief Manifold could produce self-organizing "conceptual galaxies" --
coherent clusters of semantically related nodes that tighten over time
through geometry-driven rewiring, without central coordination.

The simulation confirmed three behaviors that are directly useful for
AI memory:

  1. Semantic tension tends to correlate with negative Ollivier-Ricci
     curvature -- diverging neighborhoods are a signal worth surfacing,
     though negative curvature has multiple causes beyond contradiction.

  2. Coherent regions generate positive curvature. Clusters tighten
     and become more reliable retrieval targets over time.

  3. Edge creation driven by curvature produces galaxy formation from
     both structured (community) and unstructured (random) initial
     topologies. Structure emerges from geometry alone.

What this module adds to VM
-----------------------------
VM already has excellent contradiction detection, epistatic silencing,
and spectral coherence measurement. What it does not have is a geometric
consolidation process that runs between sessions to reorganize the belief
graph toward coherence -- the memory equivalent of sleep consolidation.

Veritas Geometria runs as a background pass. It:

  - Computes Ollivier-Ricci curvature for edges using exact Wasserstein-1
    (via POT) falling back to a greedy approximation if POT is unavailable.
  - Weakens edges with persistently negative curvature (semantic tension).
  - Strengthens edges with positive curvature (semantic coherence).
  - Creates new edges between nodes that share coherent neighborhoods,
    tightening galaxy clusters.
  - Updates edge weights using feedback from the belief metric trace,
    closing the loop between geometry and embedding structure (matching
    the simulation's step 7 feedback rule).
  - Identifies coherent clusters (galaxies) and surfaces isolated nodes
    for the illumination nagging signal.

Relationship to existing modules
----------------------------------
  SIRPropagationAnalyzer  -- blast radius of contradictions after detection
  SALCoherenceLayer       -- spectral convergence before committing
  EpistaticGate           -- silencing of beliefs under governance pressure
  VeritasGeometria        -- geometric consolidation between sessions

Integration
-----------
  geometria = VeritasGeometria(graph_engine)

  # Background consolidation pass
  result = geometria.consolidate(GraphZone.WORK_KNOWLEDGE)
  print(result.summary)

  # Dry run first to preview
  result = geometria.consolidate(GraphZone.WORK_KNOWLEDGE, dry_run=True)

  # Enrich illumination results with curvature signal
  enriched = geometria.enrich_illumination(illumination_results, GraphZone.WORK_KNOWLEDGE)

  # Raw curvature map for diagnostics
  cmap = geometria.curvature_map(GraphZone.WORK_KNOWLEDGE)

Dependencies
-------------
numpy       -- required, already in VM requirements
POT (ot)    -- optional but strongly recommended for accurate curvature.
               Falls back to greedy W1 approximation if unavailable.
               pip install POT

Deferred features
------------------
The following extensions are architecturally sound but require changes
outside this module before they can be implemented correctly. They are
documented here so the path forward is clear when the prerequisites land.

  DIRECTIONAL CURVATURE (concept flow geometry)
  -----------------------------------------------
  Current curvature treats all edges as undirected geometry. VM's EdgeKind
  enum is already rich with directional semantics -- SUPPORTS, DEPENDS_ON,
  IMPLEMENTS, REFINES, TEMPORAL_NEXT, FACT_UPDATES_BELIEF,
  EVIDENCE_SUPPORTS_DECISION all have clear flow direction.

  Directed Ollivier-Ricci curvature would compare outgoing vs incoming
  transport rather than symmetric neighborhoods:

    kappa_forward(u,v)   = 1 - W1(out_dist(u), in_dist(v))  / d(u,v)
    kappa_backward(u,v)  = 1 - W1(out_dist(v), in_dist(u))  / d(u,v)
    kappa_asymmetry(u,v) = kappa_forward - kappa_backward

  Large asymmetry identifies reasoning channels -- edges that carry concept
  flow strongly in one direction (gravity -> planetary motion, but not the
  reverse). Positive directional curvature along directed paths would
  produce concept hierarchies (Physics -> mechanics, thermodynamics,
  electromagnetism) naturally from the geometry rather than by explicit
  tagging. Contradiction edges between clusters would produce strongly
  negative curvature in both directions, forming geometric tension
  boundaries.

  PREREQUISITE: GraphEngine._adj is out-edges only. Directed curvature
  requires in_dist(v), which means an in-edges index. Two options:

    Option A -- add _adj_in to GraphEngine (preferred):
      A persistent reverse adjacency cache maintained alongside _adj.
      Clean, no per-pass overhead, consistent with how _adj works.
      Requires modifying GraphEngine.__init__, _load_graph, add_edge,
      and _rebuild_cache.

    Option B -- build reverse index per consolidation pass (not recommended):
      O(E) scan of _adj on every pass to reconstruct in-edges.
      Works but wastes computation and obscures the architectural intent.

  When Option A lands, directed curvature slots in as a separate
  _directed_ollivier_ricci(u, v, adj, adj_in, ...) method. The existing
  undirected path stays for SEMANTIC_SIMILARITY edges where direction
  is arbitrary. ConsolidationResult gains optional directed_curvature_map
  and reasoning_channels fields without breaking the existing interface.

  Tracked in: deferred_concepts.md
"""

from __future__ import annotations


import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set, Tuple


import numpy as np

from graph_types import GraphZone

try:
    import ot as _ot
    _POT_AVAILABLE = True
except ImportError:
    _POT_AVAILABLE = False


logger = logging.getLogger(__name__)

if not _POT_AVAILABLE:
    logger.warning(
        "VeritasGeometria: POT library not found. "
        "Falling back to greedy Wasserstein-1 approximation. "
        "Install with: pip install POT"
    )


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_REMOVAL_THRESHOLD:   float = -0.3
DEFAULT_CREATION_THRESHOLD:  float = 0.25
DEFAULT_WEAKENING_RATE:      float = 0.15
DEFAULT_STRENGTHENING_RATE:  float = 0.10
DEFAULT_MIN_WEIGHT:          float = 0.05
DEFAULT_MAX_WEIGHT:          float = 2.0
DEFAULT_MAX_NEW_EDGES:       int   = 50
DEFAULT_MAX_NODES:           int   = 2000
DEFAULT_FEEDBACK_RATE:       float = 0.005

# HSH zones eligible for geometric consolidation during sleep pass.
# GOVERNANCE is excluded — it is the gravitational anchor of the disk and
# curvature-driven rewiring of authoritative directives is never appropriate.
# RATIONALE is included: rationale graphs evolve as reasoning matures and
# ORC-driven rewiring correctly surfaces emerging conceptual clusters.
# WORK_KNOWLEDGE and TEMPORAL_KNOWLEDGE are included for matter context
# evolution and expiry-driven topology changes respectively.
_ALLOWED_ZONES: set = {
    GraphZone.RATIONALE.value,
    GraphZone.WORK_KNOWLEDGE.value,
    GraphZone.TEMPORAL_KNOWLEDGE.value,
}


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class GalaxyCluster:
    cluster_id:          int
    node_ids:            List[str]
    mean_internal_kappa: float
    cohesion_score:      float
    bridge_edges:        List[Tuple[str, str, float]]
    zone:                str


@dataclass
class ConsolidationResult:
    zone:               str
    analyzed_at:        str
    nodes_analyzed:     int
    edges_analyzed:     int
    edges_weakened:     int
    edges_minimized:    int
    edges_created:      int
    edges_strengthened: int
    mean_kappa_before:  float
    mean_kappa_after:   float
    galaxies:           List[GalaxyCluster]
    galaxies_detected:  int
    curvature_map:      Dict[str, float]
    summary:            str
    dry_run:            bool
    bridges_detected:   int = 0
    bridges_protected:  int = 0


@dataclass
class CurvatureMap:
    zone:                  str
    computed_at:           str
    edge_kappas:           Dict[str, float]
    node_mean_kappa:       Dict[str, float]
    high_curvature_nodes:  List[str]
    low_curvature_nodes:   List[str]
    mean_kappa:            float


class VeritasGeometria:

    def __init__(
        self,
        graph_engine,
        removal_threshold:   float = DEFAULT_REMOVAL_THRESHOLD,
        creation_threshold:  float = DEFAULT_CREATION_THRESHOLD,
        weakening_rate:      float = DEFAULT_WEAKENING_RATE,
        strengthening_rate:  float = DEFAULT_STRENGTHENING_RATE,
        min_weight:          float = DEFAULT_MIN_WEIGHT,
        max_weight:          float = DEFAULT_MAX_WEIGHT,
        max_new_edges:       int   = DEFAULT_MAX_NEW_EDGES,
        max_nodes:           int   = DEFAULT_MAX_NODES,
        feedback_rate:       float = DEFAULT_FEEDBACK_RATE,
        galaxy_threshold:    Optional[float] = 0.0,
        protect_bridges:     bool  = True,
    ):
        self.graph              = graph_engine
        self.removal_threshold  = removal_threshold
        self.creation_threshold = creation_threshold
        self.weakening_rate     = weakening_rate
        self.strengthening_rate = strengthening_rate
        self.min_weight         = min_weight
        self.max_weight         = max_weight
        self.max_new_edges      = max_new_edges
        self.max_nodes          = max_nodes
        self.feedback_rate      = feedback_rate
        self.galaxy_threshold   = galaxy_threshold
        self.protect_bridges    = protect_bridges

    def consolidate(
        self,
        zone,
        dry_run:   bool = False,
        max_nodes: Optional[int] = None,
    ) -> ConsolidationResult:
        """
        Run a full geometry-driven consolidation pass on a zone.

        Steps (mirroring the RBM simulation loop):
          1. Build adjacency snapshot and edge inventory
          2. Build graph distance cache (ground metric for W1)
          3. Compute Ollivier-Ricci curvature per edge
          4. Compute belief metric trace per node
          5. Identify structural bridges (protect_bridges=True by default)
          6. Decide: weaken / strengthen / remove edges by curvature
             Bridges are never minimized -- they hold inter-cluster connectivity.
          7. Apply belief metric feedback to edge weights (closes the
             B -> w -> kappa -> G -> B loop from the simulation)
          8. Find edge creation candidates from shared coherent neighbors
          9. Write changes through GraphEngine (unless dry_run=True)
         10. Identify galaxy clusters from final curvature structure

        Parameters
        ----------
        zone : GraphZone
            Zone to consolidate. Must be KNOWLEDGE or WORK.
        dry_run : bool
            Compute and report changes but do not write them.
        max_nodes : int, optional
            Override instance max_nodes for this pass.
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        now      = _utc_now()

        if zone_val not in _ALLOWED_ZONES:
            return _empty_result(zone_val, now, dry_run,
                f"Zone '{zone_val}' is not eligible for geometry consolidation.")

        limit    = max_nodes or self.max_nodes
        adj      = self.graph._adj.get(zone, {})
        blocked  = self.graph._blocked_nodes

        all_nodes = [n for n in adj.keys() if n not in blocked]
        if len(all_nodes) > limit:
            seeds    = set(random.sample(all_nodes, min(limit // 2, len(all_nodes))))
            expanded = set(seeds)
            for seed in seeds:
                for dst, *_ in adj.get(seed, []):
                    if dst not in blocked:
                        expanded.add(dst)
                    if len(expanded) >= limit:
                        break
                if len(expanded) >= limit:
                    break
            all_nodes = list(expanded)
            logger.info(
                "VeritasGeometria: ego-expanded sample -- %d seeds -> %d nodes "
                "(limit %d) in %s",
                len(seeds), len(all_nodes), limit, zone_val
            )

        if len(all_nodes) < 3:
            return _empty_result(zone_val, now, dry_run,
                f"Zone '{zone_val}' has too few eligible nodes ({len(all_nodes)}).")

        node_set       = set(all_nodes)
        edge_inventory = self._build_edge_inventory(adj, node_set, blocked)

        if not edge_inventory:
            return _empty_result(zone_val, now, dry_run,
                f"No eligible edges found in zone '{zone_val}'.")

        graph_dist = self._build_distance_cache(adj, node_set, blocked)

        kappas: Dict[Tuple[str, str], float] = {
            ek: self._ollivier_ricci(ek[0], ek[1], adj, node_set, blocked, graph_dist)
            for ek in edge_inventory
        }

        mean_kappa_before = sum(kappas.values()) / len(kappas)
        b_traces = self._compute_belief_traces(adj, node_set, blocked)

        kappa_vals      = list(kappas.values())
        kappa_mean      = float(np.mean(kappa_vals))
        kappa_std       = float(np.std(kappa_vals)) if len(kappa_vals) > 1 else 0.1
        weaken_thresh   = kappa_mean - kappa_std if kappa_std > 1e-4 else self.removal_threshold
        strengthen_thresh = kappa_mean + kappa_std if kappa_std > 1e-4 else self.creation_threshold

        bridges: Set[Tuple[str, str]] = set()
        if self.protect_bridges:
            bridges = self._find_bridges(adj, node_set, blocked)
            if bridges:
                logger.debug(
                    "VeritasGeometria: %d bridge edge(s) detected in zone '%s' -- "
                    "will not be minimized during consolidation.",
                    len(bridges), zone_val,
                )

        to_weaken:     List[Tuple] = []
        to_remove:     List[Tuple] = []
        to_strengthen: List[Tuple] = []
        bridges_protected = 0

        for edge_key, kappa in kappas.items():
            edata      = edge_inventory[edge_key]
            src, dst   = edge_key
            is_bridge  = edge_key in bridges

            if kappa < weaken_thresh:
                new_w = edata["weight"] * (1.0 - self.weakening_rate)
                if new_w < self.min_weight:
                    if is_bridge:
                        # Bridge: never minimize. Weaken only to min_weight floor.
                        # Keeps inter-cluster connectivity intact under stress.
                        bridges_protected += 1
                        logger.debug(
                            "VeritasGeometria: bridge %s<->%s shielded from minimization "
                            "(kappa=%.4f). Held at min_weight floor.",
                            src, dst, kappa,
                        )
                    else:
                        to_remove.append((src, dst, edata))
                else:
                    to_weaken.append((src, dst, new_w, edata))
            elif kappa > strengthen_thresh:
                new_w = min(
                    edata["weight"] * (1.0 + self.strengthening_rate),
                    self.max_weight
                )
                if new_w > edata["weight"]:
                    to_strengthen.append((src, dst, new_w, edata))

        skip_feedback = (
            {(s, d) for s, d, *_ in to_weaken} |
            {(s, d) for s, d, *_ in to_remove}
        )
        feedback_updates: List[Tuple] = []
        for edge_key, edata in edge_inventory.items():
            if edge_key in skip_feedback:
                continue
            src, dst = edge_key
            tr_avg   = (b_traces.get(src, 0.0) + b_traces.get(dst, 0.0)) / 2.0
            if tr_avg > 0:
                new_w = float(np.clip(
                    edata["weight"] * (1.0 + self.feedback_rate * tr_avg),
                    self.min_weight, self.max_weight
                ))
                if abs(new_w - edata["weight"]) > 1e-4:
                    feedback_updates.append((src, dst, new_w, edata))

        creation_candidates = self._find_creation_candidates(
            kappas, node_set, blocked,
            existing_edges=set(edge_inventory.keys()),
            creation_threshold=strengthen_thresh,
        )[:self.max_new_edges]

        edges_weakened = edges_minimized = edges_strengthened = edges_created = 0

        if not dry_run:
            Edge, EdgeKind, GateLevel = self._import_graph_classes()
            if Edge is None or EdgeKind is None or GateLevel is None:
                return _empty_result(zone_val, now, dry_run,
                    "Import error -- could not write changes. Check package structure.")

            for src, dst, new_w, edata in to_weaken:
                if self._write_edge(zone, src, dst, new_w, edata, Edge, GateLevel):
                    edges_weakened += 1

            for src, dst, edata in to_remove:
                if self._write_edge(zone, src, dst, self.min_weight, edata, Edge, GateLevel):
                    edges_minimized += 1

            for src, dst, new_w, edata in to_strengthen:
                if self._write_edge(zone, src, dst, new_w, edata, Edge, GateLevel):
                    edges_strengthened += 1

            feedback_attempted = len(feedback_updates)
            feedback_succeeded = 0
            for src, dst, new_w, edata in feedback_updates:
                if self._write_edge(zone, src, dst, new_w, edata, Edge, GateLevel):
                    feedback_succeeded += 1

            if feedback_succeeded < feedback_attempted:
                logger.warning(
                    "VeritasGeometria: %d of %d feedback writes failed in zone '%s'. "
                    "Cache reflects partial update.",
                    feedback_attempted - feedback_succeeded, feedback_attempted, zone_val
                )

            for src, dst in creation_candidates:
                try:
                    e = Edge(
                        graph=zone, src_id=src, dst_id=dst,
                        kind=EdgeKind.SEMANTIC_SIMILARITY,
                        weight=0.5, gate=GateLevel.SUGGEST,
                        source_ids=[],
                    )
                    self.graph.add_edge(e)
                    edges_created += 1
                except Exception as ex:
                    logger.debug("VeritasGeometria: edge creation skipped %s->%s: %s", src, dst, ex)

            total_written = edges_weakened + edges_minimized + edges_strengthened + edges_created
            if total_written > 0:
                self.graph._rebuild_cache()
            elif (len(to_weaken) + len(to_remove) + len(to_strengthen) + len(creation_candidates)) > 0:
                logger.warning(
                    "VeritasGeometria: all writes failed in zone '%s'. "
                    "Cache not rebuilt -- state unchanged.", zone_val
                )

        else:
            edges_weakened     = len(to_weaken)
            edges_minimized    = len(to_remove)
            edges_strengthened = len(to_strengthen)
            edges_created      = len(creation_candidates)

        if not dry_run and (edges_weakened + edges_minimized + edges_strengthened + edges_created) > 0:
            adj2    = self.graph._adj.get(zone, {})
            einv2   = self._build_edge_inventory(adj2, node_set, blocked)
            dist2   = self._build_distance_cache(adj2, node_set, blocked)
            kappas2 = {
                ek: self._ollivier_ricci(ek[0], ek[1], adj2, node_set, blocked, dist2)
                for ek in einv2
            }
            mean_kappa_after  = sum(kappas2.values()) / len(kappas2) if kappas2 else mean_kappa_before
            kappas_for_galaxy = kappas2
        else:
            mean_kappa_after  = mean_kappa_before
            kappas_for_galaxy = kappas

        galaxies = self._identify_galaxies(kappas_for_galaxy, node_set, zone_val)

        summary = _build_summary(
            zone_val, len(all_nodes), len(kappas),
            edges_weakened, edges_minimized, edges_strengthened, edges_created,
            mean_kappa_before, mean_kappa_after, len(galaxies), dry_run,
            len(bridges), bridges_protected,
        )
        logger.info("VeritasGeometria: %s", summary.splitlines()[0])

        return ConsolidationResult(
            zone=zone_val, analyzed_at=now,
            nodes_analyzed=len(all_nodes),
            edges_analyzed=len(kappas),
            edges_weakened=edges_weakened,
            edges_minimized=edges_minimized,
            edges_created=edges_created,
            edges_strengthened=edges_strengthened,
            mean_kappa_before=round(mean_kappa_before, 4),
            mean_kappa_after=round(mean_kappa_after, 4),
            galaxies=galaxies,
            galaxies_detected=len(galaxies),
            curvature_map={f"{k[0]}:{k[1]}": round(v, 4) for k, v in kappas.items()},
            summary=summary,
            dry_run=dry_run,
            bridges_detected=len(bridges),
            bridges_protected=bridges_protected,
        )

    def curvature_map(self, zone) -> CurvatureMap:
        """Compute curvature map without making any changes.

        Used for diagnostics and to enrich illumination results with
        geometric isolation signal before calling enrich_illumination().
        """
        zone_val = zone.value if hasattr(zone, "value") else str(zone)
        now      = _utc_now()
        adj      = self.graph._adj.get(zone, {})
        blocked  = self.graph._blocked_nodes
        node_set = set(n for n in adj.keys() if n not in blocked)

        edge_inv   = self._build_edge_inventory(adj, node_set, blocked)
        graph_dist = self._build_distance_cache(adj, node_set, blocked)

        kappas = {
            ek: self._ollivier_ricci(ek[0], ek[1], adj, node_set, blocked, graph_dist)
            for ek in edge_inv
        }

        node_k: Dict[str, List[float]] = defaultdict(list)
        for (src, dst), k in kappas.items():
            node_k[src].append(k)
            node_k[dst].append(k)

        node_mean = {n: round(sum(v) / len(v), 4) for n, v in node_k.items()}
        mean_k    = sum(kappas.values()) / len(kappas) if kappas else 0.0

        return CurvatureMap(
            zone=zone_val, computed_at=now,
            edge_kappas={f"{k[0]}:{k[1]}": round(v, 4) for k, v in kappas.items()},
            node_mean_kappa=node_mean,
            high_curvature_nodes=[n for n, k in node_mean.items() if k > mean_k + 0.1],
            low_curvature_nodes=[n for n, k in node_mean.items() if k < mean_k - 0.1],
            mean_kappa=round(mean_k, 4),
        )

    def enrich_illumination(
        self,
        illumination_results: list,
        zone,
        isolation_boost: float = 1.4,
    ) -> list:
        """
        Enrich IlluminationResult objects with curvature signal.

        Geometrically isolated nodes (low mean kappa) get a boosted
        nagging_score -- memories not yet integrated into any coherent
        cluster. Plugs into the same pattern as SALCoherenceLayer.enrich_illumination().
        """
        cmap    = self.curvature_map(zone)
        low_set = set(cmap.low_curvature_nodes)

        for result in illumination_results:
            setattr(result, "mean_kappa", cmap.node_mean_kappa.get(result.node_id))
            if result.node_id in low_set:
                result.nagging_score *= isolation_boost
                setattr(
                    result,
                    "geometry_note",
                    "Geometrically isolated -- low Ollivier-Ricci curvature. "
                    "Not integrated into any coherent conceptual cluster.",
                )
            else:
                setattr(result, "geometry_note", "Member of coherent conceptual cluster.")

        illumination_results.sort(key=lambda r: r.nagging_score, reverse=True)
        return illumination_results

    def _ollivier_ricci(
        self,
        u: str, v: str,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
        graph_dist: Dict[str, Dict[str, float]],
    ) -> float:
        m_u = self._neighborhood_dist(u, adj, node_set, blocked)
        m_v = self._neighborhood_dist(v, adj, node_set, blocked)
        if not m_u or not m_v:
            return 0.0

        d_uv = graph_dist.get(u, {}).get(v, None)
        if d_uv is None or d_uv >= 50.0:
            return 0.0
        d_uv = max(d_uv, 1e-10)
        w1   = (
            self._w1_exact(m_u, m_v, graph_dist, adj)
            if _POT_AVAILABLE
            else self._w1_greedy(m_u, m_v, adj)
        )
        return round(1.0 - w1 / d_uv, 6)

    def _neighborhood_dist(
        self,
        node: str,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        neighbors: List[Tuple[str, float]] = []
        total_w = 0.0

        for dst, kind, weight, gate in adj.get(node, []):
            if dst not in node_set or dst in blocked:
                continue
            if (gate.value if hasattr(gate, "value") else str(gate)) == "block_until_resolved":
                continue
            if (kind.value if hasattr(kind, "value") else str(kind)) == "contradicts":
                continue
            w = float(weight)
            neighbors.append((dst, w))
            total_w += w

        dist: Dict[str, float] = {node: 1.0 - alpha}
        if not neighbors or total_w == 0.0:
            return dist
        for dst, w in neighbors:
            dist[dst] = alpha * (w / total_w)
        return dist

    def _w1_exact(
        self,
        m_u: Dict[str, float],
        m_v: Dict[str, float],
        graph_dist: Dict[str, Dict[str, float]],
        adj: Dict,
    ) -> float:
        nu = list(m_u.keys())
        nv = list(m_v.keys())
        a  = np.array([m_u[n] for n in nu], dtype=np.float64)
        b  = np.array([m_v[n] for n in nv], dtype=np.float64)
        a /= a.sum()
        b /= b.sum()
        M = np.array(
            [[float(graph_dist.get(i, {}).get(j, 50.0)) for j in nv] for i in nu],
            dtype=np.float64
        )
        if (M == 50.0).any():
            logger.debug(
                "W1 exact: disconnected node pair(s) in neighborhood -- "
                "fallback distance 50.0 used. Curvature estimate may be unreliable."
            )
        try:
            result: float = float(_ot.emd2(a, b, M))  # type: ignore[arg-type]
            return result
        except Exception as ex:
            logger.debug("POT W1 failed, using greedy: %s", ex)
            return self._w1_greedy(m_u, m_v, adj)

    def _w1_greedy(
        self,
        m_u: Dict[str, float],
        m_v: Dict[str, float],
        adj: Dict,
    ) -> float:
        all_n = set(m_u) | set(m_v)
        surplus: Dict[str, float] = {}
        deficit: Dict[str, float] = {}
        for n in all_n:
            diff = m_u.get(n, 0.0) - m_v.get(n, 0.0)
            if diff > 1e-9:
                surplus[n] = diff
            elif diff < -1e-9:
                deficit[n] = -diff

        if not surplus and not deficit:
            return 0.0

        cost = 0.0
        rs   = dict(surplus)
        rd   = dict(deficit)

        for s_node in list(rs):
            if rs.get(s_node, 0) < 1e-9:
                continue
            for d_node in {dst for dst, *_ in adj.get(s_node, [])} & rd.keys():
                if rs.get(s_node, 0) < 1e-9:
                    break
                if rd.get(d_node, 0) < 1e-9:
                    continue
                t = min(rs[s_node], rd[d_node])
                cost += t
                rs[s_node] -= t
                rd[d_node] -= t

        unmatched = max(
            sum(v for v in rs.values() if v > 1e-9),
            sum(v for v in rd.values() if v > 1e-9),
        )
        cost += unmatched * 2.0
        return cost

    def _build_distance_cache(
        self,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
        max_depth: int = 6,
    ) -> Dict[str, Dict[str, float]]:
        """BFS unweighted distance cache (mirrors nx.all_pairs_shortest_path_length)."""
        dist: Dict[str, Dict[str, float]] = {}
        for start in node_set:
            if start in blocked:
                continue
            d: Dict[str, float] = {start: 0.0}
            queue = deque([(start, 0)])
            while queue:
                cur, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for dst, kind, weight, gate in adj.get(cur, []):
                    if dst not in node_set or dst in blocked:
                        continue
                    if (gate.value if hasattr(gate, "value") else str(gate)) == "block_until_resolved":
                        continue
                    if dst not in d:
                        d[dst] = float(depth + 1)
                        queue.append((dst, depth + 1))
            dist[start] = d
        return dist

    def _compute_belief_traces(
        self,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
    ) -> Dict[str, float]:
        traces: Dict[str, float] = {}
        for node in node_set:
            if node in blocked:
                continue
            total  = 0.0
            degree = 0
            for dst, kind, weight, gate in adj.get(node, []):
                if dst not in node_set or dst in blocked:
                    continue
                if (gate.value if hasattr(gate, "value") else str(gate)) == "block_until_resolved":
                    continue
                if (kind.value if hasattr(kind, "value") else str(kind)) == "contradicts":
                    continue
                total  += float(weight)
                degree += 1
            traces[node] = (total / degree) if degree > 0 else 0.0
        return traces

    def _build_edge_inventory(
        self,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
    ) -> Dict[Tuple[str, str], Dict]:
        inventory: Dict[Tuple[str, str], Dict] = {}
        for src in node_set:
            if src in blocked:
                continue
            for dst, kind, weight, gate in adj.get(src, []):
                if dst not in node_set or dst in blocked:
                    continue
                if (gate.value if hasattr(gate, "value") else str(gate)) == "block_until_resolved":
                    continue
                if (kind.value if hasattr(kind, "value") else str(kind)) == "contradicts":
                    continue
                ek = (min(src, dst), max(src, dst))
                if ek not in inventory:
                    inventory[ek] = {
                        "src": src, "dst": dst,
                        "kind": kind, "weight": float(weight), "gate": gate,
                    }
        return inventory

    @staticmethod
    def _import_graph_classes():
        try:
            from graph_types import (
                Edge,
                EdgeKind,
                GateLevel,
            )
            return Edge, EdgeKind, GateLevel
        except ImportError:
            return None, None, None

    def _write_edge(
        self,
        zone, src: str, dst: str, new_weight: float,
        edata: Dict, Edge, GateLevel,
    ) -> bool:
        try:
            e = Edge(
                graph=zone, src_id=src, dst_id=dst,
                kind=edata["kind"], weight=new_weight,
                gate=edata["gate"], source_ids=[],
            )
            self.graph.add_edge(e)
            return True
        except Exception as ex:
            logger.debug("VeritasGeometria: write failed %s->%s: %s", src, dst, ex)
            return False

    def _find_bridges(
        self,
        adj: Dict,
        node_set: Set[str],
        blocked: Dict,
    ) -> Set[Tuple[str, str]]:
        """
        Iterative Tarjan bridge detection.

        Returns the set of (min_id, max_id) edge keys that are structural
        bridges -- edges whose removal would increase the number of connected
        components. These edges are protected from minimization during
        consolidation because they hold inter-cluster connectivity together.

        Uses an iterative DFS to avoid Python's recursion limit on large graphs.
        Applies the same adjacency filters as the rest of the consolidation
        pass: skips blocked nodes, block_until_resolved gates, and contradicts
        edges.

        Complexity: O(V + E).
        """
        def _eligible_neighbors(node: str):
            for dst, kind, weight, gate in adj.get(node, []):
                if dst not in node_set or dst in blocked:
                    continue
                if (gate.value if hasattr(gate, "value") else str(gate)) == "block_until_resolved":
                    continue
                if (kind.value if hasattr(kind, "value") else str(kind)) == "contradicts":
                    continue
                yield dst

        visited:  Dict[str, int] = {}
        low:      Dict[str, int] = {}
        bridges:  Set[Tuple[str, str]] = set()
        timer:    List[int] = [0]

        for start in node_set:
            if start in visited or start in blocked:
                continue

            visited[start] = low[start] = timer[0]
            timer[0] += 1

            # Stack entries: (node, parent_node_or_None, neighbor_iterator)
            stack: List[Tuple[str, Optional[str], Iterator[str]]] = [
                (start, None, _eligible_neighbors(start))
            ]

            while stack:
                node, parent, nbr_iter = stack[-1]
                try:
                    dst = next(nbr_iter)
                    if dst == parent:
                        # Skip the single back-edge to parent in undirected graph.
                        # Assumption: no parallel edges (consistent with VM data model).
                        continue
                    if dst not in visited:
                        visited[dst] = low[dst] = timer[0]
                        timer[0] += 1
                        stack.append((dst, node, _eligible_neighbors(dst)))
                    else:
                        # Cross/back edge: update low without bridge check
                        low[node] = min(low[node], visited[dst])
                except StopIteration:
                    stack.pop()
                    if stack:
                        par_node = stack[-1][0]
                        low[par_node] = min(low[par_node], low[node])
                        if low[node] > visited[par_node]:
                            bridges.add((min(par_node, node), max(par_node, node)))

        return bridges

    def _find_creation_candidates(
        self,
        kappas: Dict[Tuple[str, str], float],
        node_set: Set[str],
        blocked: Dict,
        existing_edges: Set[Tuple[str, str]],
        creation_threshold: Optional[float] = None,
    ) -> List[Tuple[str, str]]:
        threshold = creation_threshold if creation_threshold is not None else self.creation_threshold
        hk_adj: Dict[str, Set[str]] = defaultdict(set)
        for (src, dst), kappa in kappas.items():
            if kappa > threshold:
                hk_adj[src].add(dst)
                hk_adj[dst].add(src)

        candidates: Set[Tuple[str, str]] = set()
        for v in node_set:
            if v in blocked:
                continue
            nbrs = list(hk_adj.get(v, set()))
            if len(nbrs) < 2:
                continue
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    ek = (min(nbrs[i], nbrs[j]), max(nbrs[i], nbrs[j]))
                    if ek not in existing_edges:
                        candidates.add(ek)
            if len(candidates) >= self.max_new_edges * 3:
                break

        return list(candidates)

    def _identify_galaxies(
        self,
        kappas: Dict[Tuple[str, str], float],
        node_set: Set[str],
        zone_val: str,
    ) -> List[GalaxyCluster]:
        kappa_vals = list(kappas.values())
        if self.galaxy_threshold is None:
            threshold = float(np.mean(kappa_vals)) if kappa_vals else 0.0
        else:
            threshold = self.galaxy_threshold

        pos_adj: Dict[str, Set[str]] = defaultdict(set)
        for (src, dst), kappa in kappas.items():
            if kappa > threshold:
                pos_adj[src].add(dst)
                pos_adj[dst].add(src)

        visited: Dict[str, int] = {}
        cid = 0
        clusters: Dict[int, List[str]] = {}

        for node in node_set:
            if node in visited:
                continue
            queue   = [node]
            visited[node] = cid
            members = [node]
            while queue:
                cur = queue.pop()
                for nb in pos_adj.get(cur, set()):
                    if nb not in visited:
                        visited[nb] = cid
                        members.append(nb)
                        queue.append(nb)
            clusters[cid] = members
            cid += 1

        galaxies: List[GalaxyCluster] = []
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            mset             = set(members)
            internal_kappas  = [k for (s, d), k in kappas.items() if s in mset and d in mset]
            bridge_edges     = [(s, d, k) for (s, d), k in kappas.items()
                                if (s in mset) != (d in mset)]
            mean_internal    = sum(internal_kappas) / len(internal_kappas) if internal_kappas else 0.0
            n                = len(members)
            possible         = n * (n - 1) / 2
            cohesion         = len(internal_kappas) / possible if possible > 0 else 0.0

            galaxies.append(GalaxyCluster(
                cluster_id=cluster_id,
                node_ids=members,
                mean_internal_kappa=round(mean_internal, 4),
                cohesion_score=round(cohesion, 4),
                bridge_edges=bridge_edges,
                zone=zone_val,
            ))

        galaxies.sort(key=lambda g: len(g.node_ids), reverse=True)
        return galaxies


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_result(zone_val: str, now: str, dry_run: bool, reason: str) -> ConsolidationResult:
    return ConsolidationResult(
        zone=zone_val, analyzed_at=now,
        nodes_analyzed=0, edges_analyzed=0,
        edges_weakened=0, edges_minimized=0,
        edges_created=0, edges_strengthened=0,
        mean_kappa_before=0.0, mean_kappa_after=0.0,
        galaxies=[], galaxies_detected=0,
        curvature_map={},
        summary=reason, dry_run=dry_run,
    )


def _build_summary(
    zone: str, nodes: int, edges: int,
    weakened: int, removed: int, strengthened: int, created: int,
    kappa_before: float, kappa_after: float,
    galaxies: int, dry_run: bool,
    bridges_detected: int = 0, bridges_protected: int = 0,
) -> str:
    mode  = "DRY RUN" if dry_run else "APPLIED"
    delta = kappa_after - kappa_before
    sign  = "+" if delta >= 0 else ""
    bridge_line = (
        f"  Bridges detected:   {bridges_detected}  "
        f"(protected from minimization: {bridges_protected})\n"
        if bridges_detected > 0
        else ""
    )
    return (
        f"Veritas Geometria [{mode}] -- zone '{zone}'\n"
        f"  Nodes analyzed:     {nodes}\n"
        f"  Edges evaluated:    {edges}\n"
        f"  Edges weakened:     {weakened}\n"
        f"  Edges at minimum:   {removed}\n"
        f"  Edges strengthened: {strengthened}\n"
        f"  Edges created:      {created}\n"
        f"{bridge_line}"
        f"  Mean kappa before:  {kappa_before:.4f}\n"
        f"  Mean kappa after:   {kappa_after:.4f}  ({sign}{delta:.4f})\n"
        f"  Galaxy clusters:    {galaxies}\n"
    )
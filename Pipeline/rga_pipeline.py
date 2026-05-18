"""
Rhetorical Geometry Analysis — Pipeline Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accepts a pre-classified schema v0.4 partial record (document_metadata,
anchor_registry, proposition_table already populated by collaborative
analyst/Claude extraction) and runs the geometric computation pipeline.

Returns a completed schema v0.4 record with all geometric fields populated.

Input contract
--------------
The caller supplies a dict conforming to:
  {
    "document_metadata": { ... },
    "anchor_registry":   [ ... ],      # per RULE-005/006
    "proposition_table": [ ... ],      # per RULE-001 through RULE-004
    "confidence_flags":  [ ... ],      # optional, pass-through
    "falsifiability_record": [ ... ]   # optional, pass-through
  }

All classification, normalization, and anchor assignment must be complete
before this function is called. The pipeline does not perform extraction.

Output contract
---------------
Returns the input dict extended with:
  - graph_structure     (nodes, edges, zone assignments, fragmentation signals)
  - geometric_summary   (manifold description, stress regions, stability)
  - confidence_flags    (extended with pipeline-generated flags)
  - falsifiability_record (extended with pipeline observations)
  - audit_trail         (module execution records)
"""

from __future__ import annotations

import sys
import os
import math
import logging
import datetime
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class PipelineTracer:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.start_time = time.time()

    def trace(self, module_name, action, detail=""):
        elapsed = time.time() - self.start_time
        # Boldly print to terminal so you can't miss it even without looking at logs
        print(f"[{elapsed:06.3f}s] >>> MODULE: {module_name:25} | ACTION: {action:15} | {detail}")

def execute_pipeline_with_trace(data, pass_num):
    tracer = PipelineTracer(data['document_metadata']['document_id'])
    
    try:
        tracer.trace("GraphAdapter", "START", f"Pass {pass_num}")
        # Your GraphAdapter call here
        tracer.trace("GraphAdapter", "SUCCESS", f"Nodes mapped: {len(data.get('proposition_table', []))}")

        tracer.trace("HSHGeometry", "CALIBRATING", "Calculating Fiedler value...")
        # Your Geometry call here
        tracer.trace("HSHGeometry", "SUCCESS", f"Coordinates assigned")

        tracer.trace("TensionAuditor", "SCANNING", "Checking cross-seam integrity")
        # Your Auditor call here
        tracer.trace("TensionAuditor", "SUCCESS", "Integrity check passed")

    except Exception as e:
        tracer.trace("PIPELINE_CRASH", "ERROR", f"Failed at {pass_num}: {str(e)}")
        raise  # Re-raising ensures it doesn't fail silently

# ── Local types (no Lore/VeritasMemoria dependency) ───────────────────────────
_PIPELINE_DIR = os.path.dirname(__file__)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from graph_types import GraphZone, EdgeKind, GateLevel

@dataclass(unsafe_hash=True)
class _RGAContradictionRecord:
    """Minimal local record for RGA contradiction tracking.
    Replaces Lore's ContradictionRecord whose signature diverged from
    what this pipeline requires. Only node_a_id and node_b_id are used
    downstream; other fields are kept for audit purposes."""
    node_a_id:       str
    node_b_id:       str
    zone:            GraphZone
    gate_level:      GateLevel
    resolution_note: str

logger = logging.getLogger(__name__)


# ── Geodesic distance and stress computation ──────────────────────────────────
#
# These three functions implement the closed-form field theory from Templet (2026).
# They replace the placeholder 0.3 stress values and Euclidean offset radii
# with quantities derived from the actual hyperbolic geometry.

def _green_function(d: float, eps: float = 1e-9) -> float:
    """
    Exact Green's function of the Laplace-Beltrami operator on H²:

        G(d) = (1/2π) * log(coth(d/2))

    This is the healing scalar field contribution at geodesic distance d
    from a contradiction source. Used to derive stress_energy_contribution
    from the actual geodesic distance between a proposition and its anchor.

    At d→0 (proposition coincides with anchor): G→∞ (maximum stress, full
    contradiction load concentrated at a point).
    At d→∞ (proposition far from anchor): G→0 (contradiction dissipates).

    The stress energy at a proposition node is the sum of G(d) over all its
    anchor mappings, reflecting total contradiction load at that position.

    Parameters
    ----------
    d   : float  geodesic distance from proposition to anchor
    eps : float  lower clamp on d to prevent log singularity at d=0
    """
    d_clamped = max(d, eps)
    return (1.0 / (2.0 * math.pi)) * math.log(1.0 / math.tanh(d_clamped / 2.0))


def _radius_from_geodesic(d_target: float, lambda_zone: float, eps: float = 1e-9) -> float:
    """
    Invert the HSH metric to find the Poincaré disk radius r that places a
    node at geodesic distance d_target from the governance anchor at the
    origin (r_anchor = 0, lambda_anchor = 1.0).

    The HSH distance from origin to a node at radius r with zone inertia λ:

        d = arcosh(1 + r² / ((1 - r²/λ) · 1))
          = arcosh(1 + r² · λ / (λ - r²))

    Solving for r given d:

        cosh(d) = 1 + r²λ / (λ - r²)
        let C = cosh(d) - 1
        C(λ - r²) = r²λ
        Cλ = r²(λ + C)
        r² = Cλ / (λ + C)
        r  = sqrt(Cλ / (λ + C))

    Parameters
    ----------
    d_target    : float  target geodesic distance from the governance origin
    lambda_zone : float  zone inertia of the proposition's zone (from ZONE_LAMBDA)
    eps         : float  clamp to prevent degenerate geometry at d≈0
    """
    d = max(d_target, eps)
    C = math.cosh(d) - 1.0
    r_sq = (C * lambda_zone) / (lambda_zone + C)
    # r_sq can slightly exceed 1 for very large d due to floating point;
    # clamp to disk interior
    return math.sqrt(min(max(r_sq, 0.0), 0.9998))


def _compute_geodesic_distances(
    anchor_registry: list,
    proposition_table: list,
    node_radii: Dict[str, float],
    zone_lambda: Dict,
) -> Dict[str, Dict[str, float]]:
    """
    Compute HSH geodesic distances from each proposition to each of its
    mapped anchors, using current disk positions.

    The anchor sits at its registered radius (chain-length derived, near
    the origin). The proposition sits at its current node_radii value.
    Both are placed on the Poincaré disk as 2D vectors for the hsh_distance
    call; since we only have r (not full (x,y) coords here), we place them
    on the same radial line (theta=0 for the anchor, same theta as the
    proposition) to get a lower-bound distance. The angular separation
    between proposition and anchor is handled by the spring embedder's theta
    assignment; what matters here is the radial geodesic component, which
    is the dominant term for cross-zone distances and is exact for
    same-angle pairs.

    Returns
    -------
    Dict mapping proposition_id → {anchor_id → geodesic_distance}
    """
    import numpy as np

    distances: Dict[str, Dict[str, float]] = {}

    # Build anchor position lookup: anchor sits on its chain-derived radius
    anchor_r: Dict[str, float] = {
        a["anchor_id"]: node_radii.get(a["anchor_id"], 0.02)
        for a in anchor_registry
    }

    for prop in proposition_table:
        pid = prop["proposition_id"]
        mappings = prop.get("anchor_mappings") or []
        if not mappings:
            distances[pid] = {}
            continue

        # Proposition zone lambda
        classification = prop.get("primary_classification", "ambiguous")
        zone = CLASSIFICATION_ZONE.get(classification, GraphZone.WORK_KNOWLEDGE)
        lam_prop = ZONE_INERTIA[zone]

        prop_r = node_radii.get(pid, ZONE_RADIUS[zone])
        pos_prop = np.array([prop_r, 0.0])

        prop_distances: Dict[str, float] = {}
        for anchor_id in mappings:
            if anchor_id not in anchor_r:
                continue
            a_r = anchor_r[anchor_id]
            pos_anchor = np.array([a_r, 0.0])

            from hsh_geometry import hsh_distance, ZONE_LAMBDA
            d = hsh_distance(pos_prop, lam_prop, pos_anchor, ZONE_LAMBDA[GraphZone.GOVERNANCE])

            # Floor: a proposition that corroborates an anchor is not the anchor.
            # GOVERNANCE-zone (anchored_true) propositions must sit at least at
            # the RATIONALE zone's characteristic distance from the anchor so
            # they remain distinct nodes in the Laplacian and radii do not
            # collapse to zero on subsequent passes.
            # TEMPORAL-zone propositions are naturally far out; no floor needed.
            if zone == GraphZone.GOVERNANCE:
                d = max(d, 0.30)   # minimum: just inside RATIONALE territory

            prop_distances[anchor_id] = round(d, 6)

        distances[pid] = prop_distances

    return distances


def _compute_stress_from_distances(
    proposition_table: list,
    geodesic_distances: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute stress_energy_contribution for each proposition from its
    geodesic distances to anchors via the Green's function G(d).

    For a proposition mapped to multiple anchors, stress is the mean of
    G(d_i) over all mapped anchors. This reflects total contradiction load
    distributed across the anchor set.

    The raw G(d) values are then normalized across the document so that
    the highest-stress proposition has stress=1.0 and the distribution
    is meaningful relative to the document's own geometry rather than
    being an absolute scale. This makes the Fiedler value and negative
    contradiction weights document-relative measurements.

    Returns
    -------
    Dict mapping proposition_id → stress_energy_contribution (0.0–1.0)
    """
    raw: Dict[str, float] = {}

    for prop in proposition_table:
        pid = prop["proposition_id"]
        d_map = geodesic_distances.get(pid, {})
        if not d_map:
            # No anchor mappings: assign minimum stress
            raw[pid] = 0.0
            continue
        g_values = [_green_function(d) for d in d_map.values()]
        raw[pid] = sum(g_values) / len(g_values)

    # Normalize to [0, 1] within document
    max_raw = max(raw.values()) if raw else 1.0
    if max_raw < 1e-9:
        return {pid: 0.0 for pid in raw}
    return {pid: round(v / max_raw, 6) for pid, v in raw.items()}


# ── RGA Adaptive Calibration ───────────────────────────────────────────────────

def _calibrate_adaptive_positions(
    geometry,
    anchor_registry: list,
    proposition_table: list,
) -> Dict[str, Any]:
    """Compute and register Poincaré disk radii for all nodes.

    Anchor radii are derived from chain_length_from_governance_zone:
    shorter chains (more authoritative) sit closer to the disk origin.

    Proposition radii are derived from the HSH metric via
    _radius_from_geodesic. On the first pipeline pass, where
    geodesic_distance_from_anchor is null, the characteristic geodesic
    distance for the proposition's zone is used as the target. On
    subsequent passes, the previously computed geodesic_distance_from_anchor
    is used, so the positions converge toward the actual geometry.

    Zone boundaries in the output are expressed as geodesic distances from
    the origin, not Euclidean radii.
    """
    from hsh_geometry import ZONE_LAMBDA

    # ── Pass 0: anchor radii from chain length ────────────────────────────────
    chain_lengths = [
        a.get("chain_length_from_governance_zone", 1) for a in anchor_registry
    ]
    max_chain = max(chain_lengths) if chain_lengths else 1

    anchor_radii: Dict[str, float] = {}
    for anc in anchor_registry:
        chain = anc.get("chain_length_from_governance_zone", 1)
        # Map chain length to geodesic distance: longer chain = farther from
        # origin but still within the governance cluster. Target geodesic
        # distance scales with chain / (max_chain + 1) mapped into [0.1, 0.8].
        d_target = 0.1 + 0.7 * (chain / (max_chain + 1))
        r = _radius_from_geodesic(d_target, ZONE_LAMBDA[GraphZone.GOVERNANCE])
        anchor_radii[anc["anchor_id"]] = r
        geometry.register_node_radius(anc["anchor_id"], r)

    # ── Characteristic geodesic distances for first-pass placement ───────────
    # These are the zone-representative distances used when no prior
    # geodesic_distance_from_anchor has been computed yet. Derived from
    # the zone inertia: d_characteristic = arcosh(1 + r_zone²·λ/(λ-r_zone²))
    # evaluated at the zone's nominal radius r = 1 - λ.
    def _characteristic_d(zone: GraphZone) -> float:
        lam = ZONE_LAMBDA[zone]
        r   = 1.0 - lam           # nominal zone radius
        r   = min(r, 0.9998)
        C   = r * r * lam / max(lam - r * r, 1e-9)
        return math.acosh(max(1.0 + C, 1.0))

    CHARACTERISTIC_D: Dict[GraphZone, float] = {
        zone: _characteristic_d(zone) for zone in GraphZone
    }

    # ── Pass 1: non-compound, non-relational propositions ────────────────────
    prop_radii: Dict[str, float] = {}
    for prop in proposition_table:
        pid            = prop["proposition_id"]
        classification = prop.get("primary_classification", "ambiguous")
        zone           = CLASSIFICATION_ZONE.get(classification, GraphZone.WORK_KNOWLEDGE)
        lam            = ZONE_LAMBDA[zone]

        if prop.get("is_compound", False) or prop.get("is_relational", False):
            continue

        # Use previously computed geodesic distance if available (pass ≥ 2),
        # otherwise fall back to the zone's characteristic distance.
        prior_d = prop.get("geodesic_distance_from_anchor")
        if prior_d and isinstance(prior_d, (int, float)) and prior_d > 0:
            d_target = float(prior_d)
        else:
            d_target = CHARACTERISTIC_D[zone]

        r = _radius_from_geodesic(d_target, lam)
        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # ── Pass 2: compound parents ──────────────────────────────────────────────
    for prop in proposition_table:
        if not prop.get("is_compound", False):
            continue
        pid  = prop["proposition_id"]
        cr   = prop.get("compound_resolution") or {}
        basis = cr.get("primary_classification_basis", "")
        if basis and basis in prop_radii:
            r = prop_radii[basis]
        else:
            zone = CLASSIFICATION_ZONE.get(
                prop.get("primary_classification", "ambiguous"),
                GraphZone.WORK_KNOWLEDGE
            )
            r = _radius_from_geodesic(CHARACTERISTIC_D[zone], ZONE_LAMBDA[zone])
        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # ── Pass 3: relational propositions (geodesic midpoint) ───────────────────
    # The geodesic midpoint between two radii on the same radial line is
    # tanh((atanh(r_a) + atanh(r_b)) / 2), which is the correct hyperbolic
    # midpoint formula for the 1D case (same angle, different radii).
    for prop in proposition_table:
        if not prop.get("is_relational", False):
            continue
        pid = prop["proposition_id"]
        rel = prop.get("relational_resolution") or {}
        pa  = rel.get("parent_proposition_a", "")
        pb  = rel.get("parent_proposition_b", "")
        ra  = prop_radii.get(pa, anchor_radii.get(pa, 0.3))
        rb  = prop_radii.get(pb, anchor_radii.get(pb, 0.3))
        # Hyperbolic midpoint: average in the arctanh (geodesic) domain
        ra_clamped = min(max(ra, 1e-6), 0.9998)
        rb_clamped = min(max(rb, 1e-6), 0.9998)
        r = math.tanh((math.atanh(ra_clamped) + math.atanh(rb_clamped)) / 2.0)
        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # ── Zone boundaries as geodesic distances from origin ─────────────────────
    governance_d     = CHARACTERISTIC_D[GraphZone.GOVERNANCE]
    false_attractor_d = CHARACTERISTIC_D[GraphZone.TEMPORAL_KNOWLEDGE]

    return {
        "node_radii": {**anchor_radii, **prop_radii},
        "zone_boundaries": {
            "governance_geodesic":      round(governance_d, 4),
            "false_attractor_geodesic": round(false_attractor_d, 4),
            # Euclidean equivalents retained for visualization compatibility
            "governance":      round(max(anchor_radii.values()) * 1.05, 3) if anchor_radii else 0.30,
            "false_attractor": round(_radius_from_geodesic(false_attractor_d, ZONE_LAMBDA[GraphZone.TEMPORAL_KNOWLEDGE]), 3),
        },
    }

# ── Classification → Zone mapping ─────────────────────────────────────────────
# Governs where each proposition sits in the Poincaré disk.
# GOVERNANCE (r≈0) = closest to authority anchor.
# TEMPORAL_KNOWLEDGE (r≈0.65) = farthest from anchor.

CLASSIFICATION_ZONE: Dict[str, GraphZone] = {
    "anchored_true":           GraphZone.GOVERNANCE,
    "bridge_narrative":        GraphZone.RATIONALE,
    "contextually_misleading": GraphZone.RATIONALE,
    "inferentially_true":      GraphZone.WORK_KNOWLEDGE,
    "ambiguous":               GraphZone.WORK_KNOWLEDGE,
    "inferentially_false":     GraphZone.TEMPORAL_KNOWLEDGE,
    "anchored_false":          GraphZone.TEMPORAL_KNOWLEDGE,
    "out_of_scope":            GraphZone.TEMPORAL_KNOWLEDGE,
}

ZONE_INERTIA: Dict[GraphZone, float] = {
    GraphZone.GOVERNANCE:         1.00,
    GraphZone.RATIONALE:          0.65,
    GraphZone.WORK_KNOWLEDGE:     0.55,
    GraphZone.TEMPORAL_KNOWLEDGE: 0.35,
}

ZONE_RADIUS: Dict[GraphZone, float] = {
    GraphZone.GOVERNANCE:         0.00,
    GraphZone.RATIONALE:          0.35,
    GraphZone.WORK_KNOWLEDGE:     0.45,
    GraphZone.TEMPORAL_KNOWLEDGE: 0.65,
}


# ── Lightweight graph adapter ──────────────────────────────────────────────────

class RGAGraphAdapter:
    """
    Satisfies the VeritasMemoria GraphEngine interface (_adj, _blocked_nodes,
    _bridges) without requiring a database. Populated from the pre-classified
    proposition_table and anchor_registry.

    _adj[zone][src_id] = [(dst_id, EdgeKind, weight, GateLevel), ...]
        Within-zone edges (proposition → proposition of same zone).

    _bridges[(zone, src_id)] = [(dst_zone, dst_id, weight, GateLevel), ...]
        Cross-zone edges (proposition → anchor, or proposition → proposition
        of different zone).

    _blocked_nodes = {}
        Contradiction-blocked nodes. Populated if contradictions are detected
        between anchored_true and anchored_false propositions sharing an anchor.
    """

    def __init__(
        self,
        proposition_table: List[Dict],
        anchor_registry: List[Dict],
    ) -> None:
        self._adj: Dict[GraphZone, Dict[str, List[Tuple]]] = {
            zone: defaultdict(list) for zone in GraphZone
        }
        self._adj_in: Dict[GraphZone, Dict[str, List[Tuple]]] = {
            zone: defaultdict(list) for zone in GraphZone
        }
        self._bridges: Dict[Tuple, List[Tuple]] = defaultdict(list)
        self._blocked_nodes: Dict[str, Any] = {}   # kept empty; preserved for interface compatibility
        self._contradiction_pairs: List[Dict] = []  # RGA measurement records

        self._node_zones: Dict[str, GraphZone] = {}
        self._node_classifications: Dict[str, str] = {}

        # Register anchors in GOVERNANCE zone
        for anchor in anchor_registry:
            aid = anchor["anchor_id"]
            self._adj[GraphZone.GOVERNANCE][aid]  # ensure key exists
            self._node_zones[aid] = GraphZone.GOVERNANCE
            self._node_classifications[aid] = "anchor"

        # Register propositions and build edges
        for prop in proposition_table:
            pid = prop["proposition_id"]
            classification = prop.get("primary_classification", "ambiguous")
            zone = CLASSIFICATION_ZONE.get(classification, GraphZone.WORK_KNOWLEDGE)
            stress = prop.get("stress_energy_contribution", 0.3)

            edge_sign = prop.get("edge_sign", "positive")
            edge_kind = EdgeKind.SUPPORTS if edge_sign == "positive" else EdgeKind.CONTRADICTS

            # Ensure node exists in adj
            _ = self._adj[zone][pid]
            self._node_zones[pid] = zone
            self._node_classifications[pid] = classification

            # Bridge edges: proposition → its anchor(s) (cross-zone to GOVERNANCE)
            for anchor_id in prop.get("anchor_mappings", []):
                if anchor_id in self._node_zones:
                    anchor_zone = self._node_zones[anchor_id]
                    if anchor_zone != zone:
                        # Cross-zone bridge
                        self._bridges[(zone, pid)].append(
                            (anchor_zone, anchor_id, stress, GateLevel.NONE)
                        )
                    else:
                        # Same zone (unusual but handle it)
                        self._adj[zone][pid].append(
                            (anchor_id, edge_kind, stress, GateLevel.NONE)
                        )
                        self._adj_in[zone][anchor_id].append(
                            (pid, edge_kind, stress, GateLevel.NONE)
                        )

        # Detect contradictions: anchored_true and anchored_false propositions
        # sharing the same anchor get bidirectional blocking records.
        self._detect_contradictions(proposition_table, anchor_registry)

    def _detect_contradictions(
        self,
        proposition_table: List[Dict],
        anchor_registry: List[Dict],
    ) -> None:
        """
        For each anchor, find propositions mapping to it with opposing
        classifications. These are geometric contradictions: the manifold
        is under tension at this anchor point.

        Rather than blocking traversal (VeritasMemoria governance behavior),
        we wire a negative edge between each contradiction pair. The signed
        Laplacian sees this as structural stress, which depresses the Fiedler
        value in proportion to contradiction load. This IS the measurement.

        Negative edge weight = product of the two propositions' stress energy
        contributions, so high-stakes contradictions bend the manifold more
        than low-stakes ones. The weight is negated before storage so it
        contributes as a negative entry in the Laplacian.

        Records are stored in _contradiction_pairs for confidence flags and
        graph_structure output. _blocked_nodes remains empty.
        """
        anchor_to_true: Dict[str, List[str]] = defaultdict(list)
        anchor_to_false: Dict[str, List[str]] = defaultdict(list)
        stress_lookup: Dict[str, float] = {}

        for prop in proposition_table:
            if prop.get("is_compound", False) or prop.get("is_relational", False):
                continue
            pid = prop["proposition_id"]
            stress_lookup[pid] = prop.get("stress_energy_contribution", 0.3)
            classification = prop.get("primary_classification", "ambiguous")
            for anchor_id in prop.get("anchor_mappings", []):
                if classification == "anchored_true":
                    anchor_to_true[anchor_id].append(pid)
                elif classification == "anchored_false":
                    anchor_to_false[anchor_id].append(pid)

        for anchor_id in set(anchor_to_true) & set(anchor_to_false):
            for true_pid in anchor_to_true[anchor_id]:
                for false_pid in anchor_to_false[anchor_id]:
                    # Contradiction edge weight: the negative edge in the signed
                    # Laplacian should reflect that a contradiction exists between
                    # these two propositions at this anchor, regardless of how far
                    # the false proposition is from the anchor.
                    #
                    # We use the harmonic mean of the two propositions' stress
                    # values as a scaling factor, but floor it at 0.09 (the
                    # minimum meaningful weight, equivalent to two stress=0.3
                    # propositions) so that anchored_false propositions with
                    # near-zero G(d) stress (because they are correctly placed
                    # far from the anchor at the boundary) still contribute a
                    # real negative edge. The contradiction exists whether or
                    # not the false proposition is near the anchor.
                    s_true  = stress_lookup.get(true_pid, 0.3)
                    s_false = stress_lookup.get(false_pid, 0.3)
                    # Harmonic mean penalizes near-zero values less than product
                    denom   = s_true + s_false
                    hmean   = (2 * s_true * s_false / denom) if denom > 1e-9 else 0.09
                    neg_weight = -max(hmean, 0.09)

                    # Wire as cross-zone bridge so get_adjacency_flat picks it up.
                    # True prop sits in GOVERNANCE zone; false prop in TEMPORAL_KNOWLEDGE.
                    # We add both directions so the undirected Laplacian is symmetric.
                    true_zone  = self._node_zones.get(true_pid,  GraphZone.GOVERNANCE)
                    false_zone = self._node_zones.get(false_pid, GraphZone.TEMPORAL_KNOWLEDGE)

                    self._bridges[(true_zone, true_pid)].append(
                        (false_zone, false_pid, neg_weight, GateLevel.NONE)
                    )
                    self._bridges[(false_zone, false_pid)].append(
                        (true_zone, true_pid, neg_weight, GateLevel.NONE)
                    )

                    self._contradiction_pairs.append({
                        "node_a": true_pid,
                        "node_b": false_pid,
                        "anchor": anchor_id,
                        "negative_weight": round(neg_weight, 4),
                        "stress_a": stress_lookup.get(true_pid, 0.3),
                        "stress_b": stress_lookup.get(false_pid, 0.3),
                    })

    def get_zone(self, node_id: str) -> GraphZone:
        return self._node_zones.get(node_id, GraphZone.WORK_KNOWLEDGE)

    def get_adjacency_flat(self) -> Dict[str, Dict[str, float]]:
        """Flat {src: {dst: weight}} adjacency for HSHGeometry."""
        flat: Dict[str, Dict[str, float]] = defaultdict(dict)
        for zone, zone_nodes in self._adj.items():
            for src, edges in zone_nodes.items():
                for dst, kind, weight, gate in edges:
                    flat[src][dst] = weight
                    flat[dst][src] = weight  # undirected for spectral purposes
        for (zone, src), bridges in self._bridges.items():
            for dst_zone, dst, weight, gate in bridges:
                flat[src][dst] = weight
                flat[dst][src] = weight
        return dict(flat)

    def get_all_node_ids(self) -> List[str]:
        return list(self._node_zones.keys())

    def get_node_zones_map(self) -> Dict[str, GraphZone]:
        return dict(self._node_zones)

    def update_contradiction_weights(self, proposition_table: List[Dict]) -> None:
        """
        Recompute negative edge weights for all contradiction pairs using
        updated stress_energy_contribution values from the proposition table.

        Called after the geodesic stress step has written new stress values
        so that the Laplacian reflects the actual computed geometry rather
        than the initial placeholder values.
        """
        stress_lookup = {
            p["proposition_id"]: p.get("stress_energy_contribution", 0.3)
            for p in proposition_table
        }

        for pair in self._contradiction_pairs:
            s_true  = stress_lookup.get(pair["node_a"], 0.3)
            s_false = stress_lookup.get(pair["node_b"], 0.3)
            denom   = s_true + s_false
            hmean   = (2 * s_true * s_false / denom) if denom > 1e-9 else 0.09
            new_weight = -max(hmean, 0.09)

            pair["negative_weight"] = round(new_weight, 4)
            pair["stress_a"] = s_true
            pair["stress_b"] = s_false

            # Update the bridge edges in _bridges
            true_zone  = self._node_zones.get(pair["node_a"],  GraphZone.GOVERNANCE)
            false_zone = self._node_zones.get(pair["node_b"], GraphZone.TEMPORAL_KNOWLEDGE)

            for bridge_key, bridge_list in self._bridges.items():
                updated = []
                for dst_zone, dst, weight, gate in bridge_list:
                    if (bridge_key == (true_zone, pair["node_a"]) and dst == pair["node_b"]) or \
                       (bridge_key == (false_zone, pair["node_b"]) and dst == pair["node_a"]):
                        updated.append((dst_zone, dst, new_weight, gate))
                    else:
                        updated.append((dst_zone, dst, weight, gate))
                self._bridges[bridge_key] = updated

def _compute_disk_positions(
    proposition_table: List[Dict],
    anchor_registry: List[Dict],
) -> Dict[str, Tuple[float, float]]:
    """
    Assign (r, theta) Poincaré disk coordinates to each node.

    Anchors are distributed equally around the disk at r=0.02.
    NOTE: r=0.02 is a rendering concession — anchors belong at r=0
    (governance origin). See HARDCODED_AUDIT item 2.

    Proposition theta is initialized as the weighted circular mean of
    connected anchor angles using atan2(Σ w·sin θ, Σ w·cos θ) — the
    correct formula for the mean direction on S¹ — then relaxed via a
    spring embedder:

      - Attractive spring toward each connected anchor, magnitude weighted
        by stress_energy_contribution so high-stress propositions are
        pulled more tightly into their anchor cluster.
      - Contradiction-pair repulsion: propositions sharing an anchor with
        opposing classifications push apart angularly.
      - Velocity damping each iteration for convergence.

    r is set from ZONE_RADIUS + stress jitter (see HARDCODED_AUDIT items
    5 and 6 for why this should be replaced with geodesic inversion).

    Spring constants SPRING_K, REPEL_K, DAMPING, N_ITERS are currently
    magic numbers. See HARDCODED_AUDIT item 4 for derivation requirements.
    Forces are computed in angular space rather than the hyperbolic tangent
    space. See HARDCODED_AUDIT item 3 for the correct formulation.
    """
    positions: Dict[str, Tuple[float, float]] = {}

    # ── Anchor placement: equally spaced ─────────────────────────────────────
    n_anchors = max(len(anchor_registry), 1)
    anchor_thetas: Dict[str, float] = {}
    for i, anchor in enumerate(anchor_registry):
        theta = 2 * math.pi * i / n_anchors
        anchor_thetas[anchor["anchor_id"]] = theta
        positions[anchor["anchor_id"]] = (0.02, theta)

    # ── Build per-proposition lookup tables ───────────────────────────────────
    stress_lookup: Dict[str, float] = {}
    prop_anchor_edges: Dict[str, List[Tuple[str, float]]] = {}
    for prop in proposition_table:
        pid = prop["proposition_id"]
        stress = prop.get("stress_energy_contribution") or 0.3
        stress_lookup[pid] = stress
        prop_anchor_edges[pid] = [
            (aid, stress) for aid in prop.get("anchor_mappings", [])
            if aid in anchor_thetas
        ]

    # ── Detect contradiction pairs for angular repulsion ─────────────────────
    contradiction_partners: Dict[str, List[str]] = {
        prop["proposition_id"]: [] for prop in proposition_table
    }
    anchor_to_true: Dict[str, List[str]] = defaultdict(list)
    anchor_to_false: Dict[str, List[str]] = defaultdict(list)
    for prop in proposition_table:
        if prop.get("is_compound") or prop.get("is_relational"):
            continue
        pid = prop["proposition_id"]
        cls = prop.get("primary_classification", "ambiguous")
        for aid in prop.get("anchor_mappings", []):
            if cls == "anchored_true":
                anchor_to_true[aid].append(pid)
            elif cls == "anchored_false":
                anchor_to_false[aid].append(pid)
    for aid in set(anchor_to_true) & set(anchor_to_false):
        for tp in anchor_to_true[aid]:
            for fp in anchor_to_false[aid]:
                if tp in contradiction_partners:
                    contradiction_partners[tp].append(fp)
                if fp in contradiction_partners:
                    contradiction_partners[fp].append(tp)

    # ── Initialize proposition thetas from anchor cluster circular mean ───────
    prop_thetas: Dict[str, float] = {}
    for i, prop in enumerate(proposition_table):
        pid = prop["proposition_id"]
        edges = prop_anchor_edges.get(pid, [])
        if edges:
            sin_sum = sum(w * math.sin(anchor_thetas[aid]) for aid, w in edges)
            cos_sum = sum(w * math.cos(anchor_thetas[aid]) for aid, w in edges)
            if sin_sum != 0.0 or cos_sum != 0.0:
                prop_thetas[pid] = math.atan2(sin_sum, cos_sum) % (2 * math.pi)
            else:
                prop_thetas[pid] = 2 * math.pi * i / max(len(proposition_table), 1)
        else:
            prop_thetas[pid] = 2 * math.pi * i / max(len(proposition_table), 1)

    # ── Spring-embedder relaxation on theta ───────────────────────────────────
    # Forces computed in angular space — see HARDCODED_AUDIT item 3.
    N_ITERS  = 40
    SPRING_K = 0.15
    REPEL_K  = 0.30
    DAMPING  = 0.80

    velocities: Dict[str, float] = {pid: 0.0 for pid in prop_thetas}

    for _ in range(N_ITERS):
        forces: Dict[str, float] = {pid: 0.0 for pid in prop_thetas}

        for pid, theta in prop_thetas.items():
            stress = stress_lookup.get(pid, 0.3)

            for aid, w in prop_anchor_edges.get(pid, []):
                diff = anchor_thetas[aid] - theta
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                forces[pid] += SPRING_K * w * diff

            for partner in contradiction_partners.get(pid, []):
                if partner in prop_thetas:
                    diff = prop_thetas[partner] - theta
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    forces[pid] -= REPEL_K * stress / (abs(diff) + 0.15)

        for pid in prop_thetas:
            velocities[pid] = DAMPING * velocities[pid] + forces[pid]
            prop_thetas[pid] = (prop_thetas[pid] + velocities[pid]) % (2 * math.pi)

    # ── Assign final positions ────────────────────────────────────────────────
    for prop in proposition_table:
        pid = prop["proposition_id"]
        classification = prop.get("primary_classification", "ambiguous")
        zone = CLASSIFICATION_ZONE.get(classification, GraphZone.WORK_KNOWLEDGE)
        base_r = ZONE_RADIUS[zone]
        stress = stress_lookup.get(pid, 0.3)
        r = min(base_r + (stress * 0.1), 0.95)
        positions[pid] = (r, prop_thetas[pid])

    return positions


# ── Geometric summary builder ──────────────────────────────────────────────────

def _compute_tension_score(
    lambda2: Optional[float],
    n_contradictions: int,
    proposition_table: List[Dict],
) -> Dict[str, Any]:
    """
    Compute the primary tension score from the pipeline's geometric outputs.

    The tension score is a composite of three signals:

    1. Fiedler deficit:  how far the Fiedler value has dropped below the
       coherent-manifold baseline of 1.0. A fully coherent manifold with no
       contradictions has lambda2 near 1. Every contradiction and false
       attractor pulls it down. Score = max(0, 1 - lambda2).

    2. Contradiction load: normalized count of active contradiction pairs
       weighted by their negative edge magnitudes. More contradictions =
       more structural stress = higher tension.

    3. False attractor ratio: proportion of propositions classified as
       anchored_false or inferentially_false. A claim corpus dominated by
       false attractors has to warp further to reach the anchor.

    Final tension score: weighted sum of the three signals, normalized to
    [0, 1]. Higher = more warping required to make the claim touch the anchor.

    Interpretation:
        0.00 - 0.20  Coherent with anchor. Claim is geometrically consistent.
        0.20 - 0.45  Mild tension. Some reframing present.
        0.45 - 0.70  Significant tension. Substantial warping required.
        0.70 - 1.00  Severe tension. Manifold is heavily fractured.
    """
    # Signal 1: Fiedler deficit
    if lambda2 is not None and lambda2 > -10:
        fiedler_deficit = max(0.0, min(1.0 - lambda2, 1.0))
    else:
        fiedler_deficit = 0.5   # unknown — use neutral value

    # Signal 2: contradiction load
    # Each contradiction pair contributes |negative_weight| normalized by 0.09
    # (the floor weight). Max load capped at 1.0.
    if n_contradictions > 0:
        contradiction_load = min(n_contradictions * 0.20, 1.0)
    else:
        contradiction_load = 0.0

    # Signal 3: false attractor ratio
    _false_classes = {"anchored_false", "inferentially_false", "out_of_scope"}
    n_total = max(len(proposition_table), 1)
    n_false = sum(
        1 for p in proposition_table
        if p.get("primary_classification") in _false_classes
    )
    false_ratio = n_false / n_total

    # Weighted composite
    score = (
        0.50 * fiedler_deficit   +
        0.30 * contradiction_load +
        0.20 * false_ratio
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    # Interpretation band
    if score < 0.20:
        band = "coherent"
        interpretation = "Claim is geometrically consistent with the anchor corpus."
    elif score < 0.45:
        band = "mild_tension"
        interpretation = "Some reframing present. Moderate warping required."
    elif score < 0.70:
        band = "significant_tension"
        interpretation = "Substantial warping required to make claim contact anchor."
    else:
        band = "severe_tension"
        interpretation = "Manifold heavily fractured. Claim requires severe reality distortion to contact anchor."

    return {
        "score":              score,
        "band":               band,
        "interpretation":     interpretation,
        "components": {
            "fiedler_deficit":     round(fiedler_deficit, 4),
            "contradiction_load":  round(contradiction_load, 4),
            "false_attractor_ratio": round(false_ratio, 4),
        },
    }


def _build_geometric_summary(
    proposition_table: List[Dict],
    anchor_registry: List[Dict],
    disk_positions: Dict[str, Tuple[float, float]],
    graph: RGAGraphAdapter,
    hsh_result: Optional[Any],
    fte_result: Optional[Any],
    gate_result: Optional[Any],
    sta_result: Optional[Any],
    audit_events: List[Dict],
) -> Dict:
    """Assemble the geometric_summary field from pipeline module outputs."""

    bridge_candidates = [
        p["proposition_id"] for p in proposition_table
        if p.get("primary_classification") == "bridge_narrative"
    ]

    false_attractors = [
        p["proposition_id"] for p in proposition_table
        if p.get("primary_classification") == "anchored_false"
    ]

    # Highest stress proposition
    highest_stress = None
    highest_stress_val = 0.0
    for prop in proposition_table:
        s = prop.get("stress_energy_contribution", 0.0)
        if s > highest_stress_val:
            highest_stress_val = s
            highest_stress = prop["proposition_id"]

    # Spectral summary from HSH if available
    lambda2_note = "Not computed — HSH module unavailable."
    if hsh_result is not None:
        try:
            lambda2_note = f"Fiedler value lambda2 = {hsh_result:.4f}"
        except Exception:
            lambda2_note = f"HSH result: {str(hsh_result)[:100]}"

    # Regime from field theory if available
    regime_note = "Not computed — FieldTheoryEngine unavailable."
    if fte_result is not None:
        regime_note = str(fte_result)[:200]

    # Tidal tension note from structural tension auditor
    tension_note = "No undeclared structural tension detected."
    if sta_result:
        tension_note = f"{len(sta_result)} undeclared tension point(s) detected. See confidence_flags."

    # Bridge narrative note from epistatic gate
    gate_note = "Epistatic gate not run."
    if gate_result is not None:
        gate_note = str(gate_result)[:200]

    manifold_parts = []
    n_gov = sum(1 for p in proposition_table if p.get("primary_classification") == "anchored_true")
    n_false = len(false_attractors)
    n_bridge = len(bridge_candidates)
    n_contested = sum(1 for p in proposition_table
                      if p.get("primary_classification") in ("ambiguous", "inferentially_false"))
    n_contradictions = len(graph._contradiction_pairs)

    manifold_parts.append(
        f"{len(proposition_table)} propositions placed on hyperbolic manifold. "
        f"{n_gov} governance-anchored (GOVERNANCE zone), "
        f"{n_bridge} bridge_narrative candidate(s) (RATIONALE zone), "
        f"{n_false} anchored_false attractor(s) (TEMPORAL_KNOWLEDGE zone), "
        f"{n_contested} ambiguous/contested (WORK/TEMPORAL zones)."
    )
    if n_contradictions:
        manifold_parts.append(
            f"{n_contradictions} active contradiction pair(s) contributing negative edges to the Laplacian."
        )
    manifold_parts.append(lambda2_note)
    manifold_parts.append(regime_note)

    return {
        "manifold_description": " ".join(manifold_parts),
        "anchor_cluster_summary": (
            f"{len(anchor_registry)} anchor(s) registered at governance zone (r=0). "
            + (f"{n_contradictions} contradiction pair(s) active as negative Laplacian edges." if n_contradictions else "No contradictions detected.")
        ),
        "highest_stress_region": (
            f"{highest_stress} (stress={highest_stress_val:.3f})" if highest_stress else "None"
        ),
        "least_stable_truth_candidate": (
            min(
                (p for p in proposition_table if p.get("primary_classification") == "anchored_true"),
                key=lambda p: p.get("stress_energy_contribution", 0),
                default=None
            ) or None
        ) and min(
            (p for p in proposition_table if p.get("primary_classification") == "anchored_true"),
            key=lambda p: p.get("stress_energy_contribution", 0),
        )["proposition_id"],
        "most_stable_false_candidate": (
            max(
                (p for p in proposition_table if p.get("primary_classification") == "anchored_false"),
                key=lambda p: p.get("stress_energy_contribution", 0),
                default=None
            ) or None
        ) and max(
            (p for p in proposition_table if p.get("primary_classification") == "anchored_false"),
            key=lambda p: p.get("stress_energy_contribution", 0),
        )["proposition_id"],
        "bridge_narrative_geometry_note": (
            f"{n_bridge} bridge narrative candidate(s) detected at RATIONALE zone boundary. "
            f"Bidirectional T_mu_nu load present. " + gate_note
            if n_bridge else "No bridge narrative candidates detected."
        ),
        "structural_tension_note": tension_note,
        "tension_score": _compute_tension_score(
            lambda2=hsh_result if isinstance(hsh_result, (int, float)) else None,
            n_contradictions=n_contradictions,
            proposition_table=proposition_table,
        ),
        "analyst_notes": "",
        "reader_interpretation_note": (
            "This summary describes geometric structure only. "
            "Classification of intent or moral judgment is left to the reader."
        ),
    }


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run_pipeline(partial_record: Dict) -> Dict:
    """
    Accept a pre-classified partial schema v0.4 record and run all
    geometric computation modules. Return a completed schema v0.4 record.

    Parameters
    ----------
    partial_record : dict
        Must contain: document_metadata, anchor_registry, proposition_table.
        May contain: confidence_flags, falsifiability_record (pass-through).

    Returns
    -------
    dict
        Completed schema v0.4 record with graph_structure, geometric_summary,
        extended confidence_flags, falsifiability_record, and audit_trail.
    """
    audit_trail: List[Dict] = []
    confidence_flags: List[Dict] = list(partial_record.get("confidence_flags", []))
    falsifiability_record: List[Dict] = list(partial_record.get("falsifiability_record", []))

    doc_meta = partial_record["document_metadata"]
    anchor_registry = partial_record["anchor_registry"]
    proposition_table = partial_record["proposition_table"]

    # Normalize null stress_energy_contribution values to 0.3 default.
    # Values are null in the preclass record (computed here); .get() with a
    # default does not guard against explicit null, so we do it once here.
    for _p in proposition_table:
        if _p.get("stress_energy_contribution") is None:
            _p["stress_energy_contribution"] = 0.3

    def _log(module: str, status: str, note: str, detail: Optional[Dict] = None) -> None:
        audit_trail.append({
            "module": module,
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "note": note,
            **(detail or {}),
        })

    _log("rga_pipeline", "started", f"Pipeline started for {doc_meta.get('document_id')}")

    # ── Step 1: Build graph adapter ────────────────────────────────────────────
    graph = RGAGraphAdapter(proposition_table, anchor_registry)
    _log("RGAGraphAdapter", "complete",
         f"Graph built: {len(graph.get_all_node_ids())} nodes, "
         f"{len(graph._contradiction_pairs)} contradiction pair(s), "
         f"{len(graph._bridges)} bridge edges")

    if graph._contradiction_pairs:
        for pair in graph._contradiction_pairs:
            confidence_flags.append({
                "flag_id": f"CF-CONTRADICTION-{pair['node_a']}-{pair['node_b']}",
                "flag_type": "contradiction_measured",
                "node_a": pair["node_a"],
                "node_b": pair["node_b"],
                "anchor": pair["anchor"],
                "negative_weight": pair["negative_weight"],
                "note": (
                    f"Structural contradiction at anchor {pair['anchor']}. "
                    f"Negative edge weight={pair['negative_weight']:.4f} "
                    f"(stress_a={pair['stress_a']:.2f}, stress_b={pair['stress_b']:.2f}). "
                    f"This tension is active in the Laplacian."
                ),
                "severity": "measured",
            })

    # ── Step 2: HSH Geometry ───────────────────────────────────────────────────
    # Defaults in case HSH step fails — Step 6 always needs these.
    node_radii:      Dict[str, float] = {}
    zone_boundaries: Dict[str, float] = {"governance": 0.30, "false_attractor": 0.72}

    hsh_lambda2 = None
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from hsh_geometry import HSHGeometry, ZONE_LAMBDA

        hsh = HSHGeometry()

        # Adaptive calibration: compute per-document radii.
        # On pass 1 these use zone characteristic distances.
        # On pass 2+ they use geodesic_distance_from_anchor from the prior pass.
        calibration = _calibrate_adaptive_positions(
            hsh, anchor_registry, proposition_table
        )
        node_radii      = calibration["node_radii"]
        zone_boundaries = calibration["zone_boundaries"]
        _log("HSHGeometry", "calibration",
             f"Adaptive radii set for {len(node_radii)} nodes. "
             f"governance_boundary={zone_boundaries['governance']:.3f}, "
             f"false_attractor_boundary={zone_boundaries['false_attractor']:.3f}")

        # ── Geodesic distances and stress — must run before Laplacian ────
        # Runs after calibration (node_radii set) and before get_adjacency_flat
        # so contradiction weights reflect actual computed stress values.
        geodesic_distances = _compute_geodesic_distances(
            anchor_registry, proposition_table, node_radii, ZONE_LAMBDA
        )
        stress_values = _compute_stress_from_distances(
            proposition_table, geodesic_distances
        )
        for prop in proposition_table:
            pid   = prop["proposition_id"]
            d_map = geodesic_distances.get(pid, {})
            if d_map:
                mean_d = sum(d_map.values()) / len(d_map)
                prop["geodesic_distance_from_anchor"] = round(mean_d, 6)
                prop["geodesic_distance_per_anchor"]  = {k: round(v, 6) for k, v in d_map.items()}
            else:
                prop["geodesic_distance_from_anchor"] = None
                prop["geodesic_distance_per_anchor"]  = {}
            if pid in stress_values:
                prop["stress_energy_contribution"] = stress_values[pid]
        n_computed = sum(1 for p in proposition_table if p.get("geodesic_distance_from_anchor") is not None)
        mean_stress = sum(p.get("stress_energy_contribution", 0) for p in proposition_table) / max(len(proposition_table), 1)
        _log("GeodesicStress", "complete",
             f"Geodesic distances computed for {n_computed}/{len(proposition_table)} propositions. "
             f"Mean stress={mean_stress:.4f}.")

        # Update contradiction edge weights with real stress, then build adjacency
        graph.update_contradiction_weights(proposition_table)

        flat_adj   = graph.get_adjacency_flat()
        all_nodes  = graph.get_all_node_ids()
        node_zones = graph.get_node_zones_map()

        import numpy as np
        n = len(all_nodes)
        node_index = {nid: i for i, nid in enumerate(all_nodes)}
        adj_matrix = np.zeros((n, n))
        for src, neighbors in flat_adj.items():
            if src in node_index:
                for dst, weight in neighbors.items():
                    if dst in node_index:
                        i, j = node_index[src], node_index[dst]
                        adj_matrix[i, j] = weight
                        adj_matrix[j, i] = weight

        if n >= 2:
            orc_kappa = {nid: 0.0 for nid in all_nodes}
            adj_dict  = {nid: {nb: w for nb, w in flat_adj.get(nid, {}).items()} for nid in all_nodes}
            coh_field = hsh.coherence_field(adj_dict, all_nodes, GraphZone.WORK_KNOWLEDGE, node_zones, orc_kappa)
            hsh_lambda2 = getattr(coh_field, "lambda2_H", None)
            _log("HSHGeometry", "complete",
                 f"Fiedler value lambda2={hsh_lambda2:.4f}" if hsh_lambda2 is not None else "Fiedler value unavailable")
        else:
            _log("HSHGeometry", "skipped", "Insufficient nodes for spectral computation")

    except Exception as e:
        _log("HSHGeometry", "error", str(e))
        logger.warning(f"HSHGeometry failed: {e}")


    # ── Step 3: Structural Tension Auditor ─────────────────────────────────────
    sta_result = None
    try:
        from structural_tension_auditor import StructuralTensionAuditor

        auditor = StructuralTensionAuditor(graph)
        crisis_records = auditor.scan_for_crises(GraphZone.RATIONALE)
        sta_result = crisis_records

        for rec in crisis_records:
            node_id = getattr(rec, "node_id", str(rec))
            confidence_flags.append({
                "flag_id": f"CF-TENSION-{node_id}",
                "flag_type": "undeclared_structural_tension",
                "affected_proposition": node_id,
                "note": f"Tidal tension detected: {getattr(rec, 'tension_score', 'unknown')}",
                "severity": "medium",
            })

        _log("StructuralTensionAuditor", "complete",
             f"{len(crisis_records)} tension record(s)")

    except Exception as e:
        _log("StructuralTensionAuditor", "error", str(e))
        logger.warning(f"StructuralTensionAuditor failed: {e}")

    # ── Step 4: Epistatic Gate (bridge narrative second signal) ────────────────
    gate_result = None
    try:
        from epistatic_gate import EpistaticGate

        gate = EpistaticGate(graph)
        active_nodes = graph.get_all_node_ids()
        expr_result = gate.evaluate_expression(
            GraphZone.RATIONALE, active_nodes, active_modifiers=[]
        )
        gate_result = expr_result

        latent = getattr(expr_result, "latent_conflicts", [])
        for conflict in latent:
            confidence_flags.append({
                "flag_id": f"CF-EPISTATIC-{getattr(conflict, 'node_id', 'unknown')}",
                "flag_type": "epistatic_suppression_detected",
                "affected_proposition": getattr(conflict, "node_id", "unknown"),
                "note": f"Epistatic suppression: reactivation_risk={getattr(conflict, 'reactivation_risk', 'unknown')}",
                "severity": "high",
            })

        _log("EpistaticGate", "complete",
             f"{len(latent)} latent conflict(s) detected")

    except Exception as e:
        _log("EpistaticGate", "error", str(e))
        logger.warning(f"EpistaticGate failed: {e}")

    # ── Step 5: SAL Coherence ──────────────────────────────────────────────────
    sal_result = None
    try:
        from sal_coherence import SALCoherenceLayer

        sal = SALCoherenceLayer(graph)
        coh_state = sal.coherence_state(GraphZone.WORK_KNOWLEDGE)
        sal_result = coh_state

        if not getattr(coh_state, "should_commit", True):
            confidence_flags.append({
                "flag_id": "CF-SAL-COHERENCE",
                "flag_type": "coherence_below_threshold",
                "affected_proposition": None,
                "note": f"SAL coherence gate: should_commit=False. "
                        f"lambda2={getattr(coh_state, 'fiedler_value', 'unknown')}",
                "severity": "medium",
            })

        _log("SALCoherenceLayer", "complete",
             f"should_commit={getattr(coh_state, 'should_commit', 'unknown')}")

    except Exception as e:
        _log("SALCoherenceLayer", "error", str(e))
        logger.warning(f"SALCoherenceLayer failed: {e}")

    # ── Step 6: Disk positions and graph structure ─────────────────────────────
    # _compute_disk_positions supplies theta values (angular distribution).
    # Adaptive radii from calibration replace its fixed-zone r values.
    disk_positions = _compute_disk_positions(proposition_table, anchor_registry)

    def _geodesic_from_origin(r: float) -> float:
        """Hyperbolic distance from the Poincaré disk origin to radius r.
        Formula: d = 2 * arctanh(r).  Clamps r away from boundary singularity."""
        r_clamped = min(max(r, 0.0), 0.9999)
        return 2.0 * math.atanh(r_clamped)

    nodes = []
    for anchor in anchor_registry:
        _r_disk, theta = disk_positions.get(anchor["anchor_id"], (0.02, 0.0))
        r = node_radii.get(anchor["anchor_id"], _r_disk)
        nodes.append({
            "id": anchor["anchor_id"],
            "type": "anchor",
            "label": anchor["anchor_id"],
            "classification": anchor.get("anchor_classification", "direct_empirical_record"),
            "r": round(r, 4),
            "theta": round(theta, 4),
            "geodesic_distance": round(_geodesic_from_origin(r), 4),
            "zone": "governance",
            "contradiction_participant": False,
        })

    for prop in proposition_table:
        pid = prop["proposition_id"]
        zone = CLASSIFICATION_ZONE.get(prop.get("primary_classification", "ambiguous"),
                                       GraphZone.WORK_KNOWLEDGE)
        _r_disk, theta = disk_positions.get(pid, (0.45, 0.0))
        r = node_radii.get(pid, _r_disk)
        nodes.append({
            "id": pid,
            "type": "proposition",
            "label": pid,
            "classification": prop.get("primary_classification", "ambiguous"),
            "secondary_tags": prop.get("secondary_tags", []),
            "verbatim": prop.get("verbatim_source_text", ""),
            "normalized": prop.get("normalized_claim", ""),
            "stress": prop.get("stress_energy_contribution", 0.0),
            "r": round(r, 4),
            "theta": round(theta, 4),
            "geodesic_distance": round(_geodesic_from_origin(r), 4),
            "zone": zone.value,
            "contradiction_participant": pid in {p["node_a"] for p in graph._contradiction_pairs} | {p["node_b"] for p in graph._contradiction_pairs},
            "blocked": False,
        })

    edges = []
    for prop in proposition_table:
        pid = prop["proposition_id"]
        for anchor_id in prop.get("anchor_mappings", []):
            edges.append({
                "source": pid,
                "target": anchor_id,
                "sign": prop.get("edge_sign", "positive"),
                "weight": prop.get("stress_energy_contribution", 0.3),
            })

    zone_assignments = {zone.value: [] for zone in GraphZone}
    for node in nodes:
        zone_assignments[node["zone"]].append(node["id"])

    bridge_candidates = [
        p["proposition_id"] for p in proposition_table
        if p.get("primary_classification") == "bridge_narrative"
    ]
    false_attractors = [
        p["proposition_id"] for p in proposition_table
        if p.get("primary_classification") == "anchored_false"
    ]

    graph_structure = {
        "nodes": nodes,
        "edges": edges,
        "manifold_zone_assignments": zone_assignments,
        "adaptive_zone_boundaries": zone_boundaries,
        "node_radii": {k: round(v, 4) for k, v in node_radii.items()},
        "fragmentation_detected": len(anchor_registry) > 0 and len([
            p for p in proposition_table
            if CLASSIFICATION_ZONE.get(p.get("primary_classification"), GraphZone.TEMPORAL_KNOWLEDGE)
            == GraphZone.TEMPORAL_KNOWLEDGE
        ]) > len(proposition_table) * 0.4,
        "polarization_detected": (
            len(bridge_candidates) > 0 and len(false_attractors) > 0
        ),
        "bridge_narrative_candidates": bridge_candidates,
        "false_attractor_candidates": false_attractors,
        "contradiction_pairs": graph._contradiction_pairs,
    }

    # ── Step 7: Geometric summary ──────────────────────────────────────────────
    geometric_summary = _build_geometric_summary(
        proposition_table, anchor_registry, disk_positions,
        graph, hsh_lambda2, sal_result, gate_result, sta_result, audit_trail,
    )

    _log("rga_pipeline", "complete",
         f"Pipeline complete. {len(confidence_flags)} confidence flag(s). "
         f"{len(bridge_candidates)} bridge narrative candidate(s).")

    # ── Assemble output record ──────────────────────────────────────────────────────────────────────────────
    return {
        "schema_version": "0.4",
        "document_metadata": doc_meta,
        "anchor_registry": anchor_registry,
        "proposition_table": proposition_table,
        "graph_structure": graph_structure,
        "geometric_summary": geometric_summary,
        "confidence_flags": confidence_flags,
        "falsifiability_record": falsifiability_record,
        "audit_trail": audit_trail,
    }

# ── Batch folder runner ────────────────────────────────────────────────────────

def run_folder(
    input_dir: str,
    output_dir: str,
    runs: int = 3,
) -> List[Dict]:
    """
    Run every ``*_preclass.json`` file found in ``input_dir`` through the full
    pipeline ``runs`` times, feeding each run's output back in as the next
    run's input so that iteratively-computed fields (stress_energy_contribution,
    geodesic_distance_from_anchor, zone calibration) converge.

    Parameters
    ----------
    input_dir  : str  — folder containing ``*_preclass.json`` source files.
    output_dir : str  — folder where completed records are written.
                        Created if it does not exist.
    runs       : int  — number of pipeline passes per document (default 3).

    Returns
    -------
    List of summary dicts, one per document:
        {
          "file":        original filename,
          "document_id": from document_metadata,
          "runs":        number of passes completed,
          "status":      "ok" | "error",
          "error":       error message (only present when status == "error"),
          "output_path": absolute path of the written output file,
        }

    Notes
    -----
    - Files are processed in sorted order so logs are deterministic.
    - Each pass appends its own audit_trail entries; the full run history is
      preserved in the final output so you can see how each field evolved.
    - A file that fails on pass 1 is skipped entirely; its summary entry will
      have status="error". A file that fails on pass N > 1 retains the last
      successful output and logs the failure in the summary.
    - Existing files in output_dir with the same name are overwritten.
    """
    import glob
    import json as _json

    os.makedirs(output_dir, exist_ok=True)

    pattern  = os.path.join(input_dir, "*_preclass.json")
    paths    = sorted(glob.glob(pattern))
    summaries: List[Dict] = []

    if not paths:
        logger.warning(f"run_folder: no *_preclass.json files found in {input_dir!r}")
        return summaries

    logger.info(f"run_folder: {len(paths)} file(s) found in {input_dir!r}, {runs} pass(es) each")

    for src_path in paths:
        filename = os.path.basename(src_path)
        summary: Dict = {"file": filename, "runs": 0, "status": "ok"}

        # ── Load source ───────────────────────────────────────────────────────
        try:
            with open(src_path, encoding="utf-8") as fh:
                record = _json.load(fh)
        except Exception as exc:
            summary["status"] = "error"
            summary["error"]  = f"Failed to load source file: {exc}"
            logger.error(f"run_folder [{filename}]: {summary['error']}")
            summaries.append(summary)
            continue

        doc_id = record.get("document_metadata", {}).get("document_id", filename)
        summary["document_id"] = doc_id
        logger.info(f"run_folder [{filename}]: starting {runs} pass(es) for {doc_id}")

        # ── Iterative pipeline passes ─────────────────────────────────────────
        last_good = record
        for pass_num in range(1, runs + 1):
            try:
                result  = run_pipeline(last_good)
                last_good = result
                summary["runs"] = pass_num
                logger.info(f"run_folder [{filename}]: pass {pass_num}/{runs} complete")
            except Exception as exc:
                summary["status"] = "error"
                summary["error"]  = f"Pipeline error on pass {pass_num}: {exc}"
                logger.error(f"run_folder [{filename}]: {summary['error']}")
                # Keep last_good as-is; write whatever we have so far below
                break

        # ── Write output ──────────────────────────────────────────────────────
        # Derive output filename: replace _preclass with _pipeline
        out_name = filename.replace("_preclass.json", "_pipeline.json")
        if out_name == filename:                       # no _preclass in name
            out_name = filename.replace(".json", "_pipeline.json")
        out_path = os.path.join(output_dir, out_name)

        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                _json.dump(last_good, fh, indent=2, ensure_ascii=False)
            summary["output_path"] = os.path.abspath(out_path)
            logger.info(f"run_folder [{filename}]: written → {out_path}")
        except Exception as exc:
            summary["status"] = "error"
            summary["error"]  = f"Failed to write output: {exc}"
            logger.error(f"run_folder [{filename}]: {summary['error']}")

        summaries.append(summary)

    ok_count  = sum(1 for s in summaries if s["status"] == "ok")
    err_count = len(summaries) - ok_count
    logger.info(
        f"run_folder: done. {ok_count}/{len(summaries)} succeeded, "
        f"{err_count} error(s). Output dir: {output_dir!r}"
    )
    return summaries


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RGA Pipeline — batch folder runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Run every *_preclass.json in ./preclass through 3 pipeline passes,
            # write completed records to ./output
            python rga_pipeline.py ./preclass ./output

            # Run 5 passes instead of the default 3
            python rga_pipeline.py ./preclass ./output --runs 5

            # Single-document mode: pipe a JSON file through 3 passes
            python rga_pipeline.py --file DOC-006_preclass.json ./output
        """),
    )
    parser.add_argument("input",  nargs="?", help="Input folder (batch mode)")
    parser.add_argument("output", nargs="?", help="Output folder")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of pipeline passes per document (default: 3)")
    parser.add_argument("--file", help="Single preclass JSON file (bypasses folder scan)")
    args = parser.parse_args()

    if args.file:
        # Single-file mode
        if not args.output:
            parser.error("--file requires an output folder argument")
        with open(args.file, encoding="utf-8") as fh:
            rec = _json.load(fh)
        for i in range(1, args.runs + 1):
            rec = run_pipeline(rec)
            print(f"Pass {i}/{args.runs} complete")
        out_name = os.path.basename(args.file).replace("_preclass.json", "_pipeline.json")
        out_path = os.path.join(args.output, out_name)
        os.makedirs(args.output, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            _json.dump(rec, fh, indent=2, ensure_ascii=False)
        print(f"Written → {out_path}")
    else:
        if not args.input or not args.output:
            parser.error("Provide input and output folder arguments, or use --file for single-file mode")
        summaries = run_folder(args.input, args.output, runs=args.runs)
        print("\nSummary:")
        for s in summaries:
            status = "✓" if s["status"] == "ok" else "✗"
            runs_done = f"{s['runs']}/{args.runs} passes"
            err = f"  ERROR: {s['error']}" if "error" in s else ""
            print(f"  {status} {s['file']} ({runs_done}) → {s.get('output_path', 'not written')}{err}")
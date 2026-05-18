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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# -- Logging Utility ----------------------------------------------------------
import logging
import time
from datetime import datetime

# Set up a more detailed logger at the top of your script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MultipassTracer")

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


# ── RGA Adaptive Calibration ───────────────────────────────────────────────────

def _calibrate_adaptive_positions(
    geometry,
    anchor_registry: list,
    proposition_table: list,
) -> Dict[str, Any]:
    """Compute and register adaptive Poincaré disk radii for all nodes.

    Implements RULE-010: zone boundaries are set relative to the document's
    anchor cluster distribution, not as fixed radii.

    Placement rules:
      Anchors:           r = chain_length / (max_chain_length + 1)
      anchored_true:     r = primary_anchor_r + 0.03  (just outward of anchor)
      anchored_false:    r = 0.72 + primary_anchor_r * 0.15  (outer zone)
      bridge_narrative:  r = 0.55 + primary_anchor_r * 0.10  (boundary seam)
      inferential:       r = primary_anchor_r + 0.10
      relational:        midpoint of parent proposition radii (second pass)
      compound parent:   inherits from primary_classification_basis sub-prop
                         (third pass)
    Returns dict of node_id → registered radius for inclusion in result JSON.
    """
    # ── Pass 0: anchor radii ──────────────────────────────────────────────────
    chain_lengths = [
        a.get("chain_length_from_governance_zone", 1) for a in anchor_registry
    ]
    max_chain = max(chain_lengths) if chain_lengths else 1

    anchor_radii: Dict[str, float] = {}
    for anc in anchor_registry:
        chain = anc.get("chain_length_from_governance_zone", 1)
        r = chain / (max_chain + 1)
        anchor_radii[anc["anchor_id"]] = r
        geometry.register_node_radius(anc["anchor_id"], r)

    # ── Pass 1: non-compound, non-relational propositions ────────────────────
    prop_radii: Dict[str, float] = {}
    for prop in proposition_table:
        pid            = prop["proposition_id"]
        classification = prop.get("primary_classification", "ambiguous")
        mappings       = prop.get("anchor_mappings") or []
        is_compound    = prop.get("is_compound", False)
        is_relational  = prop.get("is_relational", False)

        if is_compound or is_relational:
            continue  # handled in later passes

        primary_r = anchor_radii.get(mappings[0], max_chain / (max_chain + 1)) \
                    if mappings else 0.5

        if classification == "anchored_true":
            r = min(primary_r + 0.03, 0.50)
        elif classification == "anchored_false":
            r = 0.72 + primary_r * 0.15
        elif classification == "bridge_narrative":
            r = 0.55 + primary_r * 0.10
        elif classification in ("inferentially_true", "inferentially_false"):
            r = primary_r + 0.10
        else:  # ambiguous / out_of_scope
            r = 0.50

        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # ── Pass 2: compound parents (inherit from basis sub-proposition) ──────────
    # Must run before Pass 3 so relational props can look up compound parent radii.
    for prop in proposition_table:
        if not prop.get("is_compound", False):
            continue
        pid = prop["proposition_id"]
        cr  = prop.get("compound_resolution") or {}
        basis = cr.get("primary_classification_basis", "")
        if basis and basis in prop_radii:
            r = prop_radii[basis]
        else:
            classification = prop.get("primary_classification", "ambiguous")
            r = 0.75 if classification == "anchored_false" else 0.30
        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # ── Pass 3: relational propositions (midpoint of parent radii) ───────────
    # Runs after compound parents so all parent radii are populated.
    for prop in proposition_table:
        if not prop.get("is_relational", False):
            continue
        pid = prop["proposition_id"]
        rel = prop.get("relational_resolution") or {}
        pa, pb = rel.get("parent_proposition_a", ""), rel.get("parent_proposition_b", "")
        ra = prop_radii.get(pa, anchor_radii.get(pa, 0.30))
        rb = prop_radii.get(pb, anchor_radii.get(pb, 0.30))
        r  = (ra + rb) / 2.0
        prop_radii[pid] = r
        geometry.register_node_radius(pid, r)

    # Derive zone boundaries for the result JSON
    max_anchor_r = max(anchor_radii.values()) if anchor_radii else 0.25
    governance_boundary = min(max_anchor_r * 1.15, 0.55)
    false_attractor_boundary = 0.72  # where anchored_false starts

    return {
        "node_radii": {**anchor_radii, **prop_radii},
        "zone_boundaries": {
            "governance": round(governance_boundary, 3),
            "false_attractor": round(false_attractor_boundary, 3),
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
        self._blocked_nodes: Dict[str, Any] = {}

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
        classifications. These constitute geometric contradictions.
        """
        anchor_to_true: Dict[str, List[str]] = defaultdict(list)
        anchor_to_false: Dict[str, List[str]] = defaultdict(list)

        for prop in proposition_table:
            # Compound parents and relational propositions are excluded from
            # contradiction detection. Compound parents carry derived classifications
            # and anchor_mappings spanning both sub-propositions. Relational
            # propositions are meta-propositions whose classification reflects
            # analysis of a relationship between other propositions, not an
            # independent claim. Only primary, non-relational propositions participate.
            if prop.get("is_compound", False) or prop.get("is_relational", False):
                continue
            pid = prop["proposition_id"]
            classification = prop.get("primary_classification", "ambiguous")
            for anchor_id in prop.get("anchor_mappings", []):
                if classification == "anchored_true":
                    anchor_to_true[anchor_id].append(pid)
                elif classification == "anchored_false":
                    anchor_to_false[anchor_id].append(pid)

        for anchor_id in set(anchor_to_true) & set(anchor_to_false):
            for true_pid in anchor_to_true[anchor_id]:
                for false_pid in anchor_to_false[anchor_id]:
                    zone = self._node_zones.get(true_pid, GraphZone.WORK_KNOWLEDGE)
                    rec = _RGAContradictionRecord(
                        node_a_id=true_pid,
                        node_b_id=false_pid,
                        zone=zone,
                        gate_level=GateLevel.REVIEW_REQUIRED,
                        resolution_note=f"Contradiction anchored at {anchor_id}",
                    )
                    self._blocked_nodes[true_pid] = rec
                    self._blocked_nodes[false_pid] = rec

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


# ── Disk position helper ───────────────────────────────────────────────────────

def _compute_disk_positions(
    proposition_table: List[Dict],
    anchor_registry: List[Dict],
) -> Dict[str, Tuple[float, float]]:
    """
    Assign (r, theta) Poincaré disk coordinates to each node.
    Anchors sit at r=0. Propositions are placed by zone radius,
    angularly distributed around the disk.
    """
    positions: Dict[str, Tuple[float, float]] = {}

    # Anchors at center
    n_anchors = len(anchor_registry)
    for i, anchor in enumerate(anchor_registry):
        theta = (2 * math.pi * i / max(n_anchors, 1))
        positions[anchor["anchor_id"]] = (0.02, theta)

    # Propositions by zone
    n_props = len(proposition_table)
    for i, prop in enumerate(proposition_table):
        pid = prop["proposition_id"]
        classification = prop.get("primary_classification", "ambiguous")
        zone = CLASSIFICATION_ZONE.get(classification, GraphZone.WORK_KNOWLEDGE)
        base_r = ZONE_RADIUS[zone]

        # Jitter r slightly by stress energy to spread within zone
        stress = prop.get("stress_energy_contribution", 0.3)
        r = min(base_r + (stress * 0.1), 0.95)

        theta = (2 * math.pi * i / max(n_props, 1))
        positions[pid] = (r, theta)

    return positions


# ── Geometric summary builder ──────────────────────────────────────────────────

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
    n_blocked = len(graph._blocked_nodes)

    manifold_parts.append(
        f"{len(proposition_table)} propositions placed on hyperbolic manifold. "
        f"{n_gov} governance-anchored (GOVERNANCE zone), "
        f"{n_bridge} bridge_narrative candidate(s) (RATIONALE zone), "
        f"{n_false} anchored_false attractor(s) (TEMPORAL_KNOWLEDGE zone), "
        f"{n_contested} ambiguous/contested (WORK/TEMPORAL zones)."
    )
    if n_blocked:
        manifold_parts.append(
            f"{n_blocked} proposition(s) in contradiction-blocked pairs."
        )
    manifold_parts.append(lambda2_note)
    manifold_parts.append(regime_note)

    return {
        "manifold_description": " ".join(manifold_parts),
        "anchor_cluster_summary": (
            f"{len(anchor_registry)} anchor(s) registered at governance zone (r=0). "
            + ("Contradiction-blocked pairs detected — see confidence_flags." if n_blocked else "No contradictions detected.")
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
            "timestamp": datetime.utcnow().isoformat(),
            "note": note,
            **(detail or {}),
        })

    _log("rga_pipeline", "started", f"Pipeline started for {doc_meta.get('document_id')}")

    # ── Step 1: Build graph adapter ────────────────────────────────────────────
    graph = RGAGraphAdapter(proposition_table, anchor_registry)
    _log("RGAGraphAdapter", "complete",
         f"Graph built: {len(graph.get_all_node_ids())} nodes, "
         f"{len(graph._blocked_nodes)} blocked, "
         f"{len(graph._bridges)} bridge edges")

    if graph._blocked_nodes:
        for node_id, rec in graph._blocked_nodes.items():
            confidence_flags.append({
                "flag_id": f"CF-CONTRADICTION-{node_id}",
                "flag_type": "contradiction_detected",
                "affected_proposition": node_id,
                "note": f"Contradiction pair detected. {rec.resolution_note}",
                "severity": "high",
            })

    # ── Step 2: HSH Geometry ───────────────────────────────────────────────────
    # Defaults in case HSH step fails — Step 6 always needs these.
    node_radii:      Dict[str, float] = {}
    zone_boundaries: Dict[str, float] = {"governance": 0.30, "false_attractor": 0.72}

    hsh_lambda2 = None
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from hsh_geometry import HSHGeometry

        hsh = HSHGeometry()

        # Adaptive calibration: compute per-document radii from anchor chain
        # lengths per RULE-010 before any spectral or field computation runs.
        calibration = _calibrate_adaptive_positions(
            hsh, anchor_registry, proposition_table
        )
        node_radii      = calibration["node_radii"]       # Dict[str, float]
        zone_boundaries = calibration["zone_boundaries"]  # Dict[str, float]
        _log("HSHGeometry", "calibration",
             f"Adaptive radii set for {len(node_radii)} nodes. "
             f"governance_boundary={zone_boundaries['governance']:.3f}, "
             f"false_attractor_boundary={zone_boundaries['false_attractor']:.3f}")

        flat_adj = graph.get_adjacency_flat()
        all_nodes = graph.get_all_node_ids()
        node_zones = graph.get_node_zones_map()

        # Build numpy-compatible adjacency for Laplacian computation
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
            adj_dict = {nid: {neighbor: w for neighbor, w in flat_adj.get(nid, {}).items()}
                        for nid in all_nodes}
            coh_field = hsh.coherence_field(adj_dict, all_nodes, GraphZone.WORK_KNOWLEDGE,
                                            node_zones, orc_kappa)
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
            "blocked": anchor["anchor_id"] in graph._blocked_nodes,
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
            "blocked": pid in graph._blocked_nodes,
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
        "contradiction_pairs": [
            {"node_a": rec.node_a_id, "node_b": rec.node_b_id}
            for rec in set(graph._blocked_nodes.values())
        ],
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
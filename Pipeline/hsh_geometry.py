"""
VeritasMemoria - HSH Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperbolic Spectral Homeostasis (HSH) geometry layer.

Replaces the flat Euclidean belief graph with a Hyperbolic Simplicial Complex
embedded in a Poincare disk. Distance between nodes is governed by curvature;
the "truth" of the system is governed by spectral stability of that curved space.

Theory
------
In HSH, every zone has a Zone Inertia scalar Lambda that determines its radial
position in the unit Poincare disk. Higher Lambda = closer to the center =
closer to the governance anchor = more authoritative and stable.

    Zone              Lambda    Disk radius (r = 1 - Lambda)
    GOVERNANCE         1.0      r=0.00  (at the center)
    RATIONALE          0.65     r=0.35
    WORK_KNOWLEDGE     0.55     r=0.45
    TEMPORAL_KNOWLEDGE 0.35     r=0.65  (near the boundary)

The HSH Metric Tensor
---------------------
Distance between any two nodes u and v uses a modified Poincare metric that
incorporates Zone Inertia Lambda:

    d_HSH(u, v) = arcosh(1 + |u - v|^2 / ((1 - |u|^2/Lambda_u)(1 - |v|^2/Lambda_v)))

This forces Work nodes to curve through Governance before connecting to each
other, structurally preventing policy violations in reasoning chains.

The Curvature-Adaptive Laplacian
---------------------------------
Standard signed normalized Laplacian L_s is replaced by a Curvature-Adaptive
Laplacian L_H where each edge weight is scaled by the Gaussian curvature K:

    w_H(u, v) = w(u, v) * K(u, v)

Where K(u, v) is the sectional curvature of the edge in hyperbolic space.
Flatter regions (K near 0) have their weights reduced, effectively downweighting
reasoning chains that lack hierarchical depth.

The HSH Coherence Field
------------------------
Stopping criterion for memory commit under HSH:

    Phi_H(Z) = lambda_2(L_H) + integral(K(x) dA) >= lambda_min

Where the Gaussian Curvature integral adds a geometric penalty term. A
"hallucination" (content with no hierarchical grounding) has K near negative
infinity at the disk boundary, making Phi_H fail the threshold even when
lambda_2 of the flat graph might superficially pass.

Integration
-----------
    from veritas_memoria.analysis.coherence.hsh_geometry import (
        HSHGeometry, ZONE_LAMBDA, hsh_distance, curvature_adaptive_weight
    )

    geom = HSHGeometry()

    # Get the disk position for a node in a zone
    pos = geom.node_position("mem_123", GraphZone.WORK_KNOWLEDGE)

    # Compute HSH distance between two nodes
    d = geom.distance("mem_abc", GraphZone.WORK_KNOWLEDGE,
                      "mem_def", GraphZone.GOVERNANCE)

    # Build curvature-adaptive Laplacian from adjacency
    L_H = geom.curvature_adaptive_laplacian(adj, nodes)

    # Compute full HSH coherence field Phi_H
    phi = geom.coherence_field(adj, nodes)
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph_types import GraphZone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Zone Lambda constants (Zone Inertia)
# ─────────────────────────────────────────────────────────────

ZONE_LAMBDA: Dict[GraphZone, float] = {
    GraphZone.GOVERNANCE:         1.0,    # gravitational anchor — center of disk
    GraphZone.RATIONALE:          0.65,   # reasoning artifacts and knowledge corpus
    GraphZone.WORK_KNOWLEDGE:     0.55,   # matter-scoped persistent facts
    GraphZone.TEMPORAL_KNOWLEDGE: 0.35,   # ephemeral — near the disk boundary
}

# Radial position in the Poincare disk: r = 1 - Lambda
# Higher Lambda -> closer to center -> more authoritative
ZONE_RADIUS: Dict[GraphZone, float] = {
    zone: 1.0 - lam for zone, lam in ZONE_LAMBDA.items()
}

# Gaussian curvature K(r) = -4 / (1-r²)² evaluated at each zone's
# characteristic disk radius.  Precomputed here so displacement_signal()
# and directional_curvature_factor() run without touching the eigensolver.
#
#   GOVERNANCE:         r=0.00  K = -4.000
#   RATIONALE:          r=0.35  K ≈ -5.194
#   WORK_KNOWLEDGE:     r=0.45  K ≈ -6.289
#   TEMPORAL_KNOWLEDGE: r=0.65  K ≈ -11.988
ZONE_CURVATURE: Dict[GraphZone, float] = {
    zone: -4.0 / max((1.0 - r * r) ** 2, 1e-9)
    for zone, r in ZONE_RADIUS.items()
}

# String-keyed versions for callers that pass zone values (e.g. from DB)
ZONE_LAMBDA_STR: Dict[str, float] = {z.value: lam for z, lam in ZONE_LAMBDA.items()}
ZONE_RADIUS_STR: Dict[str, float] = {z.value: r   for z, r   in ZONE_RADIUS.items()}

# Minimum Phi_H to commit (same semantic as lambda_min in SALCoherenceLayer)
DEFAULT_PHI_MIN: float = 0.05

# Per-seam fragile-bridge thresholds for D(u) under the directional curvature
# formula.  Calibrated from simulation failure modes; values differ because
# the curvature gradient amplifies raw pressure differently at each seam.
#
#   GOV->RATIONALE:   small differential (~1.19) but critical seam — keep tight
#   RATIONALE->WORK:  moderate differential (~1.10) — standard tolerance
#   WORK->TEMPORAL:   large differential (~5.70) — high tolerance to avoid noise
SEAM_THRESHOLD_GOV_RATIONALE:  float = 0.9
SEAM_THRESHOLD_RATIONALE_WORK: float = 1.1
SEAM_THRESHOLD_WORK_TEMPORAL:  float = 1.5
DEFAULT_FRAGILE_BRIDGE_THRESHOLD: float = 1.0  # fallback for unknown / no seam

# M-05: maximum number of entries in the position cache. Positions are
# recomputed deterministically in O(1), so evicting old entries is safe.
_POSITION_CACHE_MAX_SIZE: int = 10_000

# Gaussian curvature of the hyperbolic plane is constant -1 (Poincare disk model).
# This value is used when a local curvature estimate is not available.
HYPERBOLIC_CURVATURE: float = -1.0


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class NodePosition:
    """
    Position of a node in the Poincare disk.

    coords: 2D coordinates in the unit disk (|coords| < 1)
    zone: the zone this node belongs to
    lambda_val: the zone inertia for this node's zone
    radius: |coords| — distance from origin (0 = center, 1 = boundary)
    """
    node_id:    str
    zone:       GraphZone
    lambda_val: float
    coords:     np.ndarray     # shape (2,), |coords| < 1
    radius:     float          # |coords|


@dataclass
class HSHCoherenceField:
    """
    Full HSH Coherence Field result for a zone.

    phi_H: the combined stopping criterion value
    lambda2_H: Fiedler value of the Curvature-Adaptive Laplacian
    curvature_integral: integral of Gaussian Curvature over the zone
    mean_curvature: mean K(x) across all nodes
    passes_threshold: whether phi_H >= phi_min
    hallucination_risk: fraction of nodes with extreme negative curvature
                        (K < -2.0), indicating boundary proximity
    node_curvatures: per-node Gaussian Curvature estimate
    fragile_bridge_nodes: nodes with D(u) > 1.0 at the governance->rationale
                          seam, sorted descending (node_id, D_value). Empty
                          when no cross-zone adjacency is present.
    """
    zone:               str
    phi_H:              float
    lambda2_H:          float
    curvature_integral: float
    mean_curvature:     float
    passes_threshold:   bool
    phi_min:            float
    hallucination_risk: float
    node_curvatures:    Dict[str, float]              = field(default_factory=dict)
    fragile_bridge_nodes: List[Tuple[str, float]]     = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Core functions (stateless)
# ─────────────────────────────────────────────────────────────

def hsh_distance(
    pos_u: np.ndarray,
    lambda_u: float,
    pos_v: np.ndarray,
    lambda_v: float,
    eps: float = 1e-9,
) -> float:
    """
    Compute the HSH metric distance between two nodes in the Poincare disk.

    d_HSH(u, v) = arcosh(1 + |u - v|^2 / ((1 - |u|^2/Lambda_u)(1 - |v|^2/Lambda_v)))

    The Zone Inertia terms in the denominator cause nodes near the boundary
    (low Lambda) to be geometrically further from the center anchor than their
    Euclidean position suggests. A hallucination node at |pos|=0.85 with
    Lambda=0.35 has effectively infinite distance to the governance anchor.

    Parameters
    ----------
    pos_u, pos_v : np.ndarray shape (2,)
        Coordinates in the unit Poincare disk.
    lambda_u, lambda_v : float
        Zone Inertia for each node's zone (from ZONE_LAMBDA).
    eps : float
        Small epsilon to prevent division by zero near the disk boundary.

    Returns
    -------
    float
        HSH distance (>= 0, can be very large near boundary).
    """
    diff_sq = float(np.sum((pos_u - pos_v) ** 2))
    denom_u = max(1.0 - float(np.sum(pos_u ** 2)) / lambda_u, eps)
    denom_v = max(1.0 - float(np.sum(pos_v ** 2)) / lambda_v, eps)
    arg = 1.0 + diff_sq / (denom_u * denom_v)
    # Clamp arg >= 1 to keep arcosh real (floating point noise can push below 1)
    return math.acosh(max(arg, 1.0))


def gaussian_curvature_at_radius(r: float, eps: float = 1e-9) -> float:
    """
    Gaussian curvature of the Poincare disk at radial distance r from origin.

    In the standard hyperbolic plane (curvature -1), the metric blows up near
    the boundary. We use the conformal factor formula:

        K(r) = -4 / (1 - r^2)^2

    At r=0 (governance anchor): K = -4
    At r=0.35 (domain knowledge): K ~ -4.8
    At r=0.45 (work knowledge): K ~ -5.3
    At r=0.65 (temporal knowledge): K ~ -8.2
    As r -> 1 (disk boundary): K -> -infinity

    The geometry enforces that hallucinations (content without hierarchical
    grounding that would sit near r=1) have K approaching negative infinity.
    """
    denom = max((1.0 - r * r) ** 2, eps)
    return -4.0 / denom


def directional_curvature_factor(zone_src: GraphZone, zone_dst: GraphZone) -> float:
    """
    Directional curvature factor |K(zone_src) - K(zone_dst)| for a cross-zone edge.

    Weights the D(u) pressure signal by how steep the curvature gradient is
    at the zone boundary being crossed.  This makes WORK->TEMPORAL transitions
    (differential ~5.70) much more alarming per unit of edge tension than
    GOVERNANCE->RATIONALE transitions (differential ~1.19), which matches the
    actual geometric risk encoded in the Poincare disk model.

    The factor is symmetric: factor(u, v) == factor(v, u).
    Same-zone crossings return 0.0 (no directional gradient).

    Parameters
    ----------
    zone_src, zone_dst : GraphZone
        The two zones connected by the edge.

    Returns
    -------
    float
        |K(zone_src) - K(zone_dst)| in [0, inf).
    """
    if zone_src == zone_dst:
        return 0.0
    k_src = ZONE_CURVATURE.get(zone_src, HYPERBOLIC_CURVATURE)
    k_dst = ZONE_CURVATURE.get(zone_dst, HYPERBOLIC_CURVATURE)
    return abs(k_src - k_dst)


def seam_threshold(seam_zones: Optional[Tuple[GraphZone, GraphZone]]) -> float:
    """
    Return the calibrated fragile-bridge threshold for a given zone seam.

    Uses per-seam constants (SEAM_THRESHOLD_*) derived from simulation failure
    modes.  Falls back to DEFAULT_FRAGILE_BRIDGE_THRESHOLD for unknown or
    absent seam pairs.

    Parameters
    ----------
    seam_zones : Tuple[GraphZone, GraphZone] or None
        The zone pair whose threshold to look up.  Order is irrelevant.

    Returns
    -------
    float
        Displacement threshold above which a node is a fragile bridge.
    """
    if seam_zones is None:
        return DEFAULT_FRAGILE_BRIDGE_THRESHOLD
    pair = frozenset(seam_zones)
    if pair == frozenset({GraphZone.GOVERNANCE, GraphZone.RATIONALE}):
        return SEAM_THRESHOLD_GOV_RATIONALE
    if pair == frozenset({GraphZone.RATIONALE, GraphZone.WORK_KNOWLEDGE}):
        return SEAM_THRESHOLD_RATIONALE_WORK
    if pair == frozenset({GraphZone.WORK_KNOWLEDGE, GraphZone.TEMPORAL_KNOWLEDGE}):
        return SEAM_THRESHOLD_WORK_TEMPORAL
    return DEFAULT_FRAGILE_BRIDGE_THRESHOLD


def curvature_adaptive_weight(w: float, k: float) -> float:
    """
    Scale an edge weight by local Gaussian Curvature to produce a
    Curvature-Adaptive edge weight for the HSH Laplacian.

        w_H(u, v) = w(u, v) * |K(midpoint)|^{-1/2}

    Flat regions (|K| near 0 — not possible in hyperbolic space, but
    approximated for shallow connections) get downweighted. Deep hierarchical
    connections in the high-curvature region near the center get upweighted.

    Since K is always negative in hyperbolic space, we use |K|.
    """
    abs_k = max(abs(k), 1e-6)
    return w * (abs_k ** -0.5)


# ─────────────────────────────────────────────────────────────
# HSH Geometry engine
# ─────────────────────────────────────────────────────────────

class HSHGeometry:
    """
    Hyperbolic Spectral Homeostasis geometry engine.

    Manages node positions in the Poincare disk and provides:
    - Deterministic position assignment from zone + node_id hash
    - HSH pairwise distances
    - Curvature-Adaptive Laplacian construction
    - HSH Coherence Field Phi_H computation

    Node positions within a zone are distributed uniformly by angle on the
    zone's characteristic radius ring. The angle is derived from a hash of
    the node_id so positions are deterministic without requiring storage.
    """

    def __init__(self, phi_min: float = DEFAULT_PHI_MIN):
        self.phi_min = phi_min
        # M-05: bounded FIFO cache.  OrderedDict gives O(1) oldest-entry eviction.
        # Positions are fully deterministic so eviction is always safe.
        self._position_cache: collections.OrderedDict = collections.OrderedDict()
        # RGA adaptive mode: per-node radii computed from anchor chain lengths.
        # When populated, node_position() uses these instead of the zone constants.
        # Keyed by node_id. Populated by register_node_radius() from rga_pipeline.
        self._rga_radii: Dict[str, float] = {}

    def register_node_radius(self, node_id: str, radius: float) -> None:
        """Register an RGA-computed adaptive radius for a specific node.

        When set, overrides the zone's characteristic radius for that node.
        Called by rga_pipeline during adaptive calibration so that per-document
        anchor chain lengths govern disk position rather than fixed zone constants.
        Per RULE-010: zone boundaries are relative to the document's anchor
        cluster distribution, not fixed radii.

        Invalidates any cached position for this node so the next call to
        node_position() recomputes with the new radius.
        """
        self._rga_radii[node_id] = max(0.0, min(radius, 0.999))
        # Invalidate all cache entries for this node (zone key varies)
        stale = [k for k in self._position_cache if k.endswith(f":{node_id}")]
        for k in stale:
            del self._position_cache[k]

    # ── Position assignment ──────────────────────────────────

    def node_position(self, node_id: str, zone: GraphZone) -> NodePosition:
        """
        Get (or compute) the Poincare disk position for a node.

        Position is fully deterministic from (node_id, zone). No storage needed.
        The node sits on the characteristic radius ring for its zone, at an
        angle derived from a hash of node_id.
        """
        # Coerce string → enum so callers can pass either form
        if not isinstance(zone, GraphZone):
            zone = GraphZone(str(zone))
        cache_key = f"{zone.value}:{node_id}"
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        # RGA adaptive mode: use per-document chain-length-derived radius if
        # registered. This satisfies RULE-010 (zone boundaries relative to
        # anchor cluster distribution). Falls back to fixed zone constants for
        # standard VeritasMemoria use.
        if node_id in self._rga_radii:
            r   = self._rga_radii[node_id]
            lam = max(1.0 - r, 1e-6)
        else:
            lam = ZONE_LAMBDA[zone]
            r   = ZONE_RADIUS[zone]
            # Governance zone sits at r=0 (disk center, the governance anchor).
            # All governance nodes would otherwise map to coords=[0,0], making
            # pairwise distances zero and the inertia-weighted Planck scale
            # degenerate. Per Templet (2026) §3.4 and VM Spec §8, a small offset
            # δ<<1 is applied so each governance node has a distinct, well-defined
            # position. δ is derived from the same hash as the angle, ensuring
            # determinism. This does not affect the architecture's functional
            # properties — governance nodes remain fixed during gradient flow.
            if zone == GraphZone.GOVERNANCE:
                h2 = int(hashlib.sha256(node_id.encode()).hexdigest()[8:16], 16)
                delta = 1e-4 + (h2 / 0xFFFFFFFF) * 1e-3   # δ in [1e-4, 1.1e-3]
                r = delta

        # Deterministic angle from node_id hash — uniform distribution on ring
        h = int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16)
        angle = (h / 0xFFFFFFFF) * 2.0 * math.pi

        coords = np.array([r * math.cos(angle), r * math.sin(angle)])
        pos = NodePosition(
            node_id=node_id, zone=zone, lambda_val=lam,
            coords=coords, radius=float(np.linalg.norm(coords)),
        )
        # M-05: evict the oldest entry before inserting to stay within the cap
        if len(self._position_cache) >= _POSITION_CACHE_MAX_SIZE:
            self._position_cache.popitem(last=False)
        self._position_cache[cache_key] = pos
        return pos

    def distance(
        self,
        node_a: str, zone_a: GraphZone,
        node_b: str, zone_b: GraphZone,
    ) -> float:
        """HSH metric distance between two nodes."""
        pos_a = self.node_position(node_a, zone_a)
        pos_b = self.node_position(node_b, zone_b)
        return hsh_distance(pos_a.coords, pos_a.lambda_val,
                            pos_b.coords, pos_b.lambda_val)

    # ── Curvature-Adaptive Laplacian ─────────────────────────

    def curvature_adaptive_laplacian(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        node_zones: Optional[Dict[str, GraphZone]] = None,
        default_zone: GraphZone = GraphZone.WORK_KNOWLEDGE,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Build the Curvature-Adaptive Laplacian L_H from a weighted adjacency.

        Each edge weight w(u,v) is scaled by the local curvature at the
        midpoint between u and v:

            w_H(u, v) = w(u, v) * |K(midpoint_radius)|^{-1/2}

        Returns
        -------
        L_H : np.ndarray shape (n, n)
            Normalized Curvature-Adaptive Laplacian.
        node_curvatures : Dict[str, float]
            Per-node Gaussian Curvature K estimate (mean over incident edges).
        """
        n = len(nodes)
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        A = np.zeros((n, n))
        node_curvatures: Dict[str, float] = {}
        curvature_samples: Dict[str, List[float]] = {nid: [] for nid in nodes}

        for u in nodes:
            neighbors = adj.get(u, {})
            zone_u = (node_zones or {}).get(u, default_zone)
            pos_u = self.node_position(u, zone_u)
            k_u = gaussian_curvature_at_radius(pos_u.radius)

            for v, w in neighbors.items():
                if v not in node_idx:
                    continue
                zone_v = (node_zones or {}).get(v, default_zone)
                pos_v = self.node_position(v, zone_v)
                k_v = gaussian_curvature_at_radius(pos_v.radius)

                # Curvature at edge midpoint
                mid_r = (pos_u.radius + pos_v.radius) / 2.0
                k_mid = gaussian_curvature_at_radius(mid_r)

                w_H = curvature_adaptive_weight(abs(w), k_mid)
                # Preserve sign from original weight (negative for contradictions)
                if w < 0:
                    w_H = -w_H

                i, j = node_idx[u], node_idx[v]
                A[i][j] = w_H
                curvature_samples[u].append(k_u)
                curvature_samples[v].append(k_v)

        # Per-node mean curvature
        for nid in nodes:
            samples = curvature_samples[nid]
            node_curvatures[nid] = (
                sum(samples) / len(samples) if samples
                else gaussian_curvature_at_radius(
                    ZONE_RADIUS.get((node_zones or {}).get(nid, default_zone), 0.45)
                )
            )

        # Normalized Curvature-Adaptive Laplacian
        # L_H = D_|A|^{-1/2} (D_|A| - A) D_|A|^{-1/2}
        abs_row_sums = np.abs(A).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(abs_row_sums > 0, 1.0 / np.sqrt(abs_row_sums), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        D_abs = np.diag(abs_row_sums)
        L_H = D_inv_sqrt @ (D_abs - A) @ D_inv_sqrt

        return L_H, node_curvatures

    # ── HSH Coherence Field ──────────────────────────────────

    def coherence_field(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        zone: GraphZone = GraphZone.WORK_KNOWLEDGE,
        node_zones: Optional[Dict[str, GraphZone]] = None,
        orc_kappa: Optional[Dict[str, float]] = None,
    ) -> HSHCoherenceField:
        """
        Compute the full HSH Coherence Field Phi_H for a zone.

        Phi_H(Z) = lambda_2(L_H) + integral(K(x) dA) >= phi_min

        The curvature integral adds a geometric penalty. When ORC kappa
        values are supplied (from VeritasGeometria), the integral uses
        topology-dependent Ollivier-Ricci curvature instead of the
        zone-constant Gaussian curvature. This allows the field to vary
        with actual graph structure rather than being fixed by zone radius.

        Parameters
        ----------
        adj : Dict[str, Dict[str, float]]
            Weighted adjacency {src: {dst: weight}}.
        nodes : List[str]
            Nodes to include.
        zone : GraphZone
            Zone being evaluated (used as default for unspecified nodes).
        node_zones : Dict[str, GraphZone], optional
            Per-node zone override. If None, all nodes use `zone`.
        orc_kappa : Dict[str, float], optional
            Per-node mean Ollivier-Ricci curvature from VeritasGeometria.
            When provided, overrides the Gaussian curvature fallback for
            the curvature_integral and hallucination_risk calculations.

        Returns
        -------
        HSHCoherenceField
        """
        # Coerce string → enum so callers can pass either form
        if not isinstance(zone, GraphZone):
            zone = GraphZone(str(zone))
        zone_val = zone.value
        n = len(nodes)

        if n < 3:
            return HSHCoherenceField(
                zone=zone_val, phi_H=0.0, lambda2_H=0.0,
                curvature_integral=0.0, mean_curvature=0.0,
                passes_threshold=False, phi_min=self.phi_min,
                hallucination_risk=0.0,
            )

        L_H, node_curvatures = self.curvature_adaptive_laplacian(
            adj, nodes, node_zones=node_zones, default_zone=zone
        )

        # Fiedler value of the curvature-adaptive Laplacian
        try:
            try:
                from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
                from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]
                sp = csr_matrix(L_H)
                k = min(3, n - 1)
                vals = eigsh(sp, k=k, which="SM", return_eigenvectors=False)
                vals = np.sort(np.real(vals))
                lambda2_H = float(vals[1]) if len(vals) > 1 else float(vals[0])
            except Exception:
                vals = np.linalg.eigvalsh(L_H)
                vals = np.sort(np.real(vals))
                lambda2_H = float(vals[1]) if len(vals) > 1 else float(vals[0])
        except Exception as e:
            logger.warning("HSHGeometry: eigensolver failed: %s", e)
            lambda2_H = 0.0

        # Curvature integral: topology-dependent when ORC kappa is available,
        # otherwise fall back to Gaussian curvature derived from node positions.
        k_values = list(node_curvatures.values())
        mean_k = float(np.mean(k_values)) if k_values else HYPERBOLIC_CURVATURE

        if orc_kappa:
            # ORC kappa path: per-node Ollivier-Ricci curvature from VeritasGeometria.
            # These values respond to actual edge topology (W1 transport distance)
            # rather than being frozen at the zone-constant Gaussian curvature.
            # kappa in [-1, 1]: positive = locally tree-like / well-connected hub,
            # negative = bottleneck or sparse bridge. Use mean directly as the
            # curvature contribution to Phi_H.
            orc_vals = [orc_kappa.get(node_id, 0.0) for node_id in nodes]
            mean_orc = float(np.mean(orc_vals)) if orc_vals else 0.0
            curvature_integral = mean_orc
            # Hallucination risk: fraction of nodes with strongly negative ORC kappa.
            # kappa < -0.2 is the standard bottleneck / disconnection threshold.
            hallucination_risk = sum(1 for v in orc_vals if v < -0.2) / max(n, 1)
        else:
            # Gaussian curvature fallback.
            # Fix (was -mean_k/4.0 which grew toward boundary — wrong direction):
            # -4.0/mean_k is a decreasing function toward the disk boundary.
            # mean_k is always negative, so -4/mean_k > 0; as r→1, mean_k→-inf
            # and -4/mean_k→0, meaning boundary nodes contribute less — correct.
            curvature_integral = -4.0 / mean_k if mean_k != 0.0 else 0.0
            extreme_threshold = -8.0  # K < -8 corresponds to r > ~0.71, near boundary
            hallucination_risk = sum(1 for k in k_values if k < extreme_threshold) / max(n, 1)

        phi_H = lambda2_H + curvature_integral

        # NOTE: fragile_bridge_nodes is intentionally empty here.
        # coherence_field() only receives a single-zone adjacency, so
        # displacement_signal() would find no cross-zone edges and always
        # return zeros.  Cross-seam D(u) computation is handled in
        # SALCoherenceLayer.coherence_state() via _build_seam_adjacency(),
        # which provides a proper multi-zone adjacency with correct node_zones.
        # Callers that pass a multi-zone adjacency directly may populate this
        # themselves by calling fragile_bridges() on the result.
        return HSHCoherenceField(
            zone=zone_val,
            phi_H=phi_H,
            lambda2_H=lambda2_H,
            curvature_integral=curvature_integral,
            mean_curvature=mean_k,
            passes_threshold=phi_H >= self.phi_min,
            phi_min=self.phi_min,
            hallucination_risk=hallucination_risk,
            node_curvatures=node_curvatures,
            fragile_bridge_nodes=[],
        )

    # ── Pre-Contradiction Displacement Signal D(u) ───────────

    def displacement_signal(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        node_zones: Optional[Dict[str, GraphZone]] = None,
        default_zone: GraphZone = GraphZone.WORK_KNOWLEDGE,
        seam_zones: Optional[Tuple[GraphZone, GraphZone]] = None,
    ) -> Dict[str, float]:
        """
        Pre-contradiction displacement signal D(u) for each node.

        D(u) measures how far node u has been pulled away from its zone's
        radial ring by cross-zone edge tension. A high D(u) at a zone seam
        (especially GOVERNANCE->RATIONALE) indicates a fragile bridge that
        is likely to become a contradiction before the coherence field detects
        it — because the Fiedler value only responds after connectivity has
        already been disrupted.

        Formula
        -------
        For each cross-zone edge (u -> v):

            directional_pressure(u, v) =
                |w(u, v)| * d_HSH(u, v) * |K(zone_u) - K(zone_v)|

        where K(zone) is the Gaussian curvature evaluated at the zone's
        characteristic disk radius (not the edge midpoint).  The
        directional curvature factor |K(zone_u) - K(zone_v)| weights the
        signal by the geometric severity of the zone crossing — a
        WORK->TEMPORAL edge (factor ~5.70) contributes far more pressure
        than a GOVERNANCE->RATIONALE edge (factor ~1.19) for the same
        raw tension, matching the actual risk in the Poincare disk model.

        Summed over the node's cross-zone neighborhood, then normalized:

            D(u) = sum_v directional_pressure(u, v)
                   / (deg(u) * ZONE_RADIUS[zone_u] + epsilon)

        Normalization by ZONE_RADIUS[zone_u] makes nodes closer to the
        governance center (lower radius) more sensitive to displacement
        pressure — which is correct because a small absolute displacement
        near the disk center represents a proportionally larger violation
        of zone inertia than the same displacement near the boundary.

        Parameters
        ----------
        adj : Dict[str, Dict[str, float]]
            Weighted adjacency {src: {dst: weight}}.
        nodes : List[str]
            Nodes to evaluate.
        node_zones : Dict[str, GraphZone], optional
            Per-node zone assignment. If None, all nodes use default_zone.
        default_zone : GraphZone
            Fallback zone when a node is absent from node_zones.
        seam_zones : Tuple[GraphZone, GraphZone], optional
            If provided, D(u) only accumulates pressure from edges that
            cross this specific zone pair (in either direction). Useful for
            targeted governance->rationale seam analysis. Default: all
            cross-zone edges contribute.

        Returns
        -------
        Dict[str, float]
            Per-node displacement D(u) in [0, inf).  Compare against
            seam_threshold() for the relevant zone pair; values above the
            threshold indicate a fragile bridge worth flagging before
            contradiction gating fires.  Isolated nodes (degree 0) return 0.0.
            Governance nodes (r=0) always return 0.0 — they define the reference
            frame and cannot themselves be displaced.
        """
        _zones: Dict[str, GraphZone] = node_zones or {}  # GraphZone values at runtime
        _eps = 1e-9

        d_signal: Dict[str, float] = {}

        for u in nodes:
            zone_u = _zones.get(u, default_zone)
            pos_u  = self.node_position(u, zone_u)
            r_u    = ZONE_RADIUS.get(zone_u, pos_u.radius)

            # Governance sits at disk center (r=0).  Normalization divides by
            # r_u, so r=0 produces near-zero denominator and astronomical D values.
            # Semantically, the governance anchor defines the reference frame —
            # it cannot itself be displaced.  Return 0.0 and skip.
            if r_u < _eps:
                d_signal[u] = 0.0
                continue

            neighbors = adj.get(u, {})
            total_pressure = 0.0
            cross_degree   = 0

            for v, w in neighbors.items():
                zone_v = _zones.get(v, default_zone)
                if zone_v == zone_u:
                    continue  # same-zone edge — skip

                # Filter to specific seam if requested
                if seam_zones is not None:
                    a, b = seam_zones
                    if not ({zone_u, zone_v} == {a, b}):
                        continue

                pos_v = self.node_position(v, zone_v)
                dist  = hsh_distance(pos_u.coords, pos_u.lambda_val,
                                     pos_v.coords, pos_v.lambda_val)
                dcf   = directional_curvature_factor(zone_u, zone_v)
                total_pressure += abs(w) * dist * dcf
                cross_degree   += 1

            if cross_degree == 0:
                d_signal[u] = 0.0
            else:
                # Normalize: cross_degree * r_u gives a per-zone baseline
                d_signal[u] = total_pressure / (cross_degree * r_u + _eps)

        return d_signal

    def fragile_bridges(
        self,
        adj: Dict[str, Dict[str, float]],
        nodes: List[str],
        node_zones: Optional[Dict[str, GraphZone]] = None,
        default_zone: GraphZone = GraphZone.WORK_KNOWLEDGE,
        threshold: Optional[float] = None,
        seam_zones: Optional[Tuple[GraphZone, GraphZone]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return nodes with D(u) above threshold, sorted descending by D(u).

        These are the pre-contradiction fragile bridges.  When threshold is
        None (default), the calibrated per-seam constant is used via
        seam_threshold(); this is the recommended path.  Pass an explicit
        float to override (e.g. for testing or non-standard seams).

        Parameters
        ----------
        adj, nodes, node_zones, default_zone, seam_zones
            Passed directly to displacement_signal().
        threshold : float or None
            D(u) must exceed this to be reported.  None auto-selects the
            calibrated seam threshold via seam_threshold(seam_zones).

        Returns
        -------
        List[Tuple[str, float]]
            (node_id, D_value) pairs sorted highest-D first.
        """
        _threshold = threshold if threshold is not None else seam_threshold(seam_zones)
        d_map = self.displacement_signal(
            adj, nodes,
            node_zones=node_zones,
            default_zone=default_zone,
            seam_zones=seam_zones,
        )
        return sorted(
            [(nid, d) for nid, d in d_map.items() if d > _threshold],
            key=lambda x: x[1],
            reverse=True,
        )

    def clear_position_cache(self) -> None:
        """Clear the in-memory position cache. Positions are re-derived on demand."""
        self._position_cache.clear()
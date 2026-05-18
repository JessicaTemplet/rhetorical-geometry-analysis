"""
VeritasMemoria - Field Theory Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements the scalar-tensor field theory machinery from Templet (2026) and
Templet (2026b), as specified in the VM Mathematical Implementation Specification
(Rev. 2, April 2026).

This module provides the NEW and MODIFIED components that extend the existing
HSH geometry, spectral coherence, and ORC flow layers:

  Phase architecture
  ------------------
  - Phase 0 (Bootstrap): retrieval methods active; field theory computed but
    dormant for state-machine decisions; λ₂ logging begins; saturation monitored.
  - Phase 1 (Operational): all field theory machinery active. Transition at
    mean ℓ_P ≤ ℓ_sat across active (non-governance) nodes.

  Healing scalar field Φ(x)  [Phase 0+, state machine use Phase 1+]
  -------------------------------------------------------------------
  Φ(x) = 1 - Σ_i ε_i · G(d_HSH(x, x_i))  [first-order approximation]
  G(d) = (1/2π) · log(coth(d/2))  — exact Green's function on H²

  Inertia-weighted Planck scale  [MODIFIED — replaces Templet 2026 Eq. 16]
  -------------------------------------------------------------------------
  ℓ_P(x_i) = (Σ_{j≠i} λ_j · e^{-d_HSH(x_i, x_j)})^{-1/2}
  Resolves the Density Paradox (Templet 2026b §4.2).

  Node gradient flow  [Phase 1+]
  --------------------------------
  Repulsive (Eq. 14): F_i^Φ = -η · Σ_{j≠i} ε_j · (-1/2π sinh(d_ij)) · ∇d_HSH
  Restoring (Eq. 15): F_i^V = -η · μ₀²αβ · cosh(β d(x_i,x_0)) · (1-Φ)² · ∇d_HSH
  Conformal step (Eq. 13): dx_i/dτ = -η · Φ(x_i) · total_force
  Governance nodes: F=0 enforced architecturally (not by forces evaluating to zero).

  Surgery trigger  [Phase 1+]
  -----------------------------
  Fires when d_HSH(x_i, x_j) < c · ℓ_P(x_i)  [c = 1.8].
  All discrete operations require human sign-off.

  λ₂ temporal logging  [Phase 0+]
  ---------------------------------
  Full time-series log, begun in Phase 0 to build the baseline Phase 1 requires.
  Feeds the derivative criterion and regime state machine.

  dλ₂/dt derivative warning criterion  [Phase 1+]
  -------------------------------------------------
  Fires when dλ₂/dt < -δ_c. Early warning for Regime 2 entry, before λ₂
  itself drops below threshold. Derivable from δ²S_eff (second variation).

  Three-regime state machine  [Phase 1+]
  ----------------------------------------
  Regime 1 (Coherent):  λ₂ above threshold AND |dλ₂/dt| < δ_c
  Regime 2 (Stressed):  dλ₂/dt < -δ_c; surgery threshold not yet met
  Regime 3 (Surgical):  d_HSH(x_i, x_j) < c·ℓ_P(x_i) for any pair

  |V|* monitoring  [Phase 1+]
  ----------------------------
  Planning horizon, not runtime control. Surfaces estimated time to critical
  graph size where contradiction generation rate exceeds human bandwidth.

  Exogeneity correlation test  [Phase 1+, optional]
  ---------------------------------------------------
  Tests whether human sign-off decisions correlate with local ∇Φ.
  Detects endogenization of the governance anchor.

Integration
-----------
  from veritas_memoria.analysis.coherence.field_theory_engine import (
      FieldTheoryEngine, ContradictionSource, PhaseState, Regime
  )
  from veritas_memoria.analysis.coherence.hsh_geometry import HSHGeometry

  hsh    = HSHGeometry()
  engine = FieldTheoryEngine(hsh)

  # Phase 0 — log λ₂, check saturation
  engine.log_lambda2("work_knowledge", lambda2_val)
  active_nodes = [("mem_abc", GraphZone.WORK_KNOWLEDGE), ...]
  engine.check_saturation(active_nodes)

  # Phase 1 — healing field, gradient flow, regime
  phi = engine.healing_field(pos, zone, contradiction_sources)
  result = engine.gradient_flow_step(active_nodes, contradiction_sources)
  regime = engine.compute_regime("work_knowledge", lambda2_val, lambda_min, False)

References
----------
  Templet, J. (2026). Epistemic Scalar-Tensor Gravity on Hyperbolic Belief
    Manifolds. Templet Solutions.
  Templet, J. (2026b). Exogeneity, Scalability Bounds, and the Geometry of
    Governance. Templet Solutions.
  VM Mathematical Implementation Specification, Rev. 2, April 2026.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from graph_types import GraphZone, Edge, EdgeKind, GateLevel 

import numpy as np

from hsh_geometry import (
    HSHGeometry,
    ZONE_LAMBDA,
    hsh_distance,
)
from graph_types import (
    GraphZone
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (from Templet 2026 §7.3 / VM Spec §7.3)
# ─────────────────────────────────────────────────────────────────────────────

ETA: float = 0.015          # gradient flow step size η
C_SURGERY: float = 1.8      # surgery threshold constant c
L_SAT_DEFAULT: float = 0.50 # default ℓ_sat for Phase 0→1 transition (§2.4)
TWO_PI: float = 2.0 * math.pi

# Governance zone offset — δ small enough to not perturb geometry but large
# enough to keep Planck scale distances well-defined. Matches the range
# produced by hsh_geometry.node_position() after the §3.4 patch.
_GOV_OFFSET_MAX: float = 1.1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class PhaseState(Enum):
    """VM operating phase (§2)."""
    BOOTSTRAP   = 0   # Phase 0: field theory computed but not used for control
    OPERATIONAL = 1   # Phase 1: all machinery active


class Regime(Enum):
    """Three-regime state machine (§11, Templet 2026b §3)."""
    COHERENT = 1   # λ₂ stable above threshold; |dλ₂/dt| < δ_c
    STRESSED = 2   # dλ₂/dt < −δ_c; surgery threshold not yet met
    SURGICAL = 3   # surgery trigger condition met


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContradictionSource:
    """
    A node that carries contradiction stress, i.e. a source term for Φ(x).

    node_id  — graph node identifier
    zone     — graph zone of this node
    position — 2D coords in the Poincaré disk (current or deterministic)
    epsilon  — contradiction source strength ε_i = Σ_{j: w_ij<0} |w_ij|
    """
    node_id:  str
    zone:     GraphZone
    position: np.ndarray   # shape (2,), |position| < 1
    epsilon:  float


@dataclass
class Lambda2LogEntry:
    """One time-stamped λ₂ measurement (§10.2)."""
    timestamp: str
    lambda2:   float
    zone:      str
    step:      int          # monotone step counter per zone


@dataclass
class RegimeState:
    """Output of the three-regime state machine (§11)."""
    regime:            Regime
    zone:              str
    timestamp:         str
    lambda2:           float
    dlambda2_dt:       float
    surgery_triggered: bool
    explanation:       str


@dataclass
class SurgeryEvent:
    """Fired when the surgery trigger condition is met for a node pair (§9.1)."""
    timestamp:             str
    node_i:                str
    node_j:                str
    d_hsh:                 float
    planck_scale_i:        float
    zone_i:                str
    requires_human_signoff: bool = True


@dataclass
class GradientFlowResult:
    """Output of one gradient flow step (§7)."""
    step:                 int
    displacements:        Dict[str, float]   # node_id → displacement magnitude
    mean_force_magnitude: float
    phi_by_node:          Dict[str, float]   # node_id → Φ(x_i)
    governance_constrained: List[str]        # governance nodes held fixed
    timestamp:            str


@dataclass
class VStarMetrics:
    """
    |V|* monitoring output (§12, Templet 2026b Theorem 2).

    This is a planning horizon, not a runtime control parameter.
    VM does not behave differently because it approaches |V|*.
    """
    timestamp:                    str
    node_count:                   int
    node_growth_rate:             float         # nodes per step (recent trend)
    contradiction_density:        float         # fraction of edges with w < 0
    r_human_estimate:             float         # sign-offs per second
    estimated_steps_to_vstar:     Optional[float]
    governance_review_recommended: bool
    planning_threshold:           float         # in steps


@dataclass
class SignOffRecord:
    """
    Human sign-off event, logged for the exogeneity correlation test (§13).

    Records the local Φ context at the time of the decision so we can
    later test whether decisions correlate with ∇Φ.
    """
    timestamp:      str
    node_ids:       List[str]
    phi_values:     Dict[str, float]          # node_id → Φ(x_i) at decision time
    phi_grad_norms: Dict[str, float]          # node_id → |∇Φ(x_i)|
    decision:       str                       # "flip" | "archive" | "defer"
    deferred:       bool


@dataclass
class ExogeneityTestResult:
    """
    Output of the governance integrity test (§13).

    Tests whether human decisions are predictable from ∇Φ — the signature of
    endogenization. A fully correlated human is geometrically equivalent to a
    high-inertia internal node and cannot serve as a genuine external reference.
    """
    timestamp:                            str
    n_sign_offs:                          int
    correlation_decision_vs_phi_gradient: float
    p_value_estimate:                     float
    endogenization_detected:              bool
    explanation:                          str


# ─────────────────────────────────────────────────────────────────────────────
# Stateless mathematical primitives
# ─────────────────────────────────────────────────────────────────────────────

def green_function(d: float, eps: float = 1e-9) -> float:
    """
    Exact Green's function of the Laplace-Beltrami operator on H².

    G(d) = (1/2π) · log(coth(d/2))

    This is the exact solution, not an approximation. It satisfies
    Δ_H G(x, y) = δ(x − y) in the distributional sense.

    Properties:
      - G(d) > 0 for all d > 0
      - G(d) → ∞ as d → 0 (short-distance divergence)
      - G(d) → 0 as d → ∞ (healing influence decays to zero far away)
      - dG/dd = −1/(2π sinh(d)) < 0 (monotone decreasing, drives repulsion)

    References: Templet (2026) Eq. (2); VM Spec §6.
    """
    d_half = max(d / 2.0, eps)
    sinh_half = math.sinh(d_half)
    cosh_half = math.cosh(d_half)
    # coth(x) = cosh(x)/sinh(x); guaranteed > 1 for x > 0
    coth_val = cosh_half / max(sinh_half, eps)
    return math.log(max(coth_val, 1.0 + eps)) / TWO_PI


def gradient_hsh_distance(
    pos_u: np.ndarray,
    lambda_u: float,
    pos_v: np.ndarray,
    lambda_v: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Gradient of d_HSH(u, v) with respect to u.

    Analytical derivation from the modified Poincaré metric:
        d_HSH = arcosh(1 + A / (B·C))
    where
        A = |u − v|²
        B = 1 − |u|²/λ_u   (> 0 inside the disk)
        C = 1 − |v|²/λ_v   (constant in u)

    ∂/∂u [A/(B·C)] = [2(u−v)·B·C + 2u·A·C/λ_u] / (B·C)²
                   = [2(u−v) + 2u·A/(λ_u·B)] / (B·C)

    ∇_u d = [∂f/∂u] / sqrt(f²−1) = [∂f/∂u] / sinh(d)

    Returns the zero vector when u = v or d ≈ 0 (no meaningful gradient).
    """
    A = float(np.sum((pos_u - pos_v) ** 2))
    B = max(1.0 - float(np.sum(pos_u ** 2)) / lambda_u, eps)
    C = max(1.0 - float(np.sum(pos_v ** 2)) / lambda_v, eps)

    arg = 1.0 + A / (B * C)
    arg = max(arg, 1.0 + eps)
    d = math.acosh(arg)
    sinh_d = max(math.sinh(d), eps)

    # ∂/∂u [A/(B·C)] = [2(u−v) + 2u·A/(λ_u·B)] / (B·C)
    diff = pos_u - pos_v
    grad_f = (2.0 * diff + 2.0 * pos_u * A / (lambda_u * B)) / (B * C)

    return grad_f / sinh_d


# ─────────────────────────────────────────────────────────────────────────────
# Field Theory Engine
# ─────────────────────────────────────────────────────────────────────────────

class FieldTheoryEngine:
    """
    Scalar-tensor field theory machinery for VeritasMemoria.

    Implements the New and Modified components from the VM Spec (Rev. 2).
    Attaches to an HSHGeometry instance for disk geometry and positions.

    Usage pattern:

      # Instantiate once, share across the system
      engine = FieldTheoryEngine(hsh_geometry_instance)

      # Every time SALCoherenceLayer computes λ₂, log it:
      engine.log_lambda2(zone_val, lambda2)

      # After each graph update during Phase 0, check saturation:
      engine.check_saturation([(node_id, zone), ...])

      # In Phase 1, run gradient flow and check regime:
      result = engine.gradient_flow_step(active_nodes, contradiction_sources)
      regime = engine.compute_regime(zone_val, lambda2, lambda_min, surgery_fired)
    """

    def __init__(
        self,
        hsh_geometry: HSHGeometry,
        l_sat:          float = L_SAT_DEFAULT,
        eta:            float = ETA,
        c_surgery:      float = C_SURGERY,
        mu0:            float = 1.0,
        alpha:          float = 1.0,
        beta:           float = 1.0,
        delta_c:        float = 0.01,
        lambda2_log_maxlen: int = 10_000,
        vstar_planning_threshold: float = 1_000,   # steps before governance review
    ):
        self.hsh       = hsh_geometry
        self.l_sat     = l_sat
        self.eta       = eta
        self.c_surgery = c_surgery
        self.mu0       = mu0
        self.alpha     = alpha
        self.beta      = beta
        self.delta_c   = delta_c

        # ── Phase state ──────────────────────────────────────
        self._phase: PhaseState = PhaseState.BOOTSTRAP
        self._phase_transition_time:  Optional[str]   = None
        self._mean_lp_at_transition:  Optional[float] = None

        # ── Mutable node positions (gradient flow, Phase 1) ──
        # Overrides HSHGeometry's deterministic hash positions.
        # Keys are node_ids; governance nodes are never written here.
        self._node_positions: Dict[str, np.ndarray] = {}

        # ── λ₂ temporal log (Phase 0 onward) ────────────────
        # zone_val → deque[Lambda2LogEntry]
        self._lambda2_log: Dict[str, deque] = {}
        self._lambda2_log_maxlen = lambda2_log_maxlen
        self._step_counter: Dict[str, int] = {}

        # ── Regime state per zone ────────────────────────────
        self._regime: Dict[str, RegimeState] = {}

        # ── Surgery events ───────────────────────────────────
        self._surgery_events: List[SurgeryEvent] = []

        # ── Gradient flow step counter ───────────────────────
        self._gradient_flow_step: int = 0

        # ── Sign-off log for exogeneity test ─────────────────
        self._sign_off_log: List[SignOffRecord] = []

        # ── |V|* node count history ──────────────────────────
        self._node_count_history: deque = deque(maxlen=1000)
        self._vstar_planning_threshold = vstar_planning_threshold

    # ── Phase state ───────────────────────────────────────────────────────────

    @property
    def phase(self) -> PhaseState:
        """Current operating phase."""
        return self._phase

    @property
    def is_operational(self) -> bool:
        """True when Phase 1 (Operational) is active."""
        return self._phase == PhaseState.OPERATIONAL

    @property
    def any_surgery_triggered(self) -> bool:
        """True if at least one surgery event has been logged (Phase 1)."""
        return bool(self._surgery_events)

    def check_saturation(
        self,
        active_nodes: List[Tuple[str, GraphZone]],
    ) -> bool:
        """
        Check the Phase 0 → Phase 1 saturation threshold (§2.4).

        Transition fires when:
            mean_i(ℓ_P(x_i)) ≤ ℓ_sat

        where the mean is taken over all active (non-governance) nodes and
        ℓ_P uses the inertia-weighted formula (§8 / Templet 2026b §4.2).

        Called after each significant graph update during Phase 0.
        No-op if already in Phase 1.

        Returns True if the transition has been triggered (Phase 1 now active).
        """
        if self._phase == PhaseState.OPERATIONAL:
            return True

        non_gov = [(nid, z) for nid, z in active_nodes if z != GraphZone.GOVERNANCE]
        if len(non_gov) < 2:
            return False

        lp_values = [
            self.planck_scale_inertia(nid, z, active_nodes)
            for nid, z in non_gov
        ]
        # Filter out infinite values (isolated nodes)
        finite_lp = [v for v in lp_values if math.isfinite(v)]
        if not finite_lp:
            return False

        mean_lp = float(np.mean(finite_lp))

        if mean_lp <= self.l_sat:
            self._phase = PhaseState.OPERATIONAL
            self._phase_transition_time = _utc_now()
            self._mean_lp_at_transition = mean_lp
            logger.info(
                "FieldTheoryEngine: Phase 0 → Phase 1 transition. "
                "mean ℓ_P = %.4f ≤ ℓ_sat = %.4f. "
                "Field theory state machine now active.",
                mean_lp, self.l_sat,
            )
            return True

        return False

    # ── Healing scalar field Φ(x) ────────────────────────────────────────────

    def healing_field(
        self,
        x_pos:                np.ndarray,
        x_zone:               GraphZone,
        contradiction_sources: List[ContradictionSource],
        eps:                  float = 1e-9,
    ) -> float:
        """
        Healing scalar field Φ(x) at an arbitrary point x in B (§6, VM Spec).

        Φ(x) = 1 − Σ_i ε_i · G(d_HSH(x, x_i))   [first-order approximation]

        Measures local epistemic coherence:
          - Φ = 1  in contradiction-free regions
          - Φ → 0  in heavily contradicted zones

        Evaluable at any point in B, not only at node positions.
        This is a closed-form field computable from existing data structures
        with no additional infrastructure.

        Parameters
        ----------
        x_pos : np.ndarray, shape (2,)
            Query point in the Poincaré disk.
        x_zone : GraphZone
            Zone of x (determines λ for the HSH distance denominator).
        contradiction_sources : List[ContradictionSource]
            All active contradiction nodes with their ε_i strengths.

        Returns
        -------
        float
            Φ(x) ∈ (−∞, 1]. Negative values are physical (strong contradiction
            clusters); callers that need a bounded signal may clamp to [0, 1].
        """
        lambda_x = ZONE_LAMBDA[x_zone]
        suppression = 0.0
        for src in contradiction_sources:
            d = hsh_distance(x_pos, lambda_x, src.position, ZONE_LAMBDA[src.zone])
            suppression += src.epsilon * green_function(d, eps=eps)
        return 1.0 - suppression

    def healing_field_at_node(
        self,
        node_id:              str,
        zone:                 GraphZone,
        contradiction_sources: List[ContradictionSource],
    ) -> float:
        """Φ(x_i) evaluated at a node's current (possibly gradient-flowed) position."""
        pos = self._get_position(node_id, zone)
        return self.healing_field(pos, zone, contradiction_sources)

    # ── Inertia-weighted Planck scale (MODIFIED — Templet 2026b §4.2) ────────

    def planck_scale_inertia(
        self,
        node_id:   str,
        node_zone: GraphZone,
        all_nodes: List[Tuple[str, GraphZone]],
        eps:       float = 1e-9,
    ) -> float:
        """
        Inertia-weighted local Planck scale ℓ_P(x_i).

        CORRECT formula (Templet 2026b §4.2, replaces Templet 2026 Eq. 16):

            ℓ_P(x_i) = (Σ_{j≠i} λ_j · e^{−d_HSH(x_i, x_j)})^{−1/2}

        where λ_j is the zone inertia of node j.

        This resolves the Density Paradox: the governance zone is sparsely
        populated by design, making its population-based Planck scale
        anomalously large and its surgery threshold maximally insensitive at
        the most critical location. The inertia-weighted form corrects this:
        three governance nodes at λ=1.0 contribute an effective density of 3.0
        per unit kernel, while five temporal nodes at λ=0.35 contribute only
        1.75. Governance registers as denser despite fewer nodes.

        Sensitivity ratio (governance/temporal) ≈ 1.7×. Bounded by the
        exponential decay e^{−d}, so distant nodes contribute negligibly
        regardless of their inertia.

        Active in Phase 0 for saturation monitoring; used in surgery trigger
        from Phase 1 only.

        Returns float('inf') for isolated nodes (weighted sum ≈ 0).
        """
        pos_i    = self._get_position(node_id, node_zone)
        lambda_i = ZONE_LAMBDA[node_zone]

        weighted_sum = 0.0
        for other_id, other_zone in all_nodes:
            if other_id == node_id:
                continue
            pos_j    = self._get_position(other_id, other_zone)
            lambda_j = ZONE_LAMBDA[other_zone]
            d = hsh_distance(pos_i, lambda_i, pos_j, lambda_j)
            weighted_sum += lambda_j * math.exp(-d)

        if weighted_sum < eps:
            return float("inf")

        return weighted_sum ** (-0.5)

    # ── Node gradient flow [Phase 1] ─────────────────────────────────────────

    def gradient_flow_step(
        self,
        active_nodes:          List[Tuple[str, GraphZone]],
        contradiction_sources: List[ContradictionSource],
        governance_anchor_pos: Optional[np.ndarray] = None,
    ) -> GradientFlowResult:
        """
        One step of node gradient flow (Phase 1 only — §7, Templet 2026).

        Implements the three equations from Templet (2026):

          Repulsive force (Eq. 14) — contradiction nodes push each other apart:
            F_i^Φ = −η · Σ_{j≠i} ε_j · (−1/(2π sinh(d_ij))) · ∇_{x_i} d_HSH(x_i, x_j)

          Restoring force (Eq. 15) — contradicted nodes pulled toward governance anchor:
            F_i^V = −η · μ₀² · α·β · cosh(β·d_HSH(x_i, x_0)) · (1−Φ(x_i))² · ∇d_HSH(x_i, x_0)

          Conformal-damped step (Eq. 13) — step measured in effective metric g̃ = Φg:
            dx_i/dτ = −η · Φ(x_i) · (δS_eff/δx_i)

        Governance constraint (§7.2): F=0 is enforced architecturally for
        governance nodes, regardless of what the force equations would produce.
        This is not a consequence of the forces naturally evaluating to zero.

        Parameters
        ----------
        active_nodes : list of (node_id, zone)
            All nodes to update. Governance nodes are excluded from position updates.
        contradiction_sources : List[ContradictionSource]
            Active contradiction sources with their positions and ε_i values.
        governance_anchor_pos : np.ndarray, optional
            Position of the governance anchor x_0. Defaults to the first
            governance node found, or origin if none present.

        Returns
        -------
        GradientFlowResult
        """
        if self._phase != PhaseState.OPERATIONAL:
            raise RuntimeError(
                "gradient_flow_step() called in Phase 0 (Bootstrap). "
                "Node gradient flow is dormant until the saturation threshold is met. "
                "Call check_saturation() after each graph update."
            )

        # Resolve governance anchor x_0
        if governance_anchor_pos is None:
            gov_nodes = [(nid, z) for nid, z in active_nodes if z == GraphZone.GOVERNANCE]
            if gov_nodes:
                governance_anchor_pos = self._get_position(*gov_nodes[0])
            else:
                governance_anchor_pos = np.zeros(2)
        lambda_anchor = ZONE_LAMBDA[GraphZone.GOVERNANCE]

        # Pre-compute Φ at all nodes
        phi_by_node: Dict[str, float] = {}
        for node_id, zone in active_nodes:
            pos_i = self._get_position(node_id, zone)
            phi_by_node[node_id] = self.healing_field(pos_i, zone, contradiction_sources)

        displacements:          Dict[str, float] = {}
        governance_constrained: List[str]        = []
        force_magnitudes:       List[float]      = []

        for node_id, zone in active_nodes:

            # ── Governance constraint (§7.2) ─────────────────
            if zone == GraphZone.GOVERNANCE:
                displacements[node_id] = 0.0
                governance_constrained.append(node_id)
                continue

            pos_i    = self._get_position(node_id, zone)
            lambda_i = ZONE_LAMBDA[zone]
            phi_i    = phi_by_node[node_id]

            # ── Repulsive force (Eq. 14) ─────────────────────
            F_repulsive = np.zeros(2)
            for src in contradiction_sources:
                if src.node_id == node_id:
                    continue
                d_ij    = hsh_distance(pos_i, lambda_i, src.position, ZONE_LAMBDA[src.zone])
                sinh_d  = max(math.sinh(d_ij), 1e-9)
                grad_d  = gradient_hsh_distance(pos_i, lambda_i, src.position, ZONE_LAMBDA[src.zone])
                # F^Φ = −η · ε_j · dG/dd · ∇d_HSH, where dG/dd = −1/(2π sinh(d))
                # So the contribution is: +η · ε_j / (2π sinh(d)) · ∇d_HSH
                F_repulsive += self.eta * src.epsilon / (TWO_PI * sinh_d) * grad_d

            # ── Restoring force (Eq. 15) ─────────────────────
            d_anchor    = hsh_distance(pos_i, lambda_i, governance_anchor_pos, lambda_anchor)
            grad_anchor = gradient_hsh_distance(pos_i, lambda_i, governance_anchor_pos, lambda_anchor)
            cosh_term   = math.cosh(self.beta * d_anchor)
            one_m_phi_sq = (1.0 - phi_i) ** 2
            F_restoring  = (
                -self.eta * (self.mu0 ** 2) * self.alpha * self.beta
                * cosh_term * one_m_phi_sq
                * grad_anchor
            )

            # ── Conformal-damped step (Eq. 13) ───────────────
            # dx_i/dτ = −η · Φ(x_i) · δS_eff/δx_i
            # Φ acts as conformal damping: node motion slows near contradiction
            # clusters, providing the UV regulator described in §7.3.
            total_force = phi_i * (F_repulsive + F_restoring)

            # Displace node
            new_pos = pos_i + total_force

            # Project back inside the Poincaré disk (|x| < 1)
            r_new = float(np.linalg.norm(new_pos))
            if r_new >= 1.0 - 1e-6:
                new_pos = new_pos * (1.0 - 1e-6) / r_new

            displacement_mag = float(np.linalg.norm(total_force))
            displacements[node_id] = displacement_mag
            force_magnitudes.append(float(np.linalg.norm(F_repulsive + F_restoring)))

            # Persist mutable position
            self._node_positions[node_id] = new_pos

        self._gradient_flow_step += 1

        return GradientFlowResult(
            step=self._gradient_flow_step,
            displacements=displacements,
            mean_force_magnitude=(
                float(np.mean(force_magnitudes)) if force_magnitudes else 0.0
            ),
            phi_by_node=phi_by_node,
            governance_constrained=governance_constrained,
            timestamp=_utc_now(),
        )

    # ── Surgery trigger [Phase 1] ─────────────────────────────────────────────

    def check_surgery_trigger(
        self,
        active_nodes: List[Tuple[str, GraphZone]],
    ) -> List[SurgeryEvent]:
        """
        Surgery trigger condition for all node pairs (§9.1, VM Spec).

        Fires when: d_HSH(x_i, x_j) < c · ℓ_P(x_i)   [c = 1.8]

        When triggered, the continuum approximation has broken down locally:
        nodes are too close relative to the healing field variation scale for
        gradient flow to be the appropriate dynamics. Discrete graph surgery
        is required. All three available operations (shadow-graph archival,
        forced edge flip, re-anchoring) require human sign-off.

        Phase 1 only. Returns empty list in Phase 0.

        Returns
        -------
        List[SurgeryEvent]
            One event per triggered pair. Also appended to self._surgery_events.
        """
        if self._phase != PhaseState.OPERATIONAL:
            return []

        events: List[SurgeryEvent] = []
        n = len(active_nodes)

        for i in range(n):
            node_i, zone_i = active_nodes[i]
            pos_i    = self._get_position(node_i, zone_i)
            lambda_i = ZONE_LAMBDA[zone_i]
            lp_i     = self.planck_scale_inertia(node_i, zone_i, active_nodes)
            threshold = self.c_surgery * lp_i

            for j in range(i + 1, n):
                node_j, zone_j = active_nodes[j]
                pos_j    = self._get_position(node_j, zone_j)
                lambda_j = ZONE_LAMBDA[zone_j]
                d_ij     = hsh_distance(pos_i, lambda_i, pos_j, lambda_j)

                if d_ij < threshold:
                    event = SurgeryEvent(
                        timestamp=_utc_now(),
                        node_i=node_i,
                        node_j=node_j,
                        d_hsh=round(d_ij, 6),
                        planck_scale_i=round(lp_i, 6),
                        zone_i=zone_i.value,
                        requires_human_signoff=True,
                    )
                    events.append(event)
                    self._surgery_events.append(event)
                    logger.warning(
                        "FieldTheoryEngine: Surgery trigger fired. "
                        "d_HSH(%s, %s) = %.4f < c·ℓ_P = %.4f. "
                        "Human sign-off required. Discrete operations available: "
                        "shadow-graph archival, forced edge flip, re-anchoring.",
                        node_i, node_j, d_ij, threshold,
                    )

        return events

    # ── λ₂ temporal logging [Phase 0 onward] ─────────────────────────────────

    def log_lambda2(
        self,
        zone_val:  str,
        lambda2:   float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Log a λ₂ measurement for a zone (§10.2).

        Logging begins in Phase 0 so that a stable baseline exists when
        Phase 1 starts. The derivative criterion in Phase 1 is only reliable
        with a baseline established over a period of Phase 0 operation.

        Called by SALCoherenceLayer (or any coherence computation) after each
        λ₂ computation step.
        """
        if zone_val not in self._lambda2_log:
            self._lambda2_log[zone_val] = deque(maxlen=self._lambda2_log_maxlen)
            self._step_counter[zone_val] = 0

        entry = Lambda2LogEntry(
            timestamp=timestamp or _utc_now(),
            lambda2=lambda2,
            zone=zone_val,
            step=self._step_counter[zone_val],
        )
        self._lambda2_log[zone_val].append(entry)
        self._step_counter[zone_val] += 1

    def lambda2_history(self, zone_val: str) -> List[Lambda2LogEntry]:
        """Return the full λ₂ log for a zone (for plotting and diagnostics)."""
        return list(self._lambda2_log.get(zone_val, []))

    def compute_dlambda2_dt(
        self,
        zone_val: str,
        window:   int = 10,
    ) -> Optional[float]:
        """
        Compute dλ₂/dt by linear regression over recent log entries (§10.2).

        Returns the slope (Δλ₂/Δstep) over the last `window` entries.
        Returns None if insufficient history exists (fewer than 2 entries).

        The derivative criterion requires no new data structures — it requires
        only that λ₂ be logged over time and differenced. Phase 1 only for
        control decisions; can be read in Phase 0 for diagnostic purposes.
        """
        log = self._lambda2_log.get(zone_val)
        if not log or len(log) < 2:
            return None

        entries = list(log)[-min(window, len(log)):]
        if len(entries) < 2:
            return None

        steps  = np.array([e.step    for e in entries], dtype=float)
        values = np.array([e.lambda2 for e in entries], dtype=float)

        denom = float(np.sum((steps - steps.mean()) ** 2))
        if denom < 1e-9:
            return None

        slope = float(
            np.sum((steps - steps.mean()) * (values - values.mean())) / denom
        )
        return slope

    # ── dλ₂/dt derivative warning criterion [Phase 1] ────────────────────────

    def derivative_warning_fires(
        self,
        zone_val: str,
        delta_c:  Optional[float] = None,
        window:   int = 10,
    ) -> bool:
        """
        Early warning criterion for Regime 2 entry (§10.2, Templet 2026b §3.2).

        Fires when: dλ₂/dt < −δ_c

        Derivable from the second variation of the effective action δ²S_eff.
        Fires BEFORE λ₂ itself drops below threshold, providing intervention
        time before connectivity is threatened. No new data beyond the λ₂ log.

        Returns False in Phase 0 (criterion dormant until Phase 1 baseline).
        Returns False if insufficient log history exists.
        """
        if self._phase != PhaseState.OPERATIONAL:
            return False
        threshold  = delta_c if delta_c is not None else self.delta_c
        dlambda2   = self.compute_dlambda2_dt(zone_val, window=window)
        return dlambda2 is not None and dlambda2 < -threshold

    # ── Three-regime state machine [Phase 1] ──────────────────────────────────

    def compute_regime(
        self,
        zone_val:         str,
        lambda2:          float,
        lambda_min:       float,
        surgery_triggered: bool,
        delta_c:          Optional[float] = None,
        window:           int = 10,
    ) -> RegimeState:
        """
        Three-regime state machine (§11, VM Spec; Templet 2026b §3).

        Regimes and entry conditions:

          Regime 1 (Coherent):
            λ₂ above threshold AND |dλ₂/dt| < δ_c
            Normal operation. Gradient flow and ORC flow run. No escalation.
            Human required: routine governance writes only.
            Exits to: Regime 2.

          Regime 2 (Stressed):
            dλ₂/dt < −δ_c. Surgery threshold not yet met.
            Pre-emptive review queue activated. Gradient flow continues.
            Contradiction edges surfaced for sign-off BEFORE surgery triggers.
            Human required: sign-offs requested proactively.
            Exits to: Regime 1 or Regime 3.

          Regime 3 (Surgical):
            d_HSH(x_i, x_j) < c·ℓ_P(x_i) for any pair.
            Discrete operations triggered. Gradient flow continues under
            conformal damping. If sign-off rate = 0: Opacity path.
            Human required: mandatory. System freezes if absent.
            Exits to: Regime 2.

        Regime 2 is the operationally critical window. The derivative criterion
        fires while transport paths are still open, giving the human time to
        resolve contradictions before the system enters surgery.

        State machine is dormant in Phase 0 — returns COHERENT with explanation.
        """
        dlambda2  = self.compute_dlambda2_dt(zone_val, window=window)
        threshold = delta_c if delta_c is not None else self.delta_c

        if self._phase != PhaseState.OPERATIONAL:
            regime = Regime.COHERENT
            explanation = (
                "Phase 0 (Bootstrap): three-regime state machine dormant. "
                "λ₂ logging active. Saturation threshold not yet reached."
            )
        elif surgery_triggered:
            regime = Regime.SURGICAL
            _dlambda2_str = f"{dlambda2:.6f}" if dlambda2 is not None else "n/a"
            explanation = (
                f"Regime 3 (Surgical): surgery trigger condition met. "
                f"Discrete intervention required. Human sign-off mandatory. "
                f"If sign-off rate drops to zero: Opacity path. "
                f"λ₂={lambda2:.4f}, dλ₂/dt={_dlambda2_str}."
            )
        elif dlambda2 is not None and dlambda2 < -threshold:
            regime = Regime.STRESSED
            explanation = (
                f"Regime 2 (Stressed): dλ₂/dt = {dlambda2:.6f} < −δ_c = {-threshold:.6f}. "
                f"Pre-emptive review queue activated. Transport paths still open. "
                f"λ₂={lambda2:.4f}. Human sign-offs requested proactively before "
                f"surgery triggers."
            )
        elif lambda2 <= lambda_min:
            # λ₂ below threshold but no rapid decline yet — still Regime 2
            regime = Regime.STRESSED
            explanation = (
                f"Regime 2 (Stressed): λ₂={lambda2:.4f} ≤ λ_min={lambda_min:.4f}. "
                f"Connectivity below threshold. dλ₂/dt = "
                f"{dlambda2:.6f if dlambda2 is not None else 'insufficient history'}. "
                f"Human sign-offs recommended."
            )
        else:
            regime = Regime.COHERENT
            dlambda2_str = f"{dlambda2:.6f}" if dlambda2 is not None else "insufficient history"
            explanation = (
                f"Regime 1 (Coherent): λ₂={lambda2:.4f} > {lambda_min}. "
                f"dλ₂/dt = {dlambda2_str} (stable). "
                f"Normal operation. Gradient flow and ORC flow running."
            )

        state = RegimeState(
            regime=regime,
            zone=zone_val,
            timestamp=_utc_now(),
            lambda2=lambda2,
            dlambda2_dt=dlambda2 if dlambda2 is not None else 0.0,
            surgery_triggered=surgery_triggered,
            explanation=explanation,
        )
        self._regime[zone_val] = state
        return state

    # ── |V|* monitoring [Phase 1] ─────────────────────────────────────────────

    def compute_vstar_metrics(
        self,
        node_count:             int,
        contradiction_density:  float,
        sign_offs_per_step:     Optional[float] = None,
        timestamp:              Optional[str]   = None,
    ) -> VStarMetrics:
        """
        |V|* monitoring (§12, VM Spec; Templet 2026b Theorem 2).

        |V|* is a planning horizon, not a runtime control parameter. VM does
        not behave differently because it approaches |V|*. What changes is the
        governance structure, and that is a human decision.

        The Bianchi conservation law and polynomial scaling of contradictions
        guarantee that |V|* exists. Its precise value depends on contradiction
        density and R_human, both empirical.

        Surfaces:
          - Current |V| and its growth rate
          - Current contradiction density
          - Estimated human sign-off bandwidth R_human
          - Estimated steps to |V|* given current trajectory

        When estimated steps to |V|* drops below the planning threshold,
        VM surfaces a governance structure review recommendation. This is an
        informational flag, not a system-level behavior change.
        """
        ts = timestamp or _utc_now()
        self._node_count_history.append((ts, node_count))

        # Estimate growth rate from recent history
        growth_rate = 0.0
        if len(self._node_count_history) >= 2:
            recent = list(self._node_count_history)[-min(20, len(self._node_count_history)):]
            if len(recent) >= 2:
                counts = [e[1] for e in recent]
                growth_rate = (counts[-1] - counts[0]) / max(len(counts) - 1, 1)

        # Estimate R_human
        r_human = sign_offs_per_step if sign_offs_per_step is not None else self._estimate_r_human_steps()

        # Estimate steps to |V|*
        # Theorem 2: |V|* satisfies contradiction_rate(|V|*) = R_human.
        # For a graph with contradiction_density d, contradiction rate ~ d · |V|².
        # At |V|*: d · |V|*² ≈ R_human → |V|* ≈ sqrt(R_human / d).
        estimated_steps: Optional[float] = None
        if growth_rate > 0 and r_human > 0 and contradiction_density > 0:
            vstar_estimate = math.sqrt(r_human / max(contradiction_density, 1e-9))
            remaining = max(vstar_estimate - node_count, 0.0)
            estimated_steps = remaining / growth_rate

        review_recommended = (
            estimated_steps is not None
            and estimated_steps < self._vstar_planning_threshold
        )

        if review_recommended:
            logger.warning(
                "FieldTheoryEngine: |V|* approach detected. "
                "Estimated %.0f steps to critical scale. "
                "Governance structure review recommended.",
                estimated_steps,
            )

        return VStarMetrics(
            timestamp=ts,
            node_count=node_count,
            node_growth_rate=growth_rate,
            contradiction_density=contradiction_density,
            r_human_estimate=r_human,
            estimated_steps_to_vstar=estimated_steps,
            governance_review_recommended=review_recommended,
            planning_threshold=self._vstar_planning_threshold,
        )

    def _estimate_r_human_steps(self) -> float:
        """Estimate sign-off rate from the sign-off log (sign-offs per step)."""
        n = len(self._sign_off_log)
        if n < 2:
            return 0.0
        # Use step range of the first and last sign-off's zone log entries
        # as a rough denominator; fall back to count-based estimate
        return float(n)  # caller can supply a more precise estimate

    # ── Exogeneity correlation test [Phase 1, optional] ──────────────────────

    def log_sign_off(
        self,
        node_ids:              List[str],
        node_zones:            Dict[str, GraphZone],
        contradiction_sources: List[ContradictionSource],
        decision:              str,
        timestamp:             Optional[str] = None,
    ) -> None:
        """
        Log a human sign-off with its local Φ context (§13.1).

        Records: the nodes reviewed, local Φ(x_i) and |∇Φ(x_i)| at each,
        and the decision. Deferred items are tracked separately.

        Called whenever a human completes a governance sign-off.
        """
        ts = timestamp or _utc_now()
        phi_values:     Dict[str, float] = {}
        phi_grad_norms: Dict[str, float] = {}

        for node_id in node_ids:
            zone = node_zones.get(node_id, GraphZone.WORK_KNOWLEDGE)
            pos  = self._get_position(node_id, zone)
            phi_values[node_id] = self.healing_field(pos, zone, contradiction_sources)
            grad = self._phi_gradient_numerical(pos, zone, contradiction_sources)
            phi_grad_norms[node_id] = float(np.linalg.norm(grad))

        self._sign_off_log.append(SignOffRecord(
            timestamp=ts,
            node_ids=list(node_ids),
            phi_values=phi_values,
            phi_grad_norms=phi_grad_norms,
            decision=decision,
            deferred=(decision == "defer"),
        ))

    def run_exogeneity_test(
        self,
        min_samples: int = 10,
    ) -> Optional[ExogeneityTestResult]:
        """
        Governance integrity test (§13, VM Spec).

        Tests whether human sign-off decisions are predictable from the local
        gradient of Φ. A statistically significant correlation indicates
        endogenization: the human's cognitive attention is being shaped by
        the manifold's stress field rather than by external reality.

        The test does not determine whether the human is making correct
        decisions. It determines whether their decisions are introducing
        information that the manifold's own dynamics cannot predict.

        A fully endogenized human (perfect correlation with ∇Φ) is
        geometrically equivalent to a high-inertia internal node and does
        not satisfy the exogeneity requirement placed by the Bianchi law.

        Returns None if fewer than min_samples sign-offs have been logged.
        """
        records = list(self._sign_off_log)
        if len(records) < min_samples:
            return None

        ts = _utc_now()
        n  = len(records)

        # Decision encoding: higher = more decisive resolution
        def _encode(d: str) -> float:
            return {"flip": 1.0, "archive": 0.7, "defer": 0.0}.get(d, 0.5)

        decisions       = np.array([_encode(r.decision) for r in records])
        phi_grad_norms  = np.array([
            float(np.mean(list(r.phi_grad_norms.values()))) if r.phi_grad_norms else 0.0
            for r in records
        ])

        # Pearson correlation between decision outcomes and |∇Φ| at decision time
        corr = 0.0
        if n > 1:
            c = np.corrcoef(decisions, phi_grad_norms)
            corr = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

        # Two-tailed t-test approximation: t = r·sqrt(n−2)/sqrt(1−r²)
        # Under H₀: ρ=0, t ~ t(n−2). For large n, t ~ N(0,1).
        abs_r   = abs(corr)
        p_value = 1.0
        if n > 2 and abs_r < 1.0:
            t_stat = abs_r * math.sqrt(n - 2) / math.sqrt(max(1.0 - abs_r ** 2, 1e-9))
            # Conservative normal approximation
            if   t_stat > 3.3: p_value = 0.001
            elif t_stat > 2.6: p_value = 0.010
            elif t_stat > 2.0: p_value = 0.050
            elif t_stat > 1.6: p_value = 0.100
            else:              p_value = 0.500

        endogenized = abs_r > 0.5 and p_value < 0.05

        if endogenized:
            explanation = (
                f"ENDOGENIZATION DETECTED (n={n}): "
                f"Pearson r(decisions, |∇Φ|) = {corr:.4f}, p ≈ {p_value:.3f}. "
                f"Human sign-off decisions show statistically significant correlation "
                f"with local Φ gradient. The governance zone label is intact, but "
                f"the human is geometrically equivalent to a high-inertia internal node. "
                f"The trust guarantee is degrading in proportion to the correlation."
            )
        else:
            explanation = (
                f"No endogenization detected (n={n}): "
                f"r(decisions, |∇Φ|) = {corr:.4f}, p ≈ {p_value:.3f}. "
                f"Human decisions remain unpredictable from ∇Φ. "
                f"Exogeneity requirement satisfied."
            )

        logger.info("FieldTheoryEngine: exogeneity test — %s", explanation)

        return ExogeneityTestResult(
            timestamp=ts,
            n_sign_offs=n,
            correlation_decision_vs_phi_gradient=round(corr, 6),
            p_value_estimate=round(p_value, 4),
            endogenization_detected=endogenized,
            explanation=explanation,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_position(self, node_id: str, zone: GraphZone) -> np.ndarray:
        """
        Current position for a node.

        Returns the gradient-flow-updated position if one exists (Phase 1),
        otherwise falls back to HSHGeometry's deterministic hash position.
        """
        if node_id in self._node_positions:
            return self._node_positions[node_id]
        pos_obj = self.hsh.node_position(node_id, zone)
        return pos_obj.coords

    def _phi_gradient_numerical(
        self,
        x_pos:                np.ndarray,
        x_zone:               GraphZone,
        contradiction_sources: List[ContradictionSource],
        delta:                float = 1e-5,
    ) -> np.ndarray:
        """
        Numerical gradient of Φ(x) using central differences.

        Used for the exogeneity test and diagnostics. Not called during
        gradient flow (which uses analytical force expressions).
        """
        grad = np.zeros(2)
        for dim in range(2):
            x_plus  = x_pos.copy(); x_plus[dim]  += delta
            x_minus = x_pos.copy(); x_minus[dim] -= delta
            # Project back inside disk
            for pt in (x_plus, x_minus):
                r = float(np.linalg.norm(pt))
                if r >= 1.0 - 1e-6:
                    pt[:] = pt * (1.0 - 1e-6) / r
            phi_plus  = self.healing_field(x_plus,  x_zone, contradiction_sources)
            phi_minus = self.healing_field(x_minus, x_zone, contradiction_sources)
            grad[dim] = (phi_plus - phi_minus) / (2.0 * delta)
        return grad

    # ── Status and diagnostics ────────────────────────────────────────────────

    def status(self) -> dict:
        """JSON-serializable status snapshot."""
        return {
            "phase":                  self._phase.name,
            "phase_transition_time":  self._phase_transition_time,
            "mean_lp_at_transition":  self._mean_lp_at_transition,
            "gradient_flow_step":     self._gradient_flow_step,
            "l_sat":                  self.l_sat,
            "c_surgery":              self.c_surgery,
            "eta":                    self.eta,
            "delta_c":                self.delta_c,
            "lambda2_log_zones": {
                zone: len(log)
                for zone, log in self._lambda2_log.items()
            },
            "regime": {
                zone: {
                    "regime":            state.regime.name,
                    "lambda2":           state.lambda2,
                    "dlambda2_dt":       state.dlambda2_dt,
                    "surgery_triggered": state.surgery_triggered,
                }
                for zone, state in self._regime.items()
            },
            "surgery_events_count":    len(self._surgery_events),
            "sign_off_count":          len(self._sign_off_log),
            "node_positions_mutable":  len(self._node_positions),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

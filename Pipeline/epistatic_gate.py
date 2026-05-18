"""
VeritasMemoria - Epistatic Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements gene-style conditional silencing (defeasible reasoning) for the
belief graph. Allows Regulatory Nodes in IDENTITY or GOVERNANCE zones to
suppress Structural Nodes in WORK or KNOWLEDGE zones without deleting the
underlying data.

Why this matters
-----------------
Standard contradiction detection is binary: Fact A contradicts Fact B, both
get blocked. Epistasis is subtler. A Medical Directive doesn't *contradict*
a Fitness Goal — it *conditionally suppresses* it when active. Remove the
Medical Directive and the Fitness Goal reappears intact. Delete it and you've
lost information. This module handles that distinction.

The Hill Equation
-----------------
Silencing is not binary. We use the Hill equation from biochemistry:

    f(S) = 1 / (1 + (S/K)^n)

    S   — activation strength of the modifier (bridge edge weight × confidence)
    K   — threshold for 50% inhibition (tunable per gate instance)
    n   — cooperativity: how "snappy" the silencing is
            n = 1 → gentle, logarithmic fade
            n = 2 → sigmoidal (default, balanced)
            n ≥ 4 → near-binary switch

f(S) returns expression level [0, 1]. Inhibition = 1 - f(S).

Multiple inhibitors
-------------------
When more than one modifier acts on a target, the expression levels multiply:

    total_expression = ∏ f(S_i)

This is the independent inhibitor model: each active modifier independently
reduces expression. Inhibitors stack but never produce absolute zero.

Regulatory Hierarchy
---------------------
To prevent the O(N²) combinatorial explosion:

    Eligible modifiers: nodes in IDENTITY or GOVERNANCE zones only.
    Eligible targets:   nodes in WORK or KNOWLEDGE zones only.
    Namespace check:    A modifier can only inhibit a target if a bridge
                        edge connects them. A bridge edge is the explicit
                        declaration of a cross-zone relationship — it is
                        proof that someone decided these two nodes are in
                        the same semantic domain.

This reduces complexity to O(B) where B is the number of bridge edges from
regulatory to structural zones — typically very small.

Latent Conflict Monitor
-----------------------
When a node is silenced, it doesn't disappear — it becomes a "Ghost Fact."
If the modifier ever weakens, the silenced node reactivates and may suddenly
contradict the current plan.

For each silenced node, the monitor computes:
    reactivation_threshold  — modifier strength at which the node reappears
    reactivation_risk       — (S - S_threshold) / S: what fraction the
                              modifier needs to weaken before reactivation
    warning                 — human-readable alert for the Auditor

A reactivation_risk of 0.20 means the modifier needs to weaken by only 20%
for the ghost fact to reappear. Flag this prominently.

Integration
-----------
EpistaticGate slots in between graph retrieval and the reasoning layer.
It is read-only: it never modifies node states or edge weights.

    from veritas_memoria.analysis.epistatic_gate import EpistaticGate

    gate = EpistaticGate(graph_engine)

    # Before returning retrieved nodes to the reasoning layer:
    result = gate.evaluate_expression(
        target_zone=GraphZone.WORK_KNOWLEDGE,
        active_node_ids=retrieved_node_ids,
        active_modifiers=current_governance_and_identity_node_ids,
    )

    # result.expressed_nodes  — safe to reason with
    # result.latent_conflicts — pass to Auditor
    # result.silenced_nodes   — log, do not discard

Relationship to contradiction detection
-----------------------------------------
Contradiction detection (graph.detect_and_gate_contradiction) handles the
case where two facts are mutually exclusive. Epistatic gating handles the
case where one rule conditionally suppresses another fact without invalidating
it. They are complementary: run contradiction detection first (it produces
hard blocks), then run the epistatic gate on the surviving nodes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Zone role definitions
# ─────────────────────────────────────────────────────────────

# Only nodes in these zones can act as epistatic inhibitors.
# These are the "regulatory genes" — governance directives and domain knowledge
# that conditionally suppress lower-authority beliefs.
# Under HSH: GOVERNANCE (lambda=1.0) and RATIONALE (lambda=0.65)
# sit closest to the hyperbolic anchor and carry maximum inhibitory authority.
_REGULATORY_ZONES: set = {"governance", "rationale"}

# Only nodes in these zones can be silenced.
# These are the "structural genes" — matter-scoped facts and ephemeral tasks
# that can be suppressed by higher-authority regulatory nodes.
# Under HSH: WORK_KNOWLEDGE (lambda=0.55) and TEMPORAL_KNOWLEDGE (lambda=0.35)
# sit farther from the anchor and yield to regulatory pressure.
_STRUCTURAL_ZONES: set = {"work_knowledge", "temporal_knowledge"}


# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────

# Inhibition level above which a node is considered silenced.
# 0.7 means the node's expression is below 30% of normal.
DEFAULT_SUPPRESSION_THRESHOLD: float = 0.70

# Modifier strength at which 50% inhibition is achieved.
# A bridge edge weight of 0.5 with K=0.5 produces 50% inhibition.
DEFAULT_HILL_K: float = 0.50

# Cooperativity: how sharply inhibition ramps up.
# 2.0 = sigmoidal (recommended default).
DEFAULT_HILL_N: float = 2.0

# Reactivation risk fraction at or above which to emit a high-priority warning.
DEFAULT_HIGH_RISK_THRESHOLD: float = 0.30


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class LatentConflict:
    """
    A silenced node that poses a future reactivation risk.

    silenced_node_id        The node currently suppressed by the gate.
    primary_modifier_id     The strongest active modifier (the one most
                            responsible for the suppression).
    all_modifier_ids        All active modifiers contributing to suppression.
    current_expression      Current expression level [0, 1]. Low = mostly silenced.
    current_inhibition      1 - current_expression.
    primary_strength        Bridge weight of the primary modifier.
    reactivation_threshold  Modifier strength at which the node reactivates.
                            Derived from Hill equation inversion.
    reactivation_risk       (S - S_threshold) / S. Fraction the primary modifier
                            needs to weaken before the ghost fact reappears.
                            0.0 = already at the edge; 1.0 = very stable.
    is_high_risk            True when reactivation_risk < high_risk_threshold:
                            the modifier only needs to weaken slightly.
    warning                 Human-readable alert for the Auditor.
    """
    silenced_node_id:      str
    primary_modifier_id:   str
    all_modifier_ids:      List[str]
    current_expression:    float
    current_inhibition:    float
    primary_strength:      float
    reactivation_threshold: float
    reactivation_risk:     float
    is_high_risk:          bool
    warning:               str


@dataclass
class ExpressionResult:
    """
    Output of EpistaticGate.evaluate_expression().

    expressed_nodes     Node IDs that passed the gate — safe to reason with.
    silenced_nodes      Dict of silenced_node_id -> list of modifier_ids.
    inhibition_scores   Dict of node_id -> inhibition level [0, 1] for all
                        evaluated nodes. Useful for partial suppression tracking.
    expression_scores   Dict of node_id -> expression level [0, 1].
    latent_conflicts    LatentConflict records for each silenced node.
    timestamp           ISO timestamp of this evaluation.
    """
    expressed_nodes:    List[str]
    silenced_nodes:     Dict[str, List[str]]
    inhibition_scores:  Dict[str, float]
    expression_scores:  Dict[str, float]
    latent_conflicts:   List[LatentConflict]
    timestamp:          str

    def has_ghosts(self) -> bool:
        """True if any silenced nodes exist with reactivation risk."""
        return len(self.latent_conflicts) > 0

    def high_risk_ghosts(self) -> List[LatentConflict]:
        """Latent conflicts where the modifier only needs minor weakening."""
        return [lc for lc in self.latent_conflicts if lc.is_high_risk]

    def summary(self) -> str:
        expressed = len(self.expressed_nodes)
        silenced = len(self.silenced_nodes)
        high_risk = len(self.high_risk_ghosts())
        return (
            f"Expressed: {expressed}, Silenced: {silenced}, "
            f"Latent conflicts: {len(self.latent_conflicts)} "
            f"({high_risk} high-risk)"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "expressed_nodes": self.expressed_nodes,
            "silenced_count": len(self.silenced_nodes),
            "silenced_nodes": self.silenced_nodes,
            "inhibition_scores": self.inhibition_scores,
            "expression_scores": self.expression_scores,
            "latent_conflicts": [
                {
                    "silenced_node_id": lc.silenced_node_id,
                    "primary_modifier_id": lc.primary_modifier_id,
                    "all_modifier_ids": lc.all_modifier_ids,
                    "current_expression": lc.current_expression,
                    "current_inhibition": lc.current_inhibition,
                    "primary_strength": lc.primary_strength,
                    "reactivation_threshold": lc.reactivation_threshold,
                    "reactivation_risk": lc.reactivation_risk,
                    "is_high_risk": lc.is_high_risk,
                    "warning": lc.warning,
                }
                for lc in self.latent_conflicts
            ],
        }


# ─────────────────────────────────────────────────────────────
# Epistatic Gate
# ─────────────────────────────────────────────────────────────

class EpistaticGate:
    """
    Pre-retrieval silencing layer based on gene-style epistasis.

    Reads bridge edges from the GraphEngine to determine which regulatory
    nodes (IDENTITY/GOVERNANCE) are active modifiers for structural nodes
    (WORK/KNOWLEDGE). Applies Hill-equation inhibition and returns which
    nodes are currently expressed.

    Read-only: never modifies edge weights, node states, or any persistent data.

    Parameters
    ----------
    graph_engine : GraphEngine
        The graph engine to read bridge edges from.
    suppression_threshold : float
        Inhibition level above which a node is silenced. Default 0.70.
    hill_K : float
        Bridge weight at which 50% inhibition is achieved. Default 0.50.
    hill_n : float
        Cooperativity. 1=gradual, 2=sigmoidal (default), ≥4=switch-like.
    high_risk_threshold : float
        Reactivation risk fraction below which a latent conflict is flagged
        as high-risk. Default 0.30 (modifier needs to weaken < 30%).
    """

    def __init__(
        self,
        graph_engine,
        suppression_threshold: float = DEFAULT_SUPPRESSION_THRESHOLD,
        hill_K: float = DEFAULT_HILL_K,
        hill_n: float = DEFAULT_HILL_N,
        high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    ):
        self.graph = graph_engine
        self.suppression_threshold = suppression_threshold
        self.hill_K = hill_K
        self.hill_n = hill_n
        self.high_risk_threshold = high_risk_threshold

    # ── Public API ────────────────────────────────────────────

    def evaluate_expression(
        self,
        target_zone,
        active_node_ids: List[str],
        active_modifiers: Optional[List[str]] = None,
    ) -> ExpressionResult:
        """
        Evaluate which nodes from a target zone are currently expressed.

        Parameters
        ----------
        target_zone : GraphZone
            The zone being filtered (WORK or KNOWLEDGE).
        active_node_ids : list of str
            Nodes from the target zone that retrieval returned and that
            are candidates for the reasoning layer.
        active_modifiers : list of str, optional
            Node IDs from IDENTITY or GOVERNANCE zones that are currently
            active in the session context. If None, all regulatory nodes
            that have bridge edges to this zone are considered active.
            Providing this explicitly lets callers scope inhibition to
            whatever governance/identity context is currently loaded.

        Returns
        -------
        ExpressionResult
            expressed_nodes, silenced_nodes, inhibition scores,
            and latent conflict records.
        """
        zone_val = target_zone.value if hasattr(target_zone, "value") else str(target_zone)
        now = datetime.now(timezone.utc).isoformat()

        if not active_node_ids:
            return ExpressionResult(
                expressed_nodes=[],
                silenced_nodes={},
                inhibition_scores={},
                expression_scores={},
                latent_conflicts=[],
                timestamp=now,
            )

        # Build the set of active modifier node IDs. If the caller provides
        # explicit modifiers, use those. Otherwise infer from bridge edges.
        if active_modifiers is not None:
            modifier_set: Set[str] = set(active_modifiers)
        else:
            modifier_set = self._infer_active_modifiers(target_zone)

        # Build regulatory pairs: (modifier_id, target_id) -> bridge_weight
        # Only pairs where modifier is in modifier_set and target is in active_node_ids.
        target_set = set(active_node_ids)
        regulatory_pairs: Dict[Tuple[str, str], float] = self._build_regulatory_pairs(
            target_zone, modifier_set, target_set
        )

        # Evaluate expression for each active node
        expressed_nodes: List[str] = []
        silenced_nodes: Dict[str, List[str]] = {}
        inhibition_scores: Dict[str, float] = {}
        expression_scores: Dict[str, float] = {}

        # Group regulatory pairs by target for efficient lookup
        target_to_modifiers: Dict[str, Dict[str, float]] = {}
        for (mod_id, tgt_id), strength in regulatory_pairs.items():
            target_to_modifiers.setdefault(tgt_id, {})[mod_id] = strength

        blocked = self.graph._blocked_nodes

        for node_id in active_node_ids:
            # Nodes with hard contradiction blocks are already excluded
            # upstream; treat them as not expressible here too.
            if node_id in blocked:
                inhibition_scores[node_id] = 1.0
                expression_scores[node_id] = 0.0
                silenced_nodes[node_id] = ["[contradiction_block]"]
                continue

            modifiers_for_node = target_to_modifiers.get(node_id, {})
            total_expression = self._total_expression(modifiers_for_node)
            inhibition = 1.0 - total_expression

            inhibition_scores[node_id] = round(inhibition, 4)
            expression_scores[node_id] = round(total_expression, 4)

            if inhibition >= self.suppression_threshold:
                # Node is silenced
                active_mod_list = list(modifiers_for_node.keys())
                silenced_nodes[node_id] = active_mod_list
                logger.debug(
                    "EpistaticGate: node '%s' silenced by %s (inhibition=%.3f)",
                    node_id, active_mod_list, inhibition,
                )
            else:
                expressed_nodes.append(node_id)

        # Build latent conflict records for silenced nodes
        latent_conflicts = self._compute_latent_conflicts(
            silenced_nodes=silenced_nodes,
            target_to_modifiers=target_to_modifiers,
            expression_scores=expression_scores,
            inhibition_scores=inhibition_scores,
        )

        result = ExpressionResult(
            expressed_nodes=expressed_nodes,
            silenced_nodes=silenced_nodes,
            inhibition_scores=inhibition_scores,
            expression_scores=expression_scores,
            latent_conflicts=latent_conflicts,
            timestamp=now,
        )

        if silenced_nodes:
            logger.info("EpistaticGate (%s): %s", zone_val, result.summary())

        return result

    def hill_expression(self, S: float, K: Optional[float] = None, n: Optional[float] = None) -> float:
        """
        Compute the Hill expression level for a single modifier.

        f(S) = 1 / (1 + (S/K)^n)

        Returns expression level in [0, 1].
            S = 0   → f = 1.0 (fully expressed, no modifier)
            S = K   → f = 0.5 (50% expression, threshold)
            S >> K  → f → 0.0 (fully silenced)

        Public so callers can inspect inhibition curves for any (K, n) pair.
        """
        K = K if K is not None else self.hill_K
        n = n if n is not None else self.hill_n
        if S <= 0.0:
            return 1.0
        if K <= 0.0:
            return 0.0
        try:
            ratio = (S / K) ** n
        except OverflowError:
            return 0.0
        return 1.0 / (1.0 + ratio)

    def reactivation_threshold(self, K: Optional[float] = None, n: Optional[float] = None) -> float:
        """
        Compute the modifier strength at which a silenced node reactivates.

        Solves f(S') = 1 - suppression_threshold for S':
            S' = K × (1 / (1 - threshold) - 1)^(1/n)

        If the primary modifier's current strength drops to S', the node
        reappears in the expressed set.
        """
        K = K if K is not None else self.hill_K
        n = n if n is not None else self.hill_n
        t = self.suppression_threshold
        # Expression at reactivation = 1 - threshold (just above the gate)
        # f(S') = 1 - t  →  1 + (S'/K)^n = 1/(1-t)  →  (S'/K)^n = t/(1-t)
        inner = t / max(1.0 - t, 1e-9)
        if inner <= 0.0:
            return 0.0
        try:
            return K * (inner ** (1.0 / n))
        except (OverflowError, ValueError):
            return float("inf")

    def scan_zone_for_ghosts(self, target_zone) -> List[LatentConflict]:
        """
        Scan an entire zone for latent conflicts regardless of active retrieval.

        Useful for scheduled audits: finds all nodes that are currently being
        suppressed and would reactivate if their primary modifier weakened.
        More expensive than evaluate_expression() since it scans all bridge edges.

        Returns LatentConflict records sorted by reactivation_risk ascending
        (most dangerous first — smallest modifier weakening needed).
        """
        zone_val = target_zone.value if hasattr(target_zone, "value") else str(target_zone)
        modifier_set = self._infer_active_modifiers(target_zone)
        all_targets = self._all_targets_in_zone(target_zone)

        if not all_targets or not modifier_set:
            return []

        regulatory_pairs = self._build_regulatory_pairs(target_zone, modifier_set, all_targets)
        if not regulatory_pairs:
            return []

        target_to_modifiers: Dict[str, Dict[str, float]] = {}
        for (mod_id, tgt_id), strength in regulatory_pairs.items():
            target_to_modifiers.setdefault(tgt_id, {})[mod_id] = strength

        expression_scores = {}
        inhibition_scores = {}
        silenced_nodes: Dict[str, List[str]] = {}

        for node_id, mods in target_to_modifiers.items():
            expr = self._total_expression(mods)
            inh = 1.0 - expr
            expression_scores[node_id] = round(expr, 4)
            inhibition_scores[node_id] = round(inh, 4)
            if inh >= self.suppression_threshold:
                silenced_nodes[node_id] = list(mods.keys())

        conflicts = self._compute_latent_conflicts(
            silenced_nodes=silenced_nodes,
            target_to_modifiers=target_to_modifiers,
            expression_scores=expression_scores,
            inhibition_scores=inhibition_scores,
        )

        # Most dangerous first (smallest reactivation_risk = easiest to trigger)
        conflicts.sort(key=lambda lc: lc.reactivation_risk)
        logger.info(
            "EpistaticGate.scan_zone_for_ghosts('%s'): %d latent conflicts found.",
            zone_val, len(conflicts),
        )
        return conflicts

    # ── Internal: bridge-based regulatory map ────────────────

    def _infer_active_modifiers(self, target_zone) -> Set[str]:
        """
        Find all regulatory node IDs that have bridge edges into target_zone.
        These are the "potential inhibitors" for this zone.
        """
        modifiers: Set[str] = set()
        for (from_zone, from_id), destinations in self.graph._bridges.items():
            zone_val = from_zone.value if hasattr(from_zone, "value") else str(from_zone)
            if zone_val not in _REGULATORY_ZONES:
                continue
            for (to_zone, to_id, weight, gate) in destinations:
                to_zone_val = to_zone.value if hasattr(to_zone, "value") else str(to_zone)
                tgt_zone_val = target_zone.value if hasattr(target_zone, "value") else str(target_zone)
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if to_zone_val == tgt_zone_val and gate_val != "block_until_resolved":
                    modifiers.add(from_id)
        return modifiers

    def _all_targets_in_zone(self, target_zone) -> Set[str]:
        """Find all node IDs that appear as destinations from regulatory bridges into target_zone."""
        targets: Set[str] = set()
        tgt_zone_val = target_zone.value if hasattr(target_zone, "value") else str(target_zone)
        for (from_zone, from_id), destinations in self.graph._bridges.items():
            zone_val = from_zone.value if hasattr(from_zone, "value") else str(from_zone)
            if zone_val not in _REGULATORY_ZONES:
                continue
            for (to_zone, to_id, weight, gate) in destinations:
                to_zone_val = to_zone.value if hasattr(to_zone, "value") else str(to_zone)
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if to_zone_val == tgt_zone_val and gate_val != "block_until_resolved":
                    targets.add(to_id)
        return targets

    def _build_regulatory_pairs(
        self,
        target_zone,
        modifier_set: Set[str],
        target_set: Set[str],
    ) -> Dict[Tuple[str, str], float]:
        """
        Build (modifier_id, target_id) -> inhibitor_strength pairs.

        Only includes pairs where:
          - modifier is in modifier_set (active regulatory nodes)
          - target is in target_set (active structural nodes)
          - bridge edge gate is not BLOCK_UNTIL_RESOLVED
          - bridge points from a regulatory zone into target_zone

        Inhibitor strength = bridge_weight (the declared confidence of the
        cross-zone relationship; already scaled [0, 1] by the edge schema).
        """
        pairs: Dict[Tuple[str, str], float] = {}
        tgt_zone_val = target_zone.value if hasattr(target_zone, "value") else str(target_zone)

        for (from_zone, from_id), destinations in self.graph._bridges.items():
            if from_id not in modifier_set:
                continue
            zone_val = from_zone.value if hasattr(from_zone, "value") else str(from_zone)
            if zone_val not in _REGULATORY_ZONES:
                continue

            for (to_zone, to_id, weight, gate) in destinations:
                if to_id not in target_set:
                    continue
                to_zone_val = to_zone.value if hasattr(to_zone, "value") else str(to_zone)
                if to_zone_val != tgt_zone_val:
                    continue
                gate_val = gate.value if hasattr(gate, "value") else str(gate)
                if gate_val == "block_until_resolved":
                    continue
                # Use the stronger weight if there are multiple bridge edges
                key = (from_id, to_id)
                pairs[key] = max(pairs.get(key, 0.0), float(weight))

        return pairs

    # ── Internal: Hill equation ───────────────────────────────

    def _total_expression(self, modifiers: Dict[str, float]) -> float:
        """
        Compute total expression level for a node given its active modifiers.

        Uses the multiplicative independent inhibitor model:
            total_expression = ∏ f(S_i)

        Each modifier independently reduces expression. Two modifiers at K
        each produce 0.5 × 0.5 = 0.25 total expression (75% inhibition),
        not 100% inhibition. This prevents complete silencing from ever
        being permanent — remove one modifier and expression partially recovers.
        """
        if not modifiers:
            return 1.0
        expression = 1.0
        for strength in modifiers.values():
            expression *= self.hill_expression(strength)
        return max(0.0, min(1.0, expression))

    # ── Internal: Latent Conflict Monitor ────────────────────

    def _compute_latent_conflicts(
        self,
        silenced_nodes: Dict[str, List[str]],
        target_to_modifiers: Dict[str, Dict[str, float]],
        expression_scores: Dict[str, float],
        inhibition_scores: Dict[str, float],
    ) -> List[LatentConflict]:
        """
        Build LatentConflict records for every silenced node.

        For each silenced node, identifies:
          - The primary modifier (highest bridge weight = strongest inhibitor)
          - The reactivation threshold (S' where node would reappear)
          - The reactivation risk (how little the modifier needs to weaken)
          - A human-readable warning for the Auditor
        """
        s_reactivate = self.reactivation_threshold()
        conflicts: List[LatentConflict] = []

        for node_id, mod_ids in silenced_nodes.items():
            # [contradiction_block] is a sentinel, not a real modifier
            real_mods = [m for m in mod_ids if m != "[contradiction_block]"]
            if not real_mods:
                continue

            mods = target_to_modifiers.get(node_id, {})
            if not mods:
                continue

            # Primary modifier = the one with the highest strength
            primary_modifier_id = max(mods, key=lambda m: mods.get(m, 0.0))
            primary_strength = mods[primary_modifier_id]

            current_expression = expression_scores.get(node_id, 0.0)
            current_inhibition = inhibition_scores.get(node_id, 1.0)

            # Reactivation risk: how much the primary modifier needs to weaken.
            # If primary_strength > s_reactivate: node is silenced with margin.
            # If primary_strength <= s_reactivate: already at/near the edge.
            if primary_strength > 1e-8:
                margin = primary_strength - s_reactivate
                reactivation_risk = round(max(0.0, min(1.0, margin / primary_strength)), 4)
            else:
                reactivation_risk = 0.0

            is_high_risk = reactivation_risk < self.high_risk_threshold

            if is_high_risk:
                risk_label = "HIGH RISK"
                risk_pct = int((1.0 - reactivation_risk) * 100)
                warning = (
                    f"[{risk_label}] Node '{node_id}' is suppressed by "
                    f"'{primary_modifier_id}' (strength={primary_strength:.3f}, "
                    f"inhibition={current_inhibition:.1%}). "
                    f"If '{primary_modifier_id}' weakens by just {100 - risk_pct}%, "
                    f"this node reactivates and may contradict current plans. "
                    f"Reactivation threshold: S={s_reactivate:.3f}."
                )
            else:
                drop_pct = int((1.0 - reactivation_risk) * 100)
                warning = (
                    f"Node '{node_id}' is suppressed by "
                    f"'{primary_modifier_id}' (strength={primary_strength:.3f}, "
                    f"inhibition={current_inhibition:.1%}). "
                    f"Reactivates if modifier weakens by {drop_pct}% "
                    f"(threshold S={s_reactivate:.3f}). "
                    f"Currently stable."
                )

            conflicts.append(LatentConflict(
                silenced_node_id=node_id,
                primary_modifier_id=primary_modifier_id,
                all_modifier_ids=real_mods,
                current_expression=round(current_expression, 4),
                current_inhibition=round(current_inhibition, 4),
                primary_strength=round(primary_strength, 4),
                reactivation_threshold=round(s_reactivate, 4),
                reactivation_risk=reactivation_risk,
                is_high_risk=is_high_risk,
                warning=warning,
            ))

        return conflicts

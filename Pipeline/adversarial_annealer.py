"""
VeritasMemoria - Adversarial Annealer (Skeptic Layer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses Simulated Annealing with a Metropolis-Hastings acceptance criterion to
search for the 'Most Stable Lie' — belief graph configurations where spectral
coherence (lambda2) is high but topological circularity (Betti-1) is also high.

High lambda2 looks like a healthy, converged belief graph.
High Betti-1 means the apparent coherence is built on circular reasoning loops
rather than independent evidence paths.

The combination is the adversarial target: a graph that passes the SAL
coherence gate but is structurally deceptive.

This module is read-only with respect to the actual belief graph. It works on
a copy of the weighted adjacency dict and never writes back to storage.

Perturbation strategies
------------------------
Three operations are applied probabilistically during the annealing search:

1. Weight shift  -- scale a random edge weight by a temperature-modulated
                    factor. High temperature = large shifts (exploration).
                    Low temperature = fine-grained nudges (exploitation).

2. Loop close    -- add a weak shortcut edge between two nodes that share a
                    common neighbor but are not directly connected. Tests
                    whether a single added shortcut creates a circular path
                    that boosts apparent coherence without adding information.

3. Weight invert -- negate a random edge weight. The signed Laplacian treats
                    negative weights as contradiction edges. Applied only at
                    high temperature to test gross structural instability.

Severity scale
--------------
adversarial_energy < 0.5   CRITICAL  — strong circular coherence detected
adversarial_energy < 1.5   HIGH      — significant circularity present
adversarial_energy < 3.0   MODERATE  — mild circularity, monitor
adversarial_energy >= 3.0  LOW       — no meaningful adversarial configuration

Integration
-----------
Call run_sabotage(zone) from SystemHealthOrchestrator after each sleep pass,
or on-demand when a zone shows unexpectedly high lambda2 alongside elevated
Betti-1 from persistent_homology.py.

    from veritas_memoria.analysis.coherence.adversarial_annealer import AdversarialAnnealer

    annealer = AdversarialAnnealer(sal_layer=sal, tda_layer=ph)
    report = annealer.run_sabotage(zone=GraphZone.RATIONALE)
    if report["severity"] in ("high", "critical"):
        # Surface to SystemHealthOrchestrator
        ...
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AdversarialAnnealer:
    """
    The Skeptic Layer. Searches the belief graph's configuration space for
    metastable delusions — states that look coherent but reason in circles.

    Parameters
    ----------
    sal_layer : SALCoherenceLayer
        The spectral coherence layer to compute lambda2 from.
    tda_layer : PersistentHomologyLayer
        The TDA layer to compute Betti-1 (loop count) from.
    initial_temp : float
        Starting temperature for the annealing schedule. Higher values allow
        larger initial perturbations and broader exploration.
    cooling_rate : float
        Multiplicative cooling factor applied each iteration (0 < rate < 1).
        0.98 gives a slow, thorough search. 0.90 is faster but shallower.
    max_iterations : int
        M-07: Hard cap on the number of annealing steps regardless of
        temperature, preventing runaway loops on unusual graphs.  With the
        default cooling rate of 0.98 the temperature schedule naturally
        terminates in ~342 steps; this cap adds an explicit safety net.
    patience : int
        M-07: Early-stop threshold.  If best_energy has not improved for
        this many consecutive iterations the search is terminated.  Set to
        0 to disable early stopping.
    """

    def __init__(
        self,
        sal_layer,
        tda_layer,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.98,
        max_iterations: int = 500,
        patience: int = 50,
    ):
        self.sal = sal_layer
        self.tda = tda_layer
        self._initial_temp = initial_temp
        self.T = initial_temp
        self.cooling = cooling_rate
        self.min_temp = 0.1
        self.max_iterations = max_iterations  # M-07
        self.patience = patience              # M-07

    # ── Public API ────────────────────────────────────────────

    def calculate_energy(self, adj: Dict, nodes: list) -> float:
        """
        Cost function for the adversarial search.

        We want to MINIMISE energy. Low energy = more deceptive configuration.

        Energy = (1 / (lambda2 + epsilon))   -- rewards high coherence
               + (Betti_1 * 2.0)             -- penalises circular loops

        A state with high lambda2 AND high Betti_1 has:
            low  1/(l2+eps)  ← coherence looks good
            high Betti_1*2.0 ← lots of loops

        Their sum produces a low overall energy — the adversarial sweet spot.
        A genuinely healthy graph has high l2 AND low Betti_1, so
        the Betti_1 penalty keeps energy high (safe).
        """
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        L, _ = self.sal._build_normalized_laplacian(adj, nodes, node_idx)
        l2 = self.sal._compute_fiedler_value(L, len(nodes))

        tda_report = self.tda.analyze_topology(
            "STRESS_TEST", nodes, self._adj_to_edges(adj)
        )
        b1 = self._parse_betti_1(tda_report)

        energy = (1.0 / (l2 + 1e-6)) + (b1 * 2.0)
        return energy

    def run_sabotage(self, zone) -> Dict:
        """
        Run the adversarial annealing search on the given zone.

        Builds the weighted adjacency for the zone from SAL, then iterates
        the Metropolis-Hastings loop, proposing perturbations and accepting
        or rejecting based on the energy delta and current temperature.

        Parameters
        ----------
        zone : GraphZone
            The zone to run the adversarial search on.

        Returns
        -------
        dict
            Triage report with adversarial_energy, severity, interpretation,
            and details about any inverted edges found.
        """
        # Reset temperature so repeated calls produce independent searches.
        self.T = self._initial_temp

        adj, _ = self.sal._build_weighted_adjacency(zone)
        nodes = sorted(adj.keys())

        if not nodes:
            return self._generate_triage_report({}, float("inf"))

        current_energy = self.calculate_energy(adj, nodes)
        best_adj = {k: dict(v) for k, v in adj.items()}
        best_energy = current_energy

        # M-07: track iteration count and stagnation for convergence safeguards
        iteration = 0
        no_improve_streak = 0

        while self.T > self.min_temp and iteration < self.max_iterations:
            candidate_adj = self._propose_perturbation(adj)
            candidate_energy = self.calculate_energy(candidate_adj, nodes)

            delta_e = candidate_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / self.T):
                adj = candidate_adj
                current_energy = candidate_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = {k: dict(v) for k, v in candidate_adj.items()}
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            else:
                no_improve_streak += 1

            # M-07: early stop when the search has stagnated
            if self.patience > 0 and no_improve_streak >= self.patience:
                import logging as _log
                _log.getLogger(__name__).debug(
                    "AdversarialAnnealer: early stop at iteration %d "
                    "(no improvement for %d consecutive steps)",
                    iteration + 1, self.patience,
                )
                break

            self.T *= self.cooling
            iteration += 1

        return self._generate_triage_report(best_adj, best_energy, iterations_run=iteration + 1)

    # ── Perturbation ──────────────────────────────────────────

    def _propose_perturbation(self, adj: Dict) -> Dict:
        """
        Propose an adversarial graph perturbation.

        Three strategies selected probabilistically:

        weight_shift (55% base probability)
            Scale a random existing edge weight by a temperature-modulated
            factor. Explores local connectivity strength variations.
            The scale range narrows as temperature falls so the search
            transitions naturally from exploration to exploitation.

        loop_close (remaining probability)
            Find two nodes that share a common neighbor but have no direct
            edge and add a weak shortcut between them. This tests whether
            a single shortcut creates a circular reasoning path that boosts
            lambda2 without adding genuinely independent evidence.

        weight_invert (20% probability at high temperature only)
            Negate an edge weight. The signed Laplacian interprets negative
            edges as semantic contradictions. Applied only at high temperature
            where broad structural probing is appropriate.

        Symmetry is maintained for all modifications: if the reverse edge
        exists it is updated to match.
        """
        new_adj: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in adj.items()}
        nodes = list(new_adj.keys())
        if not nodes:
            return new_adj

        edges: List[Tuple[str, str]] = [
            (src, dst)
            for src, nbrs in new_adj.items()
            for dst in nbrs
        ]
        if not edges:
            return new_adj

        # Strategy selection
        r = random.random()
        if self.T > 10.0 and r < 0.20:
            strategy = "invert"
        elif r < 0.55:
            strategy = "weight_shift"
        else:
            strategy = "loop_close"

        if strategy == "weight_shift":
            src, dst = random.choice(edges)
            current_w = new_adj[src][dst]
            # Scale range is proportional to temperature so perturbations
            # shrink as the search focuses.
            scale_range = max(0.05, min(0.50, self.T / 200.0))
            factor = 1.0 + random.uniform(-scale_range, scale_range)
            new_w = max(0.0, min(1.0, current_w * factor))
            new_adj[src][dst] = new_w
            if dst in new_adj and src in new_adj.get(dst, {}):
                new_adj[dst][src] = new_w

        elif strategy == "loop_close":
            # Search for a non-connected pair that shares a neighbour.
            # A shortcut there creates a triangle and raises Betti-1.
            shuffled = list(nodes)
            random.shuffle(shuffled)
            closed = False
            for src in shuffled[:20]:
                nbrs_src = set(new_adj.get(src, {}).keys())
                for mid in list(nbrs_src)[:10]:
                    nbrs_mid = set(new_adj.get(mid, {}).keys())
                    candidates = nbrs_mid - nbrs_src - {src}
                    if candidates:
                        dst = random.choice(list(candidates))
                        # Weak shortcut: 40% of the source node's average weight.
                        avg_w = (
                            sum(new_adj[src].values()) / len(new_adj[src])
                            if new_adj.get(src)
                            else 0.3
                        )
                        shortcut_w = max(0.05, avg_w * 0.4)
                        new_adj.setdefault(src, {})[dst] = shortcut_w
                        new_adj.setdefault(dst, {})[src] = shortcut_w
                        closed = True
                        break
                if closed:
                    break

            if not closed:
                # No suitable triangle found — fall back to weight shift.
                src, dst = random.choice(edges)
                current_w = new_adj[src][dst]
                new_adj[src][dst] = max(
                    0.0, min(1.0, current_w * random.uniform(0.85, 1.15))
                )

        elif strategy == "invert":
            # Negate an edge's weight to simulate a semantic contradiction.
            # High temperature only — too destructive for fine-grained search.
            src, dst = random.choice(edges)
            new_adj[src][dst] = -abs(new_adj[src][dst])
            if dst in new_adj and src in new_adj.get(dst, {}):
                new_adj[dst][src] = new_adj[src][dst]

        return new_adj

    # ── Helpers ───────────────────────────────────────────────

    def _adj_to_edges(self, adj: Dict) -> List[Tuple]:
        """
        Convert {node: {neighbor: weight}} to a deduplicated edge list
        [(src, dst, weight)] for the TDA layer's analyze_topology() call.

        Uses absolute weight so negated (inverted) edges still register as
        edges in the filtration rather than disappearing.
        """
        seen: set = set()
        edges = []
        for src, nbrs in adj.items():
            for dst, weight in nbrs.items():
                key = (min(src, dst), max(src, dst))
                if key not in seen:
                    seen.add(key)
                    edges.append((src, dst, abs(float(weight))))
        return edges

    def _parse_betti_1(self, tda_report: Any) -> float:
        """
        Extract the Betti-1 number (independent loop count) from a TDA report.

        Handles dict reports, object-with-attributes reports, and None.
        Falls back to 0.0 rather than raising so a TDA failure degrades
        gracefully to spectral-only analysis.
        """
        if tda_report is None:
            return 0.0

        if isinstance(tda_report, dict):
            for key in ("betti_1", "betti1", "h1_count", "loops", "holes"):
                if key in tda_report:
                    return float(tda_report[key])
            betti = tda_report.get("betti_numbers", {})
            if isinstance(betti, dict):
                return float(betti.get(1, betti.get("1", 0.0)))

        for attr in ("betti_1", "betti1", "h1_count", "loops"):
            if hasattr(tda_report, attr):
                return float(getattr(tda_report, attr))

        return 0.0

    def _generate_triage_report(
        self, best_adj: Dict, best_energy: float, iterations_run: int = 0
    ) -> Dict:
        """
        Summarise the most adversarial configuration found by the search.

        Returns a structured dict suitable for logging, SystemHealthOrchestrator
        consumption, or direct surfacing to the human operator.

        Fields
        ------
        adversarial_energy   float   Lower = more deceptive configuration found.
        severity             str     critical | high | moderate | low
        node_count           int     Nodes in the searched zone.
        edge_count           int     Edges (undirected) in best configuration.
        inverted_edge_count  int     Edges with negative weight (sign-flipped).
        inverted_edges       list    Up to 10 inverted edges with weights.
        interpretation       str     Human-readable summary and action guidance.
        """
        nodes = list(best_adj.keys())
        edge_count = sum(len(nbrs) for nbrs in best_adj.values()) // 2

        inverted_edges = [
            {"src": src, "dst": dst, "weight": round(w, 4)}
            for src, nbrs in best_adj.items()
            for dst, w in nbrs.items()
            if w < 0
        ]

        if best_energy == float("inf"):
            severity = "low"
            interp = "Zone is empty — no adversarial search performed."
        elif best_energy < 0.5:
            severity = "critical"
            interp = (
                "[CRITICAL] Strong circular coherence detected. The belief graph "
                "may be self-confirming via reasoning loops rather than independent "
                "evidence. Run EpistaticGate.scan_zone_for_ghosts() and review "
                "SIRPropagationAnalyzer for blast radius before next commit."
            )
        elif best_energy < 1.5:
            severity = "high"
            interp = (
                "[HIGH] Significant circularity present. Apparent coherence is "
                "partially supported by loop structure. Investigate inverted_edges "
                "and cross-reference with persistent_homology H1 pairs."
            )
        elif best_energy < 3.0:
            severity = "moderate"
            interp = (
                "[MODERATE] Mild circularity detected. Monitor across sessions. "
                "No immediate action required unless lambda2 is also rising unexpectedly."
            )
        else:
            severity = "low"
            interp = (
                "No meaningful adversarial configuration found. "
                "Belief graph appears structurally sound under simulated stress."
            )

        return {
            "adversarial_energy": round(best_energy, 4) if best_energy != float("inf") else None,
            "severity": severity,
            "node_count": len(nodes),
            "edge_count": edge_count,
            "inverted_edge_count": len(inverted_edges),
            "inverted_edges": inverted_edges[:10],
            "iterations_run": iterations_run,   # M-07: expose for diagnostics
            "interpretation": interp,
        }

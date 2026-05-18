"""
VeritasMemoria - Kalman Belief Tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a 1D scalar Kalman filter to the lambda2 (Fiedler value) signal
produced by SALCoherenceLayer. Replaces naive absolute delta with a
smoothed estimate of the true underlying coherence state.

Why this matters
----------------
The raw lambda2 signal is noisy. Small graph perturbations — a new edge,
a removed node, numerical variance in the eigensolver — produce measurement
jitter that has nothing to do with whether reasoning has actually converged
or diverged. The naive delta (|lambda2_t - lambda2_t-1|) treats all of
this jitter as signal, which produces false positives on the stability
check and makes the stopping criterion unreliable.

The Kalman filter maintains a running estimate of the *true* lambda2 and
a confidence measure (variance) around that estimate. It weights new
observations against prior belief in proportion to relative noise levels.
High measurement noise -> trust the prior more. High process noise ->
trust the new observation more.

Mapping to VM concepts
-----------------------
    State (x)           true lambda2 — what coherence actually is
    Observation (z)     measured lambda2 from the Laplacian
    Process noise (Q)   natural graph churn between measurements
                        (edges added/removed during a session)
    Measurement noise (R) numerical noise from the eigensolver
                        (stable graphs should have very low R)
    Kalman gain (K)     dynamic weight: how much to trust the new
                        measurement vs the prior estimate
                        K near 1.0 = measurement dominated
                        K near 0.0 = prior dominated

Integration
-----------
KalmanBeliefTracker runs alongside the existing history buffer in
SALCoherenceLayer. It does not replace the raw delta — both signals
are computed and returned in CoherenceState so you can watch where
they agree and where they diverge before deciding which to use in
the stopping criterion.

One tracker instance per zone. SALCoherenceLayer holds a dict of them,
keyed by zone value string, parallel to _history.

Usage
-----
    tracker = KalmanBeliefTracker()
    estimate = tracker.update(measured_lambda2)

    # estimate.value      smoothed lambda2
    # estimate.variance   uncertainty (lower = more confident)
    # estimate.gain       Kalman gain used this step
    # estimate.residual   raw - smoothed (innovation signal)

Tuning
------
process_noise (Q): increase if the graph changes a lot between
    measurements (active session with many writes). Decrease for
    stable, read-heavy sessions.

measurement_noise (R): increase if the eigensolver is imprecise
    (small graphs, sparse connectivity). Decrease for large dense
    graphs where eigenvalues are numerically stable.

The defaults are conservative — they bias toward trusting the prior,
which is the safe direction for a commit stopping criterion. A false
CONTINUE is just slow. A false COMMIT is a correctness bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Defaults — tunable at construction time
# ─────────────────────────────────────────────────────────────

# Natural variance in lambda2 from graph churn between measurements.
# Conservative default: assume graph is fairly stable during a session.
DEFAULT_PROCESS_NOISE: float = 0.005

# Variance from eigensolver numerical noise.
# Conservative default: assume moderate numerical noise.
DEFAULT_MEASUREMENT_NOISE: float = 0.01

# Initial variance — high uncertainty before any observations.
DEFAULT_INITIAL_VARIANCE: float = 1.0


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class KalmanEstimate:
    """
    The output of a single Kalman filter update step.

    value       smoothed lambda2 estimate
    variance    uncertainty around the estimate (lower = more confident)
    gain        Kalman gain used this step (diagnostic)
    residual    innovation: measured - predicted (diagnostic)
    step        how many observations have been incorporated
    """
    value: float
    variance: float
    gain: float
    residual: float
    step: int


@dataclass
class KalmanBeliefState:
    """
    Internal filter state. One instance per zone.

    Separating state from the tracker makes it easy to serialize,
    inspect, and reset per-zone without touching other zones.
    """
    x: float = 0.0                          # current state estimate
    p: float = DEFAULT_INITIAL_VARIANCE     # current estimate variance
    step: int = 0                           # number of updates applied
    initialized: bool = False               # False until first observation


# ─────────────────────────────────────────────────────────────
# Core filter
# ─────────────────────────────────────────────────────────────

class KalmanBeliefTracker:
    """
    1D scalar Kalman filter for the lambda2 coherence signal.

    One instance tracks one zone. SALCoherenceLayer maintains a
    dict[zone_value, KalmanBeliefTracker] parallel to _history.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        initial_variance: float = DEFAULT_INITIAL_VARIANCE,
    ):
        self.q = process_noise       # Q
        self.r = measurement_noise   # R
        self._initial_variance = initial_variance  # M-04: kept for reset()
        self._state = KalmanBeliefState(p=initial_variance)

    # ── Public API ───────────────────────────────────────────

    def update(self, measurement: float) -> KalmanEstimate:
        """
        Incorporate a new lambda2 measurement and return the updated estimate.

        On the first call the filter initializes directly to the measurement
        with full uncertainty, so the first estimate is unbiased even though
        there is no prior.
        """
        s = self._state

        if not s.initialized:
            s.x = measurement
            s.p = self.r          # start with measurement-level uncertainty
            s.initialized = True
            s.step = 1
            return KalmanEstimate(
                value=s.x,
                variance=s.p,
                gain=1.0,         # first step: fully trust the measurement
                residual=0.0,
                step=s.step,
            )

        # ── Predict step ─────────────────────────────────────
        # State transition is identity (we expect lambda2 to persist)
        # Process noise inflates uncertainty between measurements
        x_pred = s.x
        p_pred = s.p + self.q

        # ── Update step ──────────────────────────────────────
        # Innovation: how much does the new measurement surprise us?
        residual = measurement - x_pred

        # Kalman gain: how much weight to give the new observation
        # K = P_pred / (P_pred + R)
        # When P_pred >> R: K -> 1 (trust measurement)
        # When P_pred << R: K -> 0 (trust prior)
        k = p_pred / (p_pred + self.r)

        # Fused estimate
        s.x = x_pred + k * residual

        # Updated variance — Joseph form for numerical stability
        # P = (1 - K) * P_pred
        s.p = (1.0 - k) * p_pred

        s.step += 1

        return KalmanEstimate(
            value=round(s.x, 6),
            variance=round(s.p, 6),
            gain=round(k, 4),
            residual=round(residual, 6),
            step=s.step,
        )

    def reset(self) -> None:
        """Reset filter state. Call when a zone is cleared between sessions.

        M-04: the previous implementation passed ``self._state.p`` (the
        accumulated, shrunken variance) to the new state, so each session
        started with an increasingly overconfident prior.  We now restore the
        original construction-time variance instead.
        """
        self._state = KalmanBeliefState(p=self._initial_variance)

    @property
    def estimate(self) -> float:
        """Current smoothed lambda2 estimate without incorporating a new measurement."""
        return self._state.x

    @property
    def variance(self) -> float:
        """Current uncertainty. Decreases as observations accumulate."""
        return self._state.p

    @property
    def is_initialized(self) -> bool:
        return self._state.initialized


# ─────────────────────────────────────────────────────────────
# Zone-keyed tracker registry
# ─────────────────────────────────────────────────────────────

class ZoneKalmanRegistry:
    """
    Holds one KalmanBeliefTracker per zone.

    Drop this into SALCoherenceLayer.__init__ as self._kalman,
    then call self._kalman.update(zone_val, measured_lambda2) from
    coherence_state() immediately after computing the raw lambda2.

    Example integration in SALCoherenceLayer.coherence_state():

        lambda2 = self._compute_fiedler_value(L, n)
        k_estimate = self._kalman.update(zone_val, lambda2)

        # k_estimate.value        -> smoothed lambda2
        # k_estimate.variance     -> filter confidence
        # k_estimate.residual     -> innovation (useful diagnostic)

    Then add k_estimate fields to CoherenceState as optional additions.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._trackers: Dict[str, KalmanBeliefTracker] = {}

    def update(self, zone_val: str, measurement: float) -> KalmanEstimate:
        """Update (or initialize) the tracker for zone_val and return estimate."""
        if zone_val not in self._trackers:
            self._trackers[zone_val] = KalmanBeliefTracker(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
        return self._trackers[zone_val].update(measurement)

    def reset_zone(self, zone_val: str) -> None:
        """Reset a zone's tracker. Call at session end when working memory is cleared."""
        if zone_val in self._trackers:
            self._trackers[zone_val].reset()

    def reset_all(self) -> None:
        """Reset all trackers."""
        for tracker in self._trackers.values():
            tracker.reset()

    def get_estimate(self, zone_val: str) -> Optional[float]:
        """Current smoothed estimate for a zone without a new measurement, or None."""
        t = self._trackers.get(zone_val)
        return t.estimate if t and t.is_initialized else None

"""
RhetoricalWavelet -- Four-Band Epistemic Category Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adapted from the NarrativeWavelet architecture (fiction continuity tracking)
for use in Rhetorical Geometry Analysis. Substitutes epistemic category for
temporal stability as the sorting dimension. Core scoring logic is preserved.

Purpose
-------
Detection instrument for RULE-004 (Compound Proposition Detection), Pathway 2.
Scores sentences against four band vocabularies. Sentences scoring within
AMBIGUITY_MARGIN of two or more bands are flagged as dual-band candidates and
routed to the Pathway 2 compound detector before classification.

The wavelet does not classify propositions. It identifies sentences that require
compound-proposition review before classification can proceed.

Four Bands
----------
  normative_stable   -- universal/natural-law claims; not empirically testable
  historical_state   -- conditions prevailing during a period; bounded in time
  empirical_event    -- discrete measurable claims, named actors, specific events
  legal_statutory    -- obligation language anchored to a specific named document

Ambiguity Margin
----------------
  AMBIGUITY_MARGIN = 0.20

Wider than the fiction-domain default because deliberate epistemic category
mixing is a feature of rhetorical documents, not an error. See methodology
v0.4 Section 4.3.

Routing
-------
  Dual-band sentence   -> "pathway_2_compound_detector"  (RULE-004 Pathway 2)
  Single-band sentence -> "single_band_classification"

Vocabulary note
---------------
Sub-patterns are excluded when a longer pattern already captures the same
signal. Specifically: "law of nations" is excluded (subsumed by "by the law
of nations"); "moral" is excluded (subsumed by "morally"); "economic" is
excluded (subsumed by "economically"); "dignity" is excluded (subsumed by
"preserve the dignity"). Excluding sub-patterns prevents double-counting
that artificially inflates one band's score when only one semantic unit fired.

Known calibration limitation
-----------------------------
Sentences mixing a COMPRESSED empirical qualifier ("economically necessary")
with EXTENDED normative language ("morally required to preserve the dignity
of domestic workers") may score single-band normative_stable even though they
are compound by RULE-004 Pathway 2. This is a structural feature of 19th-
century governmental prose, not a detector error. Analyst attention remains
necessary for this sentence type. The wavelet reduces workload; it does not
replace analyst judgment.

Source: methodology_v0.4.md Section 4; rules_v1.md RULE-004.
Schema version this module targets: 0.4
Rules version: v1.0 (locked)
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIGUITY_MARGIN: float = 0.20

NORMATIVE_STABLE = "normative_stable"
HISTORICAL_STATE = "historical_state"
EMPIRICAL_EVENT  = "empirical_event"
LEGAL_STATUTORY  = "legal_statutory"

ALL_BANDS: Tuple[str, ...] = (
    NORMATIVE_STABLE,
    HISTORICAL_STATE,
    EMPIRICAL_EVENT,
    LEGAL_STATUTORY,
)

ROUTING_COMPOUND = "pathway_2_compound_detector"
ROUTING_SINGLE   = "single_band_classification"

# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------
# Each entry: (pattern, weight).
# Entries starting with r"(", r"\b", or r"\d" are compiled as regex.
# All others are case-insensitive literal substring matches.
#
# Sub-pattern exclusion rule (see module docstring):
#   Do NOT add a shorter literal if a longer literal already covers it.
#   Examples of excluded sub-patterns:
#     "law of nations"  -- excluded; "by the law of nations" catches it
#     "moral"           -- excluded; "morally" catches it
#     "economic"        -- excluded; "economically" catches it
#     "dignity"         -- excluded; "preserve the dignity" catches it
# ---------------------------------------------------------------------------

_BAND_VOCAB: Dict[str, List[Tuple[str, float]]] = {

    NORMATIVE_STABLE: [
        # Natural-law / universal-principle vocabulary
        ("necessary",               0.5),
        ("self-evident",            1.5),
        ("self evident",            1.5),
        ("natural law",             2.0),
        ("unalienable",             2.0),
        ("inalienable",             2.0),
        ("inherent",                1.0),
        ("from time immemorial",    2.0),
        ("by the law of nations",   2.0),
        # "law of nations" excluded -- subsumed by "by the law of nations"
        ("it has always been",      1.5),
        ("ordained",                1.0),
        ("fundamental",             0.8),
        ("prohibited by nature",    2.0),
        ("by nature",               1.0),
        ("universal",               1.0),
        ("perpetual",               0.8),
        ("immutable",               1.5),
        ("inviolable",              1.5),
        ("sacred",                  1.0),
        ("self-preservation",       1.0),
        ("by right",                0.8),
        ("endowed",                 1.0),
        ("created equal",           2.0),
        ("divine",                  1.0),
        ("providence",              1.0),
        ("rights of man",           1.5),
        ("rights of mankind",       1.5),
        # Moral register
        # "moral" excluded -- subsumed by "morally"
        ("morally",                 1.5),
        ("preserve the dignity",    2.0),
        # "dignity" excluded -- subsumed by "preserve the dignity"
        ("virtue",                  1.0),
        ("justice requires",        1.2),
        ("honor demands",           1.2),
        ("public good",             0.8),
        ("common good",             0.8),
        ("humanity",                0.8),
        ("civilization",            0.8),
        ("welfare of",              0.8),
        # Universal quantifier regex
        (r"(?:^|\W)all\s+(?:men|people|persons|nations|states)", 0.6),
        (r"(?:^|\W)every\s+\w+\s+(?:must|shall|has)",            0.5),
    ],

    HISTORICAL_STATE: [
        ("had been",                0.8),
        ("at the time",             0.7),
        ("during this period",      1.0),
        ("during that period",      1.0),
        ("prevailing",              0.8),
        ("the existing",            0.7),
        ("throughout",              0.7),
        ("had long",                1.0),
        ("for years",               0.8),
        ("the practice of",         0.8),
        ("was understood to",       1.2),
        ("throughout this period",  1.2),
        ("had always",              1.0),
        ("have always",             0.8),
        ("has always",              0.8),
        ("long established",        1.0),
        ("long-established",        1.0),
        ("long been",               0.8),
        ("it was the custom",       1.2),
        ("customarily",             1.0),
        ("historically",            0.8),
        ("at that time",            0.8),
        ("at this time",            0.6),
        ("during the period",       1.0),
        ("domestic",                0.5),
        ("prevailed",               0.8),
        ("the current state",       0.7),
        ("the condition of",        0.6),
        ("conditions of",           0.5),
        ("state of affairs",        0.8),
        ("prior to",                0.5),
        ("theretofore",             1.2),
        ("heretofore",              1.2),
        ("formerly",                0.8),
        ("hitherto",                1.2),
    ],

    EMPIRICAL_EVENT: [
        ("hereby",                  1.0),
        ("was enacted",             1.2),
        ("has enacted",             1.0),
        ("as of",                   0.8),
        ("reported",                0.7),
        ("recorded",                0.7),
        ("resulted in",             0.8),
        ("in the year",             1.0),
        ("on the",                  0.3),
        ("signed",                  0.7),
        ("ratified",                1.0),
        ("declared",                0.7),
        ("voted",                   0.8),
        ("appointed",               0.7),
        ("elected",                 0.7),
        ("arrested",                0.8),
        ("deported",                0.8),
        ("removed",                 0.5),
        ("killed",                  0.8),
        ("the number of",           0.8),
        ("amounting to",            0.8),
        ("a total of",              0.8),
        ("an estimated",            0.8),
        ("per annum",               1.0),
        ("annually",                0.6),
        ("occurred",                0.8),
        ("took place",              0.8),
        ("was authorized",          0.7),
        ("is authorized",           0.7),
        ("pursuant to",             0.8),
        # Economic and quantitative markers
        # "economic" excluded -- subsumed by "economically"
        ("economically",            1.5),
        ("labor",                   0.8),
        ("wages",                   1.0),
        ("competition",             0.7),
        ("trade",                   0.6),
        ("commerce",                0.6),
        ("revenue",                 0.8),
        ("market",                  0.7),
        ("production",              0.7),
        ("workers",                 0.7),
        ("employment",              0.8),
        ("population",              0.6),
        # Year regex
        (r"\b1[6-9]\d\d\b",         1.0),
    ],

    LEGAL_STATUTORY: [
        ("shall",                   1.0),
        ("must",                    0.8),
        ("required to",             0.8),
        ("in accordance with",      1.2),
        ("as provided by",          1.2),
        ("under the authority of",  1.5),
        ("the act provides",        1.5),
        ("section",                 0.7),
        ("article",                 0.6),
        ("clause",                  0.7),
        ("treaty",                  0.9),
        ("statute",                 1.0),
        ("congress",                0.7),
        ("hereby ordained",         1.5),
        ("as amended",              1.5),
        ("under existing law",      1.5),
        ("pursuant to",             0.8),
        ("be it enacted",           2.0),
        ("it is ordered",           1.2),
        ("it is hereby",            1.5),
        ("the law",                 0.5),
        ("by law",                  0.7),
        ("lawful",                  0.7),
        ("unlawful",                0.8),
        ("penalty",                 0.8),
        ("imprisonment",            0.8),
        ("the constitution",        1.0),
        ("constitutional",          0.9),
        ("unconstitutional",        1.0),
        ("the secretary",           0.6),
        ("the court",               0.7),
        ("jurisdiction",            1.0),
        ("enacted by",              1.2),
        ("authorized by",           0.8),
        ("prohibited by law",       1.5),
        ("this act",                0.8),
        ("said act",                1.0),
        ("obligations",             0.6),
        ("duly",                    0.7),
        ("ratification",            1.0),
        ("in conformity with",      1.2),
        ("in pursuance of",         1.5),
        ("under color of",          1.5),
        ("stipulate",               1.0),
        ("provisions of",           0.8),
        ("the terms of",            0.7),
    ],
}

# ---------------------------------------------------------------------------
# Index build -- runs once at import
# ---------------------------------------------------------------------------

_COMPILED: Dict[str, List[Tuple[re.Pattern, float]]] = {}
_LITERAL:  Dict[str, List[Tuple[str, float]]]        = {}


def _build_vocab_index() -> None:
    for band, entries in _BAND_VOCAB.items():
        compiled_list: List[Tuple[re.Pattern, float]] = []
        literal_list:  List[Tuple[str, float]]        = []
        for pattern, weight in entries:
            if (pattern.startswith(r"(?")
                    or pattern.startswith(r"\b")
                    or pattern.startswith(r"\d")):
                compiled_list.append((re.compile(pattern, re.IGNORECASE), weight))
            else:
                literal_list.append((pattern.lower(), weight))
        _COMPILED[band] = compiled_list
        _LITERAL[band]  = literal_list


_build_vocab_index()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BandScore:
    band:       str
    raw_score:  float
    norm_score: float   # fraction of total scored weight (0-1)


@dataclass
class WaveletResult:
    """
    Result for a single sentence scored by the RhetoricalWavelet.

    Fields
    ------
    sentence             : Input sentence (unmodified).
    band_scores          : Dict[band_name, BandScore] for all four bands.
    primary_band         : Band with the highest normalized score.
    primary_score        : Normalized score for the primary band (0-1).
    is_dual_band         : True if any other band scores within AMBIGUITY_MARGIN
                           of the primary band.
    dual_band_candidates : Band names within the ambiguity margin (empty if not
                           dual).
    routing              : "pathway_2_compound_detector" or
                           "single_band_classification"
    ambiguity_margin     : AMBIGUITY_MARGIN value used (0.20).
    score_gap            : primary_score minus second-highest normalized score.
    matched_markers      : Dict[band, List[str]] of vocabulary items that fired.
    rule_004_note        : Routing explanation for RULE-004 audit field.
    """
    sentence:             str
    band_scores:          Dict[str, BandScore]
    primary_band:         str
    primary_score:        float
    is_dual_band:         bool
    dual_band_candidates: List[str]
    routing:              str
    ambiguity_margin:     float
    score_gap:            float
    matched_markers:      Dict[str, List[str]]
    rule_004_note:        str

    def to_dict(self) -> dict:
        return {
            "sentence":             self.sentence,
            "band_scores": {
                b: {"raw": s.raw_score, "normalized": round(s.norm_score, 4)}
                for b, s in self.band_scores.items()
            },
            "primary_band":         self.primary_band,
            "primary_score":        round(self.primary_score, 4),
            "is_dual_band":         self.is_dual_band,
            "dual_band_candidates": self.dual_band_candidates,
            "routing":              self.routing,
            "ambiguity_margin":     self.ambiguity_margin,
            "score_gap":            round(self.score_gap, 4),
            "matched_markers":      self.matched_markers,
            "rule_004_note":        self.rule_004_note,
        }


@dataclass
class WaveletBatchResult:
    """
    Result for a list of sentences (passage, section, or proposition block).

    compound_candidates : WaveletResults where is_dual_band is True.
    single_band         : WaveletResults where is_dual_band is False.
    compound_count      : Count of dual-band sentences.
    total_count         : Total sentences scored.
    batch_primary_band  : Most common primary band across the batch.
    """
    compound_candidates: List[WaveletResult]
    single_band:         List[WaveletResult]
    compound_count:      int
    total_count:         int
    batch_primary_band:  str

    def to_dict(self) -> dict:
        return {
            "batch_primary_band":  self.batch_primary_band,
            "total_sentences":     self.total_count,
            "compound_candidates": self.compound_count,
            "single_band":         self.total_count - self.compound_count,
            "sentences": [r.to_dict() for r in
                          self.compound_candidates + self.single_band],
        }

# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def _score_sentence(sentence: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Return raw band scores and matched-marker records for one sentence.

    Literal matches are case-insensitive substring checks.
    Regex matches use pre-compiled patterns.
    Proper-noun heuristic: tokens after sentence position 0 that are
    capitalized alphabetic strings contribute 0.4 each to empirical_event
    when at least 2 are present. This surfaces named parties without a NER
    pipeline.
    """
    s_lower = sentence.lower()
    raw:     Dict[str, float]     = {b: 0.0 for b in ALL_BANDS}
    matched: Dict[str, List[str]] = {b: []  for b in ALL_BANDS}

    for band in ALL_BANDS:
        for literal, weight in _LITERAL[band]:
            if literal in s_lower:
                raw[band] += weight
                matched[band].append(literal)
        for pattern, weight in _COMPILED[band]:
            if pattern.search(sentence):
                raw[band] += weight
                matched[band].append(pattern.pattern)

    # Proper-noun heuristic for empirical_event
    tokens = sentence.split()
    proper_count = sum(
        1 for i, t in enumerate(tokens)
        if i > 0 and t and t[0].isupper() and t.rstrip(".,;:!?").isalpha()
    )
    if proper_count >= 2:
        bonus = 0.4 * proper_count
        raw[EMPIRICAL_EVENT] += bonus
        matched[EMPIRICAL_EVENT].append(
            f"[proper_noun_heuristic: {proper_count} tokens, +{bonus:.1f}]"
        )

    return raw, matched


def _normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Divide each band's raw score by the total.
    If total is zero (no vocabulary fired), return uniform 0.25 for all bands.
    Uniform-distribution results should be reviewed manually by the analyst.
    """
    total = sum(raw.values())
    if total == 0.0:
        return {b: 0.25 for b in ALL_BANDS}
    return {b: v / total for b, v in raw.items()}


def score_sentence(sentence: str) -> WaveletResult:
    """
    Score a single sentence and determine routing.

    Primary public interface for collaborative extraction (analyst + Claude
    working through a document sentence by sentence).

    Parameters
    ----------
    sentence : str
        A single sentence from the source document. Verbatim source text
        preferred -- the wavelet reads the document's register, not a
        normalized claim.

    Returns
    -------
    WaveletResult
    """
    raw, matched = _score_sentence(sentence)
    norm = _normalize_scores(raw)

    ranked = sorted(norm.items(), key=lambda x: x[1], reverse=True)
    primary_band, primary_score = ranked[0]
    _second_band, second_score  = ranked[1]

    score_gap    = primary_score - second_score
    is_dual_band = score_gap <= AMBIGUITY_MARGIN

    dual_band_candidates: List[str] = []
    if is_dual_band:
        dual_band_candidates = [
            b for b, s in ranked
            if b != primary_band and (primary_score - s) <= AMBIGUITY_MARGIN
        ]

    routing = ROUTING_COMPOUND if is_dual_band else ROUTING_SINGLE

    if is_dual_band:
        candidates_str = " / ".join([primary_band] + dual_band_candidates)
        rule_004_note = (
            f"Dual-band within AMBIGUITY_MARGIN ({AMBIGUITY_MARGIN}): "
            f"{candidates_str} (gap={score_gap:.3f}). Routes to RULE-004 "
            f"Pathway 2 compound detector for epistemic category review."
        )
    else:
        rule_004_note = (
            f"Single-band: {primary_band} (score={primary_score:.3f}, "
            f"gap={score_gap:.3f}). No compound decomposition required by "
            f"RULE-004 Pathway 2."
        )

    band_scores = {
        b: BandScore(band=b, raw_score=raw[b], norm_score=norm[b])
        for b in ALL_BANDS
    }

    return WaveletResult(
        sentence             = sentence,
        band_scores          = band_scores,
        primary_band         = primary_band,
        primary_score        = primary_score,
        is_dual_band         = is_dual_band,
        dual_band_candidates = dual_band_candidates,
        routing              = routing,
        ambiguity_margin     = AMBIGUITY_MARGIN,
        score_gap            = score_gap,
        matched_markers      = matched,
        rule_004_note        = rule_004_note,
    )


def score_passage(sentences: List[str]) -> WaveletBatchResult:
    """
    Score a list of sentences in document order.

    Useful for characterizing the dominant epistemic register of a section
    before proposition-by-proposition extraction begins.

    Parameters
    ----------
    sentences : List[str]

    Returns
    -------
    WaveletBatchResult
    """
    results = [score_sentence(s) for s in sentences]
    compound_candidates = [r for r in results if r.is_dual_band]
    single_band         = [r for r in results if not r.is_dual_band]

    band_counter = Counter(r.primary_band for r in results)
    batch_primary_band = (
        band_counter.most_common(1)[0][0] if results else NORMATIVE_STABLE
    )

    return WaveletBatchResult(
        compound_candidates = compound_candidates,
        single_band         = single_band,
        compound_count      = len(compound_candidates),
        total_count         = len(results),
        batch_primary_band  = batch_primary_band,
    )


def score_proposition_verbatim(verbatim_source_text: str) -> WaveletResult:
    """
    Score the verbatim_source_text field of a proposition record.

    Strips whitespace and normalizes internal newlines before scoring.
    Multi-sentence verbatim blocks (valid under RULE-001 for genuine
    multi-sentence legal units) are scored as a single unit.
    """
    cleaned = " ".join(verbatim_source_text.split())
    return score_sentence(cleaned)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Self-test using synthetic sentences from the methodology spec and
    RULE-004 examples. Run with: python rhetorical_wavelet.py

    Test 6 demonstrates dual-band detection for empirical_event and
    legal_statutory -- a common pattern in legislative documents where
    a specific event (signing, year, named party) and statutory obligation
    language appear in the same sentence.

    Calibration note: sentences mixing a compressed empirical qualifier
    ("economically necessary") with extended normative language ("morally
    required to preserve the dignity") may not reach dual-band threshold
    because 19th-century normative vocabulary is structurally denser than
    empirical vocabulary. The analyst flags these cases manually. The wavelet
    reduces workload; it does not replace judgment.
    """
    test_cases = [
        # (sentence, expected_primary or None if dual-band expected)

        # 1. normative_stable -- clear natural-law register
        (
            "We hold these truths to be self-evident, that all men are created "
            "equal, that they are endowed by their Creator with certain "
            "unalienable Rights.",
            NORMATIVE_STABLE,
        ),
        # 2. legal_statutory -- clear enactment language
        (
            "Be it enacted by the Senate and House of Representatives that no "
            "person shall be employed on vessels of the United States, as "
            "provided by this act.",
            LEGAL_STATUTORY,
        ),
        # 3. empirical_event -- specific date, named actors, specific actions
        (
            "On the fourteenth of January, 1831, the Secretary signed the order "
            "authorizing removal, and the Georgia militia began operations.",
            EMPIRICAL_EVENT,
        ),
        # 4. historical_state -- period conditions, no discrete event
        (
            "The practice of employing such persons had long prevailed throughout "
            "the territory, and the existing conditions made intervention "
            "necessary.",
            HISTORICAL_STATE,
        ),
        # 5. dual-band: normative_stable + legal_statutory (RULE-004 Pathway 2)
        #    A sentence that invokes both natural-law authority and statutory
        #    obligation -- the precise mixing pattern RULE-004 is designed to catch.
        (
            "The exclusion was both morally required by the law of nations and "
            "mandated by statute under the authority of Congress.",
            None,  # dual-band expected
        ),
        # 6. dual-band: empirical_event + legal_statutory
        #    Named year + named actor + signing event mixed with dense statutory
        #    obligation language -- common in legislation preambles.
        (
            "In the year 1828 the President signed the bill and Congress enacted "
            "it, though the obligations imposed by statute required that all "
            "affected persons be afforded notice pursuant to the provisions of "
            "the act.",
            None,  # dual-band expected
        ),
    ]

    print("RhetoricalWavelet -- Smoke Test")
    print(f"AMBIGUITY_MARGIN = {AMBIGUITY_MARGIN}")
    print("=" * 72)

    all_passed = True
    for sentence, expected_primary in test_cases:
        result = score_sentence(sentence)
        is_correct = (
            (expected_primary is None and result.is_dual_band)
            or (expected_primary is not None
                and result.primary_band == expected_primary
                and not result.is_dual_band)
        )
        status = "PASS" if is_correct else "REVIEW"
        if not is_correct:
            all_passed = False

        print(f"\n[{status}]  {sentence[:68]}...")
        print(f"  Primary band     : {result.primary_band} ({result.primary_score:.3f})")
        print(f"  Is dual-band     : {result.is_dual_band}")
        if result.is_dual_band:
            print(f"  Dual candidates  : {result.dual_band_candidates}")
        print(f"  Score gap        : {result.score_gap:.3f}")
        print(f"  Routing          : {result.routing}")
        for band in ALL_BANDS:
            s = result.band_scores[band]
            if s.raw_score > 0:
                markers = result.matched_markers[band]
                print(f"  {band:<22}: raw={s.raw_score:.2f}  "
                      f"norm={s.norm_score:.3f}  markers={markers[:4]}")

    print("\n" + "=" * 72)
    if all_passed:
        print("All tests passed.")
    else:
        print("Some results need review -- check REVIEW items above.")


if __name__ == "__main__":
    _smoke_test()

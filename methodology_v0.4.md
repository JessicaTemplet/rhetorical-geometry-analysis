# Rhetorical Geometry Analysis — Methodology Document v0.4

## Status
Draft. Not locked. To be locked alongside schema v0.4 before any document analysis begins.

---

## 1. Project Overview

This methodology governs the Rhetorical Geometry Analysis project. The project applies geometric and information-theoretic tools to governmental historical documents to identify the structural shape of the belief space each document constructs. The output is a geometric description, not a verdict. Classification of intent or moral judgment is left to the reader.

---

## 2. Preregistration and Chain of Custody

Analyst priors and the document pool were committed to a public GitHub repository before any document was touched analytically. The public commit timestamp is the tamper-evident proof of the chain of custody.

**Public preregistration repository:** https://github.com/JessicaTemplet/rhetorical-geometry-analysis

The first commit to this repository — made before analysis began — contains:
- The analyst's stated priors across all ten documents
- The document pool (ten documents confirmed by RNG selection from a pool of twenty)
- The preregistration spec
- The schema at the time of first commit

Each document analysis record in the schema carries a `preregistration_commit_hash` field referencing the specific first commit. This replaces the prior SHA-256 hash-plus-custodian mechanism. A public git repository provides equivalent tamper evidence without requiring a human custodian to verify cryptographic hashes. The behavioral commitment to not read the preregistration record during analysis is self-reported and noted explicitly here as such.

---

## 3. Document Pool

Ten documents selected by RNG from a pool of twenty candidates. The pool construction criteria are documented in the preregistration repository. The ten selected documents are listed in `preregistration_record.json` in the repository.

**Pool asymmetry disclosure:** The document pool is weighted toward cases where the likely geometric finding is government rhetoric that diverged from anchored fact. This is partly inherent in the selection criteria — contested documents tend to be contested because the official account is disputed. This asymmetry is named here preemptively rather than left to be discovered. If the methodology works correctly, the geometry will show what it shows without the analyst directing it.

---

## 4. Epistemic Category Classification — Adapted NarrativeWavelet

### 4.1 Purpose

RULE-004 (Compound Proposition Detection, Pathway 2) requires identifying when a single sentence spans more than one epistemic category, because heterogeneous epistemic content in a single proposition is a signal of rhetorical compression that may require decomposition. The adapted NarrativeWavelet is the detection instrument for this.

The NarrativeWavelet architecture was originally developed for fiction continuity tracking, where it separates narrative text into frequency bands by temporal stability. The architecture transfers to rhetorical document analysis by substituting epistemic category for temporal stability as the sorting dimension. The core logic — scoring sentences against vocabulary sets for each band, flagging sentences that score within an ambiguity margin of two bands, and duplicating ambiguous sentences into both candidate bands — is preserved unchanged.

### 4.2 Four-Band Structure

Rhetorical governmental documents require four bands rather than three. The original three-band structure (LOW/MID/HIGH) is insufficient because legal/statutory claims use normative obligation language but are empirically testable against specific text, making them a genuinely distinct epistemic category.

#### Band 1: normative_stable
Claims framed as universally or permanently true, grounded in natural law, inherent right, or self-evident principle. Not testable against a specific document or empirical record — truth is asserted as prior to evidence.

**Marker vocabulary:** necessary, essential, self-evident, natural law, unalienable, inherent, from time immemorial, by the law of nations, it has always been, ordained, fundamental, prohibited by nature. Universal quantifiers (all, every, always, never, none) also score toward this band.

#### Band 2: historical_state
Claims about conditions that prevailed at a particular time — not universal but not a discrete event. Describes a state of affairs that was true during a period, with implied or explicit temporal bounds.

**Marker vocabulary:** had been, at the time, during this period, prevailing conditions, the existing, throughout, domestic, the current, had long, for years, the practice of, was understood to, throughout this period, had always.

#### Band 3: empirical_event
Discrete measurable claims, specific events, statistics, named parties performing named actions, outcomes with identifiable dates or locations. Testable against physical or documentary record.

**Marker vocabulary:** specific dates, hereby, was enacted, pursuant to (when introducing a specific action rather than a general authority), numbers and quantities, named parties performing named actions, as of, reported, recorded, established, resulted in.

#### Band 4: legal_statutory
Claims whose truth is testable against the text of a specific law, treaty, constitutional provision, or statutory authority. Uses obligation language (shall, must, required) but is anchored to a specific named document rather than to natural law or universal principle. Distinguishable from normative_stable by the presence of a specific statutory reference; distinguishable from empirical_event by the obligation structure rather than event structure.

**Marker vocabulary:** shall, pursuant to, in accordance with, as provided by, under the authority of, the act provides, section, article, clause, treaty, statute, congress, hereby ordained, as amended, under existing law.

### 4.3 Ambiguity Margin

`AMBIGUITY_MARGIN = 0.20`

Wider than the fiction-domain default because deliberate epistemic category mixing is a feature of rhetorical documents, not an error. Sentences that score within 0.20 of two band thresholds are flagged as potentially heterogeneous and are duplicated into both candidate band records for Pathway 2 review.

### 4.4 Routing

Sentences flagged as dual-band by the adapted NarrativeWavelet route to the Pathway 2 compound detector with epistemic category priors, not to the SRL enricher. The duplication mechanism — which in the fiction domain routes ambiguous sentences to both temporal bands — here routes to both epistemic categories for independent classification review.

### 4.5 Calibration Note for 19th-Century Documents

In 19th-century governmental documents, stylistic register shifts — particularly transitions from legal-bureaucratic language to biblical-paternalistic language — will trigger vocabulary anomalies in the Zipfian Auditor and may also trigger holonomy drift in the Sheaf Holonomy analysis. These signals should not be read automatically as selective omission (RULE-015).

The distinction between a stylistic register shift and a content omission is located in the vocabulary layer being shifted:

- **Stylistic register shift:** High-frequency function words and genre markers redistribute. Legal terminology gives way to paternalistic or religious register. Content vocabulary — named parties, dates, land descriptions, specific legal obligations — remains present.
- **Content omission:** Domain-specific content vocabulary disappears or thins abruptly. Named parties, geographic specifics, legal obligations, or quantitative claims drop out while the surrounding sections are dense with them. The holonomy drift and Zipfian anomaly co-occur at the same section, and the missing content can be specified by the analyst from the historical record.

RULE-015 (selective_omission) requires that the analyst be able to name what was omitted and where it exists in the historical record. A vocabulary shift that cannot be paired with a documentable omission is a register shift, not a selective omission, and the tag should not be applied.

---

## 5. Anchor Registry — Independent Genesis Requirement

All anchors must carry an `independent_genesis` record (see schema v0.4, anchor_registry). An anchor with no source independent of the administrative chain of command of the document under analysis receives a `genesis_confidence_weight` of `reduced`, not a null or disqualification. This is because institutional correspondence may be the only surviving record for some historical claims, and wholesale disqualification would eliminate anchors that are genuinely the best available evidence.

The reduced weight flag is an honest accounting of the epistemic situation, not a workaround. It must be documented in the anchor record and carries through to the confidence_flags section. Analysts reviewing reduced-weight anchors should apply heightened scrutiny to any proposition that depends on them exclusively.

---

## 6. Bridge Narrative Detection — Signal Weighting (RULE-011 Supplement)

The four signals for bridge narrative classification are not equally weighted. Epistatic suppression (Signal 3) is the load-bearing criterion.

The operational test for bridge narrative classification is counterfactual: remove the candidate node from the graph and measure whether the geodesic distance between the relevant anchor and the relevant false attractor meaningfully shortens. If it does not shorten, the node is not functioning as an insulator regardless of its position in the Poincaré disk.

Boundary position in the Poincaré disk (Signal 1) is a suspicion trigger that elevates scrutiny. It is not a signal that contributes to the four-signal qualifying count. A node may sit at the boundary because it is genuinely complex and touches multiple domains, not because it is insulating a false attractor from a contradiction. Boundary position alone does not distinguish these cases. Epistatic suppression does.

---

## 7. Falsifiability Commitment

Every extraction rule carries a falsifiability record in the schema output (see schema v0.4, falsifiability_record). The record requires:

1. Two uncomfortable outputs — one on each side of the classification error spectrum
2. A kill switch — a single sentence naming the specific observable data point that would force the rule to be declared miscalibrated rather than the document anomalous

The kill switch requirement was added in v0.4 following the recognition that documenting uncomfortable outputs is necessary but not sufficient. The analyst must also commit, before analysis begins, to the specific condition under which they would abandon the rule rather than defend it. A rule without a kill switch is not falsifiable in practice.

---

## 8. Schema Version History

| Version | Status | Key Changes |
|---------|--------|-------------|
| 0.1 | Superseded | Initial draft |
| 0.2 | Superseded | Four open questions resolved |
| 0.3 | Superseded | falsifiability_record added; preregistration spec added; schema locked for first commit |
| 0.4 | Current draft | unattributed_agency added to secondary_tags; relational_proposition convention added; independent_genesis added to anchor_registry; document_metadata updated to public repository approach; kill_switch added to falsifiability_record; NarrativeWavelet adaptation documented; bridge narrative signal weighting documented; 19th-century register shift calibration note added |

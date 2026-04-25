# Kill Switches — RULE-001 through RULE-004, RULE-006 through RULE-020
# (RULE-005, RULE-011, RULE-015 kill switches are in rules_revised_005_011_015.md)
# Status: Draft. Pending incorporation into full rule set before lock commit.

---

## RULE-001: Proposition Bounding

Rule summary: A proposition is the smallest unit of text that makes a single independently evaluable claim. Legislative preambles, purpose clauses, and recitals are excluded from extraction unless they contain a factual claim that is independently evaluable.

Kill switch: If a single sentence that any careful reader would identify as making exactly one claim — with no embedded sub-claims, no purpose clause, and no compound structure — is extracted as two or more propositions by the bounding rule, the minimum-unit definition is miscalibrated and the rule is falsified for that sentence type.

---

## RULE-001B: Conditional and Causal Proposition Extraction

Rule summary: When a document frames an action as the means to an outcome through purpose clauses, causal connectives, or conditional structures, the relational link between action and outcome is itself a separately extractable proposition evaluated against empirical evidence about whether that action type produces that outcome type.

Kill switch: If a relational proposition extracted under this rule (of the form "Action A produces Outcome B") is evaluated against anchors and classified as anchored_true for a historical case where the outcome is independently documented as having not occurred, the anchor mapping logic for relational propositions is broken and the rule is falsified.

---

## RULE-002: Verbatim Source Text Requirement

Rule summary: The verbatim_source_text field must contain an exact quote from the source document with no paraphrase, ellipsis, or condensation. Any deviation is a schema violation.

Kill switch: If the schema validation tool passes a verbatim_source_text field that contains an ellipsis, a bracketed insertion not present in the source document, or any word not appearing in the source text at that location, the validation mechanism is broken and the rule is falsified.

---

## RULE-003: Normalization Standard

Rule summary: The normalized_claim must preserve exact truth conditions of the verbatim source text. Passive constructions with no named agent within the one-sentence attribution window must be normalized to [Unspecified Actor] and flagged with secondary tag unattributed_agency.

Kill switch: If a normalized claim that substitutes a named agent — drawn from institutional context rather than the same sentence or the immediately preceding sentence — passes schema validation without a confidence_flag record, the attribution window enforcement is broken and the rule is falsified.

---

## RULE-004: Compound Proposition Detection

Rule summary: A proposition is compound if it either contains sub-claims that could receive different primary_classifications (Pathway 1) or offers multiple reasons for a single conclusion where those reasons span more than one epistemic category as defined by the adapted NarrativeWavelet (Pathway 2).

Kill switch: If a proposition that offers reasons spanning empirical_quantitative and moral_normative epistemic categories — where the moral claim has no independent external anchor — passes through RULE-004 without triggering Pathway 2 and is classified as a single anchored claim, the heterogeneous evidence detector is broken and the rule is falsified.

---

## RULE-006: Anchor Chain Length Estimation

Rule summary: Every anchor classification other than mathematical_truth requires a chain_length_from_governance_zone integer. If the chain length cannot be estimated even approximately, the claim does not meet the bar for the anchor registry.

Kill switch: If a direct_empirical_record anchor — a signed treaty, a congressional vote count, a census figure — receives a chain_length_from_governance_zone estimate of zero, the chain length calibration is wrong. Direct empirical records are not mathematical truths and cannot have a chain length of zero. If any non-mathematical anchor is assigned zero, the rule is falsified.

---

## RULE-007: Geodesic Distance Computation

Rule summary: Geodesic distance from a proposition to its mapped anchor is computed in the Poincaré disk metric. Propositions with shorter geodesic distances to the governance zone are more stable. Distance is null until the graph computation pass is complete.

Kill switch: If two propositions that map to the same anchor and have identical inferential chain lengths from that anchor receive materially different geodesic distance values in the disk, the distance computation is inconsistent and the rule is falsified.

---

## RULE-008: Edge Sign Assignment

Rule summary: Each edge between a proposition and its mapped anchor receives a sign — positive for support, negative for contradiction, unsigned for neutral contextual relationship. A proposition may not carry both a positive and a negative edge to the same anchor.

Kill switch: If a proposition classified as anchored_true carries a negative edge to its primary anchor, or a proposition classified as anchored_false carries a positive edge to its primary anchor, the edge sign assignment is inconsistent with the primary classification and the rule is falsified.

---

## RULE-009: Stress Energy Contribution

Rule summary: Each proposition's T_μν contribution is computed from its geodesic position, edge sign, and bidirectional load. Propositions at the boundary zone with bidirectional load carry the highest stress energy. T_μν is null until the field computation pass is complete.

Kill switch: If a proposition classified as anchored_true with a short geodesic distance to the governance zone and a single positive edge to its anchor receives a T_μν value in the upper quartile of the document's stress energy distribution, the stress energy computation is miscalibrated and the rule is falsified.

---

## RULE-010: Manifold Zone Assignment

Rule summary: Propositions are assigned to one of four Poincaré zones — governance, stable, contested, boundary — based on their geodesic distance from r=0. Mathematical truth anchors sit at r=0. Zone boundaries are set relative to the document's anchor cluster distribution.

Kill switch: If all propositions in a document are assigned to the governance or stable zones with none in the contested or boundary zones, and the document is one of the ten preregistered historical documents known to be contested, the zone boundary calibration is wrong and the rule is falsified. A genuinely contested historical document must produce propositions in the contested or boundary zones.

---

## RULE-012: False Attractor Identification

Rule summary: A false attractor is a proposition classified as anchored_false that has a cluster of inferentially_true propositions in its geodesic neighborhood — propositions that derive apparent truth from proximity to the false anchor rather than from independent anchoring.

Kill switch: If a proposition classified as anchored_false has no propositions in its geodesic neighborhood with positive edges pointing toward it, it cannot function as an attractor and does not qualify as a false attractor under this rule. If the rule assigns false_attractor status to an isolated anchored_false node with no dependent neighborhood, the attractor detection logic is broken and the rule is falsified.

---

## RULE-013: Fragmentation Detection

Rule summary: A document manifold is flagged fragmentation_detected = true when the proposition graph resolves into two or more incompatible metric spaces — regions whose anchor sets are mutually contradictory such that no single consistent Poincaré geometry can contain both.

Kill switch: If a document whose anchor registry contains no contradictory anchors — where all registered anchors are mutually consistent — is flagged fragmentation_detected = true, the fragmentation detection algorithm is finding structure that does not exist in the anchor set and the rule is falsified.

---

## RULE-014: Polarization Detection

Rule summary: A document manifold is flagged polarization_detected = true when two distinct position clusters exist within a single shared manifold — meaning the anchors are not contradictory but the propositions organize into two geodesically separated groups that do not bridge.

Kill switch: If a document flagged polarization_detected = true has a bridge_narrative candidate whose removal causes the two clusters to merge into a single connected region, that node is a bridge narrative and the polarization is structural rather than genuine. If removing a single node resolves the polarization, the rule should have routed to bridge narrative detection rather than polarization detection, and the routing logic is miscalibrated.

---

## RULE-016: False Precision

Rule summary: The secondary tag false_precision is applied to empirical_quantitative propositions that state a specific quantity where the source document's evidentiary basis for that precision is either unstated or demonstrably weaker than the stated precision implies. The Kolmogorov Complexity Tax is the primary detection tool.

Kill switch: If the Kolmogorov Complexity Tax flags a claim as false_precision where the source document explicitly cites the measurement methodology and margin of error for the stated figure, the compression heuristic is firing on a well-evidenced precise claim and the rule is falsified for that claim type.

---

## RULE-017: Bridge Adjacent

Rule summary: The secondary tag bridge_adjacent is applied to a proposition whose primary_classification depends on the truth of a proposition that has been classified as bridge_narrative. If the bridge narrative is removed from the graph, the bridge_adjacent proposition's classification would change.

Kill switch: If a proposition tagged bridge_adjacent has an independent anchor mapping — one that does not pass through the bridge narrative — and its classification under that independent mapping alone is identical to its classification with the bridge narrative present, then the proposition's classification does not depend on the bridge and the tag is incorrectly applied. If the independent path produces the same classification, the dependency condition is not met and the rule is falsified for that proposition.

---

## RULE-018: Bridge Narrative as Secondary Tag

Rule summary: The secondary tag bridge_narrative is applied to a proposition whose primary_classification is not bridge_narrative but which satisfies Signal 3 (epistatic suppression) plus at least two of Signals 2 and 4 as defined in RULE-011.

Kill switch: If a proposition receives the bridge_narrative secondary tag but its removal from the graph does not meaningfully shorten the geodesic distance between any registered anchor and any false attractor, Signal 3 has not been satisfied and the tag is incorrectly applied. Signal 3 is the load-bearing criterion. Its failure falsifies any bridge classification, primary or secondary.

---

## RULE-019: Decomposition Trigger and Execution

Rule summary: When a proposition is identified as compound by Pathway 1 or Pathway 2 of RULE-004, or flagged as dual-band by the adapted NarrativeWavelet, decomposition is required before classification proceeds. Decomposition must be minimal — the smallest number of sub-propositions that resolves the heterogeneity.

Kill switch: If decomposition of a compound proposition produces sub-propositions that are not each independently classifiable — meaning classifying one sub-proposition requires knowing the classification of a sibling — the decomposition did not resolve the compound structure and must be redone. If a decomposition pass is accepted by the schema with epistemically entangled sub-propositions, the independence validation is broken and the rule is falsified.

---

## RULE-020: Sub-Proposition Independence Requirement

Rule summary: Each sub-proposition produced by decomposition must be independently evaluable — its classification must not change based on the classification of any sibling sub-proposition from the same parent. Shared anchors do not violate independence. Logical entanglement does.

Kill switch: If two sub-propositions from the same decomposition are each individually classified, and then reversing the classification of one causes the correct classification of the other to change, the two are logically entangled and independence was not achieved. If this condition is detected after a decomposition pass has been committed, the decomposition is invalid and the rule is falsified for that parent proposition.

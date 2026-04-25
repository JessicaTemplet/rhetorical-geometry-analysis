# Rhetorical Geometry Analysis — Extraction and Classification Rules v1.0
# Status: Locked for second commit. Do not modify without incrementing version.
#
# Structure: Five layers, 20 numbered rules plus RULE-001B.
# Each rule contains: rule text, uncomfortable output directions A and B, kill switch.
# RULE-005, RULE-011, and RULE-015 incorporate v0.4 revisions.

---

# Layer 1: Proposition Extraction Rules

## RULE-001: Proposition Bounding

A proposition is the smallest unit of text that makes a single independently evaluable claim. The bounding unit is the claim, not the sentence. A single sentence may contain multiple propositions. A proposition may not span multiple sentences unless the sentences together constitute a single logical unit that cannot be evaluated without both.

Legislative preambles, purpose clauses, and recitals are excluded from proposition extraction unless they contain a factual claim that is independently evaluable against a registered anchor. A recital that states a historical condition as justification for a legislative act is extractable. A recital that merely states legislative intent without asserting a factual condition is not.

Uncomfortable output direction A: A complex sentence that is genuinely a single integrated claim — where the sub-clauses are all logical components of one assertion — gets split into multiple propositions, each of which is weaker and less precisely stated than the original. The bounding rule fragments a coherent claim into pieces that individually misrepresent what the document asserts.

Uncomfortable output direction B: A sentence containing two independently evaluable claims — one true and one false — is treated as a single proposition. The true sub-claim absorbs the false one, the compound form never triggers RULE-004, and the false claim is never classified.

Kill switch: If a single sentence that any careful reader would identify as making exactly one claim — with no embedded sub-claims, no purpose clause, and no compound structure — is extracted as two or more propositions by the bounding rule, the minimum-unit definition is miscalibrated and the rule is falsified for that sentence type.

---

## RULE-001B: Conditional and Causal Proposition Extraction

When a document frames an action as the means to an outcome — through purpose clauses, causal connectives, or conditional structures — the relational link between action and outcome is itself a separately extractable proposition. The form is: [Action A] produces [Outcome B]. This relational proposition has its own truth value, independent of whether Action A occurred and independent of whether Outcome B was achieved. It is evaluated against empirical evidence about whether A-type actions actually produced B-type outcomes in analogous historical contexts.

Do not subsume the relational proposition into either the action claim or the outcome claim. The causal link is where bridge narratives most commonly operate — the action and the stated outcome may each be factually grounded while the causal connection between them is false or unsupported.

Synthetic example:
Verbatim: "In order to protect Group A, Authority Z shall relocate them to Territory Y."
Extract three propositions:
P-001: Authority Z shall relocate Group A to Territory Y. (action claim)
P-002: Relocation of Group A to Territory Y will protect Group A. (outcome claim)
P-003: Relocation is the means by which protection will be achieved. (relational proposition)
P-003 is anchored against historical and empirical records of whether relocation-type actions produced protective outcomes for comparable groups. P-001 and P-002 can each be true while P-003 is false.

Uncomfortable output direction A: A standard procedural purpose clause ("In order to ensure orderly administration, the Secretary shall file quarterly reports") gets extracted as a relational proposition "quarterly filing produces orderly administration," which is too granular and procedural to be meaningfully anchored. The rule generates noise from routine administrative conditionals that carry no rhetorical weight.

Uncomfortable output direction B: A purpose clause is categorized as legislative intent and excluded from extraction entirely, letting the causal link between action and outcome — which is the entire bridge narrative — never enter the proposition graph. The relational proposition is the lie and it disappears before analysis begins.

Kill switch: If a relational proposition extracted under this rule is evaluated against anchors and classified as anchored_true for a historical case where the outcome is independently documented as having not occurred, the anchor mapping logic for relational propositions is broken and the rule is falsified.

---

## RULE-002: Verbatim Source Text Requirement

The verbatim_source_text field must contain an exact quote from the source document. No paraphrase. No ellipsis. No condensation. No bracketed insertions not present in the source. Every word in the field must appear in the source document at that location. Any deviation is a schema violation.

If a proposition spans a passage too long to quote in full, the verbatim_source_text contains the minimal continuous passage that establishes the claim, with source_location specifying the full range. No compression of the passage is permitted.

Uncomfortable output direction A: A proposition whose verbatim source text requires quoting a very long passage — a full paragraph of 19th-century legal prose — receives a truncated verbatim field because the analyst judges the full quote unwieldy. The schema violation is silent and the auditable link between the normalized claim and the source text is broken.

Uncomfortable output direction B: A verbatim_source_text that contains a minor transcription error — a word changed, a comma omitted — passes visual review and is committed. The normalized_claim is accurate to the document's intent but the verbatim field is not verbatim. The audit trail shows a clean record for a proposition whose source link is quietly wrong.

Kill switch: If the schema validation tool passes a verbatim_source_text field that contains an ellipsis, a bracketed insertion not present in the source document, or any word not appearing in the source text at that location, the validation mechanism is broken and the rule is falsified.

---

## RULE-003: Normalization Standard

The normalized_claim must preserve the exact truth conditions of the verbatim_source_text. Permitted operations: clarify ambiguous referents using explicitly named agents, expand pronouns where the referent is named in the same sentence, convert passive to active where the agent is explicitly named in the same sentence or the immediately preceding sentence. Prohibited operations: add information not present in the source text, omit qualifiers that are part of the truth conditions, convert stated grounds framing into direct factual claims, infer an agent from institutional context when no specific actor is named.

Non-Attribution Flag: If the agent of a claim is not explicitly named within the same sentence or the immediately preceding sentence, the normalization must preserve the passive construction and substitute [Unspecified Actor] in agent position. The resulting proposition is flagged with secondary tag unattributed_agency. This surfaces a geometric hole: a claim floating in the manifold with no declared edge to any governance or identity node.

Synthetic example — correctly flagged:
Verbatim: "It was determined that relocation was necessary for the welfare of the affected population."
Normalized: "[Unspecified Actor] determined that relocation was necessary for the welfare of [Group A]."
Secondary tag: unattributed_agency
Do not normalize to a named institutional actor even if context strongly implies one. The absence of named attribution is itself a geometric property of the document.

Synthetic example — correctly attributed:
Verbatim: "The Secretary determined that the measure was justified."
Normalized: "[The Secretary] determined that [the measure] was justified."
No non-attribution flag. Agent is explicitly named in the same sentence.

Uncomfortable output direction A: Standard bureaucratic passive voice where institutional attribution is conventional ("It is hereby ordered that...") gets flagged as [Unspecified Actor] even though the issuing authority is named in the document header and unambiguous in context. The geometric hole is an artifact of the one-sentence context window, not a real absence of attribution.

Uncomfortable output direction B: An agent named two sentences prior — outside the one-sentence context window — still produces [Unspecified Actor], creating a false geometric hole where real attribution exists and is recoverable by any careful reader.

Kill switch: If a normalized claim that substitutes a named agent — drawn from institutional context rather than the same sentence or the immediately preceding sentence — passes schema validation without a confidence_flag record, the attribution window enforcement is broken and the rule is falsified.

---

## RULE-004: Compound Proposition Detection

A proposition is compound if it satisfies either of the following two detection pathways. Satisfying either one is sufficient.

Pathway 1 — Independent evaluability: The proposition contains two or more sub-claims that can each be independently evaluated against registered anchors and could receive different primary_classifications from each other.

Pathway 2 — Heterogeneous evidence: The proposition offers multiple reasons for a single conclusion and those reasons belong to different epistemic categories, regardless of whether they could independently receive different primary_classifications. If reasons are epistemically heterogeneous, they must be decomposed before classification. A moral claim must not be allowed to share the spectral stiffness of an empirical anchor it has no independent right to.

Defined epistemic categories for Pathway 2:
- empirical_quantitative — testable against measurable records, statistics, or physical evidence
- legal_statutory — testable against statutory text, treaty language, or legal precedent
- moral_normative — a value judgment; not independently testable against an external anchor without first converting it to a legal or empirical claim
- historical_narrative — a claim about what occurred; testable against primary source records of events
- predictive_causal — a claim about what will happen or what causes what; testable against analogous historical outcomes

Detection instrument: The adapted NarrativeWavelet (see methodology document Section 4) scores sentences against four band vocabularies — normative_stable, historical_state, empirical_event, legal_statutory — and flags sentences scoring within AMBIGUITY_MARGIN = 0.20 of two bands as dual-band candidates. Dual-band sentences route to the Pathway 2 compound detector.

Synthetic example — correctly identified compound by Pathway 2:
"The exclusion of Group A was economically necessary [empirical_quantitative] and morally required to preserve the dignity of domestic workers [moral_normative]."
One conclusion. Two reasons from different epistemic categories. Compound. Decompose before classification.

Synthetic example — correctly identified non-compound:
"The policy was legally required because it fulfilled the treaty obligations and conformed to the statutory mandate."
One conclusion. Two reasons. Both legal_statutory. Same epistemic category. Not compound by Pathway 2.

Uncomfortable output direction A: A coherent integrated argument where empirical outcomes genuinely ground a moral obligation gets forcibly decomposed, destroying the argumentative structure and producing two weaker sub-propositions neither of which captures the actual claim.

Uncomfortable output direction B: A reason that appears to be moral_normative on its surface is actually a legal claim in disguise ("morally required because it conforms to the treaty") and gets split into the wrong epistemic category, misclassifying a potentially anchored_true claim as unanchorable.

Kill switch: If a proposition that offers reasons spanning empirical_quantitative and moral_normative epistemic categories — where the moral claim has no independent external anchor — passes through RULE-004 without triggering Pathway 2 and is classified as a single anchored claim, the heterogeneous evidence detector is broken and the rule is falsified.

---

# Layer 2: Anchor Registry Rules

## RULE-005: Anchor Validation and Independent Genesis

An anchor is valid if it meets all three conditions: it falls within one of the four anchor classification categories (mathematical_truth, direct_empirical_record, primary_source_attestation, scientific_consensus); it has a chain_length_from_governance_zone that can be estimated as an integer; and it carries an independent_genesis record.

An anchor must document whether at least one source establishing it originates outside the administrative chain of command of the document under analysis. Repetition within a single institutional chain does not constitute independent corroboration. A lie told consistently across all contemporary governmental correspondence from the same administration passes the volume test without passing the truth test. Institutional weight is not independent genesis.

The three genesis confidence weights and their consequences:

full — At least one source establishing this anchor originates entirely outside the administrative chain of command. The anchor carries full evidential weight.

reduced — All available sources originate within the same administrative chain of command. The anchor is not disqualified. Its reduced weight propagates to any proposition that maps to it exclusively. The absence of independent genesis must be documented in the genesis_note field. A confidence_flag record is required.

minimal — Provenance is unclear. A confidence_flag record is required. The anchor may not serve as the sole mapping for any proposition classified as anchored_true or anchored_false without analyst justification in the notes field.

Repetition alone does not raise a reduced anchor to full. The independence of the source is what matters, not the volume of attestation.

Uncomfortable output direction A: An anchor drawn from a scientific consensus document produced entirely within a government research program receives reduced confidence weight because its genesis is institutional. The rule targets administrative chain-of-command dependency, not institutional origin broadly. Scientific consensus anchors should be evaluated against whether the consensus itself was independently corroborated across the research community.

Uncomfortable output direction B: A coordinated falsehood repeated across twenty documents from the same administration receives full genesis confidence weight because an analyst locates a newspaper report that uncritically reproduces the official claim without independent verification. The newspaper is outside the administrative chain of command but is not an independent source of the underlying fact. The rule checks source origin, not source independence of verification.

Kill switch: If a mathematical truth — a claim formally provable without any empirical record — fails the independent genesis check because its only documentation is a government-produced mathematical table, the independent genesis requirement is miscalibrated and the rule is falsified for that anchor classification.

---

## RULE-006: Anchor Chain Length Estimation

Every anchor classification other than mathematical_truth requires a chain_length_from_governance_zone integer estimated by the analyst. Mathematical truths receive chain length 0 by definition. All other classifications require an integer — estimation is required, null is not permitted. If the chain length cannot be estimated even approximately, the claim does not meet the bar for the anchor registry and is disqualified.

chain_length_confidence must be set to exact, estimated, or contested. Scientific consensus anchors always receive estimated. Direct empirical records may receive exact if the inferential chain is short and unambiguous.

Uncomfortable output direction A: An analyst assigns chain_length = 1 to a direct_empirical_record anchor that in fact requires several inferential steps to connect to the mathematical governance zone. The underestimate makes the anchor appear closer to mathematical certainty than it is, artificially stabilizing every proposition that maps to it.

Uncomfortable output direction B: An analyst assigns a very high chain_length to a well-evidenced direct empirical record out of excessive caution, pushing propositions that depend on it into the contested zone when they would be correctly classified as anchored_true at a more accurate chain length. Conservative estimation produces false ambiguity.

Kill switch: If a direct_empirical_record anchor — a signed treaty, a congressional vote count, a census figure — receives a chain_length_from_governance_zone estimate of zero, the chain length calibration is wrong and the rule is falsified. Direct empirical records are not mathematical truths and cannot have a chain length of zero.

---

# Layer 3: Graph Construction Rules

## RULE-007: Geodesic Distance Computation

Geodesic distance from a proposition to its mapped anchor is computed in the Poincaré disk metric. The disk places the mathematical governance zone at r=0. Propositions with shorter geodesic distances to the governance zone are more stable. Distance is null until the graph computation pass is complete and must not be estimated or approximated in the extraction pass.

Uncomfortable output direction A: Two propositions that map to different anchors at different chain lengths receive geodesic distances that are compared directly, implying one proposition is more stable than the other when in fact the difference reflects the anchor chain lengths rather than the propositions' own inferential distance from their anchors. The metric conflates anchor stability with propositional stability.

Uncomfortable output direction B: A proposition that maps to multiple anchors receives a geodesic distance computed from only its nearest anchor, making it appear more stable than it is. If the nearest anchor is later disqualified, the proposition's geodesic position changes dramatically. Single-anchor geodesic computation is fragile when multiple anchor mappings exist.

Kill switch: If two propositions that map to the same anchor and have identical inferential chain lengths from that anchor receive materially different geodesic distance values in the disk, the distance computation is inconsistent and the rule is falsified.

---

## RULE-008: Edge Sign Assignment

Each edge between a proposition node and its mapped anchor node receives a sign. Positive edges indicate the proposition is supported by the anchor. Negative edges indicate the proposition is contradicted by the anchor. Unsigned edges indicate a neutral contextual relationship where the anchor is relevant but neither supports nor contradicts the proposition.

A proposition may not carry both a positive and a negative edge to the same anchor. If a proposition is supported by one aspect of an anchor and contradicted by another, the anchor must be split into two separate anchor records before edge assignment proceeds.

Uncomfortable output direction A: A proposition that is partially supported and partially contradicted by a single anchor receives an unsigned edge, which hides the contradiction. The analyst opts for unsigned rather than splitting the anchor, and the negative relationship never appears in the graph.

Uncomfortable output direction B: An anchor that is relevant to a proposition but only distantly — through several inferential steps — receives a positive edge when the actual relationship is inferentially_true rather than directly supported. The edge sign overstates the directness of the support relationship.

Kill switch: If a proposition classified as anchored_true carries a negative edge to its primary anchor, or a proposition classified as anchored_false carries a positive edge to its primary anchor, the edge sign assignment is inconsistent with the primary classification and the rule is falsified.

---

## RULE-009: Stress Energy Contribution

Each proposition's T_μν contribution is computed from its geodesic position, edge sign, and bidirectional load. A proposition that carries load toward the governance zone and load toward a false attractor simultaneously has the highest stress energy contribution. T_μν is null until the field computation pass is complete.

Stress energy is a property of the graph, not of the proposition in isolation. A proposition's T_μν value depends on what other nodes are in its neighborhood. T_μν values must not be computed incrementally as propositions are added — the full graph must be present before field computation begins.

Uncomfortable output direction A: A proposition that is a genuine load-bearing claim — one that many other propositions depend on — receives a high T_μν value even though it is anchored_true and sits close to the governance zone. High stress energy is read as a bridge signal when the proposition is simply important and well-evidenced. Stress energy measures load, not deception.

Uncomfortable output direction B: A bridge narrative that operates through a long inferential chain rather than direct geodesic proximity to the boundary zone carries a low T_μν value because its stress load is distributed across multiple intermediate nodes. The bridge is structurally significant but individually low-stress.

Kill switch: If a proposition classified as anchored_true with a short geodesic distance to the governance zone and a single positive edge to its anchor receives a T_μν value in the upper quartile of the document's stress energy distribution, the stress energy computation is miscalibrated and the rule is falsified.

---

## RULE-010: Manifold Zone Assignment

Propositions are assigned to one of four Poincaré zones based on geodesic distance from r=0:

governance — closest to r=0; propositions directly supported by mathematical truths or very short anchor chains
stable — moderate distance; propositions supported by well-evidenced anchors with longer chains
contested — high distance or bidirectional stress; propositions whose anchor mappings are weak, estimated, or contradicted
boundary — outermost zone; propositions with bidirectional load, near false attractors, or structurally ambiguous

Zone boundaries are set relative to the document's anchor cluster distribution, not as fixed radii. The governance zone boundary is calibrated so that the mathematical truth anchors sit at r=0 and the outermost uncontested direct empirical records sit near the governance-stable boundary.

Uncomfortable output direction A: A document with very few registered anchors produces a compressed anchor cluster near r=0, pushing all propositions into the contested or boundary zones regardless of their actual evidential status. Zone assignment becomes a function of anchor density rather than propositional stability.

Uncomfortable output direction B: A document with many strong anchors produces a wide governance and stable zone, pulling genuinely contested propositions into the stable zone because the anchor distribution is dense. High anchor density masks real contestation.

Kill switch: If all propositions in a document are assigned to the governance or stable zones with none in the contested or boundary zones, and the document is one of the ten preregistered historical documents known to be contested, the zone boundary calibration is wrong and the rule is falsified.

---

# Layer 4: Classification Rules

## RULE-011: Bridge Narrative Detection

A proposition receives the primary classification bridge_narrative when it satisfies the epistatic suppression criterion (Signal 3) plus at least two of Signals 2 and 4. Signal 3 is the load-bearing criterion and must be satisfied for classification to proceed.

Signal 1 — Boundary position: The proposition occupies the boundary zone of the Poincaré disk. This signal raises suspicion and triggers closer examination. It does not contribute to the signal count and does not independently qualify a proposition as a bridge narrative. It is a prior, not a signal.

Signal 2 — Bidirectional stress: The proposition carries T_μν contributions in both directions — toward the governance zone and toward a false attractor — indicating it is load-bearing in both directions simultaneously.

Signal 3 — Epistatic suppression (load-bearing criterion): Removing this node from the graph meaningfully shortens the geodesic distance between a registered anchor and a false attractor. This is the counterfactual test. If removal does not shorten that distance, the proposition is not functioning as an insulator regardless of its position on the disk. Signal 3 must be satisfied for bridge_narrative classification to proceed.

Signal 4 — Inferential dependency: At least one anchored_false proposition has this proposition in its anchor mapping chain. The false attractor depends on the bridge remaining in place.

Classification requires Signal 3 plus at least two of Signals 2 and 4.

Uncomfortable output direction A: A genuinely complex historical claim that is true and touches both a well-anchored fact and a contested causal interpretation satisfies Signal 3 because its removal slightly shortens one geodesic path. It receives bridge_narrative classification when it is simply a claim that multiple arguments depend on because it is important and true. The counterfactual test catches structural dependency but cannot distinguish deliberate placement from incidental importance.

Uncomfortable output direction B: A deliberately placed bridge narrative is distributed across two adjacent propositions. Each individually fails Signal 3 — neither alone, when removed, meaningfully shortens the geodesic distance between anchor and false attractor. Together their removal does. The rule evaluates nodes individually and misses distributed bridges.

Kill switch: If a known non-contentious historical fact — such as the ratification date of a treaty — triggers Signal 3 and satisfies the full classification threshold, the manifold zone assignments are miscalibrated and this rule is falsified. The governance zone boundary is in the wrong place if uncontested facts are registering as bridge narratives.

---

## RULE-012: False Attractor Identification

A false attractor is a proposition classified as anchored_false that has a cluster of inferentially_true propositions in its geodesic neighborhood — propositions that derive apparent truth from proximity to the false claim rather than from independent anchoring. The false attractor is structurally significant because it organizes a neighborhood of apparently true propositions around itself.

A false attractor must have at least two inferentially_true propositions in its geodesic neighborhood with positive edges pointing toward it to qualify. An isolated anchored_false proposition with no dependent neighborhood is not a false attractor — it is simply a false claim.

Uncomfortable output direction A: A true proposition that is heavily contested — one that sits far from the governance zone not because it is false but because its anchor chain is long and estimated — is misclassified as anchored_false and becomes a false attractor, organizing a neighborhood of propositions that are in fact correctly inferentially true around a misclassified anchor.

Uncomfortable output direction B: A genuine false attractor has its neighborhood of inferentially_true propositions distributed across multiple geodesic clusters rather than concentrated in one neighborhood. The clustering threshold is not met for any single false claim even though the cumulative false attractor effect is significant.

Kill switch: If a proposition classified as anchored_false has no propositions in its geodesic neighborhood with positive edges pointing toward it and is assigned false_attractor status, the attractor detection logic is broken and the rule is falsified.

---

## RULE-013: Fragmentation Detection

The document manifold is flagged fragmentation_detected = true when the proposition graph resolves into two or more incompatible metric spaces — regions whose anchor sets are mutually contradictory such that no single consistent Poincaré geometry can contain both. Fragmentation is distinct from polarization: fragmentation means the anchors themselves are contradictory; polarization means the anchors are consistent but the propositions organize into separated clusters.

Uncomfortable output direction A: Two anchor sets that are not logically contradictory but are empirically unrelated — covering entirely different subject matters — produce a fragmented graph because the geodesic distances between their respective proposition clusters are large. The graph fragments not because of contradiction but because of subject-matter distance. The flag fires on topical separation, not logical incompatibility.

Uncomfortable output direction B: Two anchor sets that are genuinely contradictory — one asserting and one denying the same historical fact — do not produce flagged fragmentation because the graph computation finds a path connecting them through a bridge narrative. The bridge narrative prevents the metric incompatibility from being detected, which is precisely what bridge narratives do. Fragmentation is suppressed by the structure the methodology is designed to find.

Kill switch: If a document whose anchor registry contains no contradictory anchors — where all registered anchors are mutually consistent — is flagged fragmentation_detected = true, the fragmentation detection algorithm is finding structure that does not exist in the anchor set and the rule is falsified.

---

## RULE-014: Polarization Detection

The document manifold is flagged polarization_detected = true when two distinct position clusters exist within a single shared manifold — the anchors are not contradictory but the propositions organize into two geodesically separated groups that do not bridge. Polarization indicates the document constructs a belief space with two stable attractors that are not in direct logical contradiction but are structurally isolated from each other.

Uncomfortable output direction A: A document that covers two genuinely unrelated policy areas produces two geodesically separated proposition clusters that are flagged as polarization when the separation is topical rather than rhetorical. The two clusters are not two positions on the same question — they are two different questions in the same document.

Uncomfortable output direction B: A polarized manifold with a single bridge narrative connecting the two clusters does not get flagged because the bridge maintains geodesic connectivity. The polarization is real — there are two stable attractor positions — but the bridge narrative suppresses the detection signal by keeping the clusters formally connected.

Kill switch: If a document flagged polarization_detected = true has a bridge_narrative candidate whose removal causes the two clusters to merge into a single connected region, the polarization is structural rather than genuine and the routing logic should have directed to bridge narrative detection. If removing a single node resolves the polarization, the rule is miscalibrated and is falsified for that document.

---

## RULE-015: Selective Omission

The secondary tag selective_omission is applied when all three of the following conditions are met: the proposition is true as stated; the Sheaf Holonomy analysis identifies a major semantic drift (max_drift_index) at or near the section containing this proposition; and the Zipfian Auditor simultaneously flags a content vocabulary anomaly in the same section. Additionally, the analyst must be able to specify in the proposition record's notes field what the omitted context is and where it exists in the historical record. All conditions are required. Neither signal alone is sufficient. Both signals without a documentable omission are not sufficient.

The Zipfian Auditor must be calibrated to detect anomalies in domain-specific content vocabulary rather than in the full Zipfian distribution. 19th-century governmental documents routinely shift between legal-bureaucratic and biblical-paternalistic registers as a stylistic convention of the era. These register shifts trigger full-distribution vocabulary anomalies that are not selective omissions.

Operational distinction:
A stylistic register shift redistributes high-frequency function words and genre markers while leaving content vocabulary — named parties, geographic specifics, legal obligations, quantitative claims — at normal density.
A content omission thins or drops domain-specific content vocabulary abruptly while the surrounding sections remain dense with it. The holonomy drift and content-vocabulary anomaly co-occur at the same section boundary. The analyst can name what is missing and where it exists in the external historical record.

Uncomfortable output direction A: A section of the Indian Removal Act shifts from legal-bureaucratic to biblical-paternalistic language — a genuine stylistic register shift. Under full-distribution Zipfian calibration, this triggers a vocabulary anomaly. The Sheaf Holonomy analysis registers drift at the same section boundary. The tag is applied when no content has been omitted. Content-vocabulary calibration is designed to prevent this, but if the analyst does not correctly separate function-word redistribution from content-word thinning, the false positive survives.

Uncomfortable output direction B: A genuine content omission is spread gradually across a document rather than concentrated at a section boundary. The holonomy max_drift_index does not fire on any single section because the drift is incremental. The Zipfian content-vocabulary anomaly is similarly diffuse. Neither signal fires at threshold. The tag is not applied even though the cumulative omission is analytically significant.

Kill switch: If a section of a document that has been independently verified as complete — meaning historians have confirmed no material was omitted at that point — triggers both the max_drift_index and the Zipfian content-vocabulary anomaly at threshold, the signal calibration is wrong and this rule is falsified for that document type.

---

## RULE-016: False Precision

The secondary tag false_precision is applied to propositions in the empirical_quantitative epistemic category that state a specific quantity, rate, percentage, or numerical claim where the source document's evidentiary basis for that precision is either unstated or demonstrably weaker than the stated precision implies. A claim stated to three significant figures when the underlying data supports only one significant figure is false precision. A claim stated as a specific ratio when the underlying record is a range is false precision.

The Kolmogorov Complexity Tax is the primary detection tool. A claim that requires significantly more description to state precisely than to state accurately — where the precise form is substantially more complex than a direct statement of the underlying evidence — is a candidate for this tag.

Note: this tag applies to the epistemic status of the precision, not to whether the number is correct. A correct number stated with more precision than the evidence supports is still false_precision.

Uncomfortable output direction A: A number stated in a primary source document that matches the contemporary best estimate — stated precisely because the authors had access to data we no longer have — gets tagged false_precision because the surviving evidentiary record does not support the precision. The tag imposes a modern evidentiary standard on a historical claim.

Uncomfortable output direction B: A number that was fabricated or extrapolated far beyond the evidence does not trigger the Kolmogorov Complexity Tax because the compression ratio of a round number is low. "One million" compresses very efficiently even if the actual count was never measured. The tag is not applied to round-number fabrications.

Kill switch: If the Kolmogorov Complexity Tax flags a claim as false_precision where the source document explicitly cites the measurement methodology and margin of error for the stated figure, the compression heuristic is firing on a well-evidenced precise claim and the rule is falsified for that claim type.

---

## RULE-017: Bridge Adjacent

The secondary tag bridge_adjacent is applied to a proposition that is not itself a bridge narrative — it does not satisfy the Signal 3 requirement of RULE-011 — but whose primary_classification depends on the truth of a proposition that has been classified as bridge_narrative either as a primary or secondary tag. If the bridge narrative is removed from the graph, the bridge_adjacent proposition's classification would change. This is the definition of structural dependency on a bridge.

This tag is computed after bridge narrative classification is complete. It cannot be assigned in the same pass as bridge narrative detection because it requires knowing which propositions are bridges before identifying which propositions depend on them. Multi-hop dependency — where the dependency runs through one or more intermediate propositions — still qualifies. The hop count should be noted in the proposition record.

Uncomfortable output direction A: A proposition that has an independent anchor mapping — one that does not pass through the bridge narrative — still receives bridge_adjacent because it also has a secondary anchor path that does pass through it. The tag is technically correct by the dependency definition but overstates the structural reliance on the bridge.

Uncomfortable output direction B: A proposition's classification depends on a bridge narrative through two intermediate steps rather than one. The bridge_adjacent tag requires only that classification depends on a bridge, not that the dependency is direct. Multi-hop dependency is still dependency. This tag should fire on indirect dependents but the hop count should be noted.

Kill switch: If a proposition tagged bridge_adjacent has an independent anchor mapping that does not pass through the bridge narrative, and its classification under that independent mapping alone is identical to its classification with the bridge narrative present, the dependency condition is not met and the tag is incorrectly applied. If the independent path produces the same classification, the rule is falsified for that proposition.

---

## RULE-018: Bridge Narrative as Secondary Tag

The secondary tag bridge_narrative is applied to a proposition whose primary_classification is not bridge_narrative but which satisfies Signal 3 (epistatic suppression) plus at least two of Signals 2 and 4 as defined in RULE-011. The proposition has a classifiable primary identity while simultaneously functioning structurally as a load-bearing insulator between a false attractor and the governance zone.

This is the case of the technically true bridge: a true claim placed and framed to prevent a direct contradiction between an anchor and a false attractor from becoming visible. It is analytically more important than the primary bridge_narrative classification in some respects, because its truth makes it more resistant to challenge and therefore a more effective insulator.

Uncomfortable output direction A: A true proposition that happens to sit near the boundary zone and carries some bidirectional stress receives this tag when in fact it is simply a contested claim that the graph has placed in a structurally significant position. The tag implies deliberate rhetorical function that the geometry cannot prove.

Uncomfortable output direction B: A technically true claim that was deliberately placed as an insulator scores below the two-signal threshold because the epistatic suppression is shared between it and an adjacent proposition. The structural function is present but distributed, and the tag is not applied because no single node accumulates sufficient signals.

Kill switch: If a proposition receives the bridge_narrative secondary tag but its removal from the graph does not meaningfully shorten the geodesic distance between any registered anchor and any false attractor, Signal 3 has not been satisfied and the tag is incorrectly applied. Signal 3 is the load-bearing criterion and its failure falsifies any bridge classification, primary or secondary.

---

# Layer 5: Compound Resolution Rules

## RULE-019: Decomposition Trigger and Execution

When a proposition is identified as compound by either Pathway 1 or Pathway 2 of RULE-004, or is flagged as a dual-band sentence by the adapted NarrativeWavelet, decomposition is required before classification proceeds. Decomposition produces sub-propositions that are each independently evaluable. The parent proposition remains in the proposition table with is_compound: true and its own record, but its primary_classification is derived from the most consequential sub-proposition's classification.

Decomposition must be minimal. Produce the smallest number of sub-propositions that resolves the heterogeneity or independent evaluability. Do not decompose further than necessary. A proposition decomposed into six sub-propositions is a signal that the original bounding was wrong, not that the decomposition is thorough.

The decomposition_note in compound_resolution must state what was separated and why — which pathway triggered the decomposition and what epistemic categories or independent evaluability criteria were violated by the compound form.

Uncomfortable output direction A: A carefully constructed integrated argument — where the moral and empirical reasoning are genuinely intertwined and cannot be separated without destroying the claim — gets decomposed under Pathway 2. The decomposition produces sub-propositions that are individually classifiable but collectively misrepresent the original argument.

Uncomfortable output direction B: A compound proposition with one anchored_true sub-claim and one anchored_false sub-claim does not get decomposed because the analyst reads the sentence as a single unified claim and does not apply the independent evaluability test. The false sub-claim is protected by the true one and the compound form never triggers RULE-004.

Kill switch: If decomposition of a compound proposition produces sub-propositions where classifying one requires knowing the classification of a sibling, the decomposition did not resolve the compound structure. If a decomposition pass is accepted by the schema with epistemically entangled sub-propositions, the independence validation is broken and the rule is falsified.

---

## RULE-020: Sub-Proposition Independence Requirement

Each sub-proposition produced by decomposition must be independently evaluable — its classification must not change based on the classification of any sibling sub-proposition from the same parent. If classifying sub-proposition A as anchored_true makes sub-proposition B more likely to be anchored_true, the two are not independent and decomposition was not executed correctly.

A sub-proposition may share anchors with its siblings. Shared anchors do not violate independence. What violates independence is when the classification of one sub-proposition is a logical input to the classification of another.

Uncomfortable output direction A: Two sub-propositions that share a large anchor set and are about the same subject receive different classifications when in fact the shared anchor set makes them logically linked. The rule says shared anchors do not violate independence, but in practice shared anchors on closely related claims create entanglement the rule does not catch.

Uncomfortable output direction B: Two sub-propositions are logically entangled — one true only if the other is false — but the entanglement runs through a proposition in a different part of the document rather than through shared anchors. The independence test passes locally but fails globally.

Kill switch: If two sub-propositions from the same decomposition are each individually classified, and reversing the classification of one causes the correct classification of the other to change, the two are logically entangled and independence was not achieved. If this condition is detected after a decomposition pass has been committed, the decomposition is invalid and the rule is falsified for that parent proposition.

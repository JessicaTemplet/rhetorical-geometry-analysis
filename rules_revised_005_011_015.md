# Revised Rules — RULE-005, RULE-011, RULE-015
# Status: Draft. Pending analyst review before incorporation into full rule set.

---

## RULE-005: Anchor Validation and Independent Genesis

An anchor is valid if it meets all three conditions: it falls within one of the four anchor classification categories (mathematical_truth, direct_empirical_record, primary_source_attestation, scientific_consensus); it has a chain_length_from_governance_zone that can be estimated as an integer; and it carries an independent_genesis record.

The independent_genesis requirement is the addition in v0.4. An anchor must document whether at least one source establishing it originates outside the administrative chain of command of the document under analysis. This requirement exists because repetition within a single institutional chain does not constitute independent corroboration. A lie told consistently across all contemporary governmental correspondence from the same administration passes the volume test without passing the truth test. Institutional weight is not independent genesis.

The three genesis confidence weights and their consequences:

**full** — At least one source establishing this anchor originates entirely outside the administrative chain of command. The anchor carries full evidential weight.

**reduced** — All available sources establishing this anchor originate within the same administrative chain of command as the document under analysis. The anchor is not disqualified. In legal discovery environments and for historical documents, institutional correspondence may be the only surviving record, and wholesale disqualification would eliminate anchors that are genuinely the best available evidence. The anchor is flagged, its reduced weight propagates to any proposition that maps to it exclusively, and the absence of independent genesis must be documented in the genesis_note field. A confidence_flag record is required.

**minimal** — Provenance is unclear. The anchor requires elevated scrutiny. A confidence_flag record is required and the anchor may not serve as the sole mapping for any proposition classified as anchored_true or anchored_false without analyst justification in the notes field.

Repetition alone — even across many documents from the same source — does not raise a reduced anchor to full. The independence of the source is what matters, not the volume of attestation.

Uncomfortable output direction A: An anchor drawn from a single authoritative scientific consensus document that was produced entirely within a government research program receives reduced confidence weight because its genesis is institutional. The independent genesis requirement treats a genuine scientific finding as epistemically equivalent to a coordinated institutional lie, which misapplies the rule. The rule targets administrative chain-of-command dependency, not institutional origin broadly. Scientific consensus anchors should be evaluated against whether the consensus itself was independently corroborated across the research community, not whether the document reporting it was government-produced.

Uncomfortable output direction B: A coordinated falsehood repeated across twenty documents from the same administration receives full genesis confidence weight because an analyst locates one document from a nominally independent source — a newspaper report — that uncritically reproduces the official claim without independent verification. The newspaper is outside the administrative chain of command but is not an independent source of the underlying fact. The rule as written checks source origin, not source independence of verification. A compliant press reproducing official claims without investigation passes the formal test and fails the epistemic one.

Kill switch: If a mathematical truth — a claim formally provable without any empirical record — fails the independent genesis check because its only documentation is a government-produced mathematical table, the independent genesis requirement is miscalibrated and the rule is falsified for that anchor classification.

---

## RULE-011: Bridge Narrative Detection

A proposition receives the primary classification bridge_narrative when it satisfies the epistatic suppression criterion (Signal 3) plus at least two of the remaining three signals. Signal 3 is the load-bearing criterion. The four signals are:

**Signal 1 — Boundary position:** The proposition occupies the boundary zone of the Poincaré disk, positioned between the governance zone anchor cluster and a false attractor. This signal raises suspicion and triggers closer examination. It does not independently qualify a proposition as a bridge narrative and does not contribute to the signal count. It is a prior, not a signal.

**Signal 2 — Bidirectional stress:** The proposition carries T_μν contributions in both directions — toward the governance zone and toward the false attractor — indicating it is load-bearing in both directions simultaneously.

**Signal 3 — Epistatic suppression (load-bearing criterion):** Removing this node from the graph meaningfully shortens the geodesic distance between a registered anchor and a false attractor. This is the counterfactual test. If removal does not shorten that distance, the proposition is not functioning as an insulator regardless of its position on the disk. A boundary-position proposition that fails the counterfactual test is a nuanced or complex truth, not a bridge. Signal 3 must be satisfied for bridge_narrative classification to proceed.

**Signal 4 — Inferential dependency:** At least one anchored_false proposition has this proposition in its anchor mapping chain. The false attractor depends on the bridge remaining in place.

Classification requires Signal 3 plus at least two of Signals 2 and 4. Boundary position (Signal 1) is not part of the count.

The reason Signal 3 is load-bearing rather than contributory: bridge narratives are defined by their structural function, not their location. A proposition that sits on the boundary but whose removal changes nothing about the graph's geodesic structure is not a bridge — it is a nuanced claim that the geometry has placed in a structurally significant position by coincidence of subject matter. The counterfactual distance test is the only signal that directly measures function rather than position or correlation.

Uncomfortable output direction A: A genuinely complex historical claim — one that is true and touches both a well-anchored fact and a contested causal interpretation — sits at the boundary zone, carries bidirectional stress because it is relevant to multiple parts of the graph, and satisfies Signal 3 because its removal slightly shortens one geodesic path. It receives bridge_narrative classification when in fact it is simply a claim that multiple arguments depend on because it is important and true. The counterfactual test catches structural dependency but cannot distinguish deliberate placement from incidental importance. The tag implies a rhetorical function the geometry cannot prove.

Uncomfortable output direction B: A deliberately placed bridge narrative is distributed across two adjacent propositions rather than concentrated in one. Each proposition individually fails Signal 3 — neither alone, when removed, meaningfully shortens the geodesic distance between the anchor and the false attractor. But together their removal does. The rule as written evaluates nodes individually. A distributed bridge that requires two nodes to be removed simultaneously to reveal the underlying contradiction is not detected.

Kill switch: If a known non-contentious historical fact — such as the ratification date of a treaty, where no reasonable contestation exists — triggers Signal 3 and satisfies the full classification threshold, the manifold zone assignments are miscalibrated and this rule is falsified. The governance zone boundary is in the wrong place if uncontested facts are registering as bridge narratives.

---

## RULE-015: Selective Omission

The secondary tag selective_omission is applied when a proposition is true as stated but the Sheaf Holonomy analysis identifies a major semantic drift — max_drift_index — at or near the section containing this proposition, and the Zipfian Auditor simultaneously flags a vocabulary anomaly in the same section, and the analyst can specify in the proposition record's notes field what the omitted context is and where it exists in the historical record. All three conditions are required. Either signal alone is not sufficient. A documentable omission without both signals is not sufficient. Both signals without a documentable omission are not sufficient.

The content vocabulary calibration is the v0.4 addition. The Zipfian Auditor must be calibrated to detect anomalies in domain-specific content vocabulary rather than in the full Zipfian distribution. This calibration is required because 19th-century governmental documents routinely shift between legal-bureaucratic and biblical-paternalistic registers as a stylistic convention of the era. These stylistic register shifts redistribute high-frequency function words and genre markers and will trigger full-distribution Zipfian anomalies. They are not selective omissions.

The operational distinction between a stylistic register shift and a content omission:

A **stylistic register shift** changes the function vocabulary and genre markers. Legal terminology gives way to paternalistic or religious register. The content vocabulary — named parties, geographic specifics, land descriptions, specific legal obligations, quantitative claims — remains present at normal density throughout the section.

A **content omission** thins or drops the domain-specific content vocabulary. Named parties, geographic specifics, legal obligations, or quantitative claims disappear or thin abruptly while the surrounding sections are dense with them. The holonomy drift and Zipfian content-vocabulary anomaly co-occur at the same section boundary. The analyst can point to what is missing and where it exists in the external historical record.

The selective_omission tag requires that the analyst name the omitted content and its external source in the notes field. A vocabulary anomaly that cannot be paired with a documentable, named omission is a register shift and the tag is not applied.

Uncomfortable output direction A: A section of the Indian Removal Act shifts from legal-bureaucratic language to biblical-paternalistic language — a genuine stylistic register shift of the era. Under full-distribution Zipfian calibration, this triggers a vocabulary anomaly. The Sheaf Holonomy analysis registers drift at the same section boundary. The tag is applied when in fact no content has been omitted; the rhetorical register simply changed. Content-vocabulary calibration is designed to prevent this false positive, but the calibration requires the analyst to correctly separate function-word redistribution from content-word thinning. If the analyst does not perform this separation, the false positive survives.

Uncomfortable output direction B: A genuine content omission is spread gradually across a document rather than concentrated at a section boundary. The holonomy max_drift_index does not fire on any single section because the drift is incremental. The Zipfian content-vocabulary anomaly is similarly diffuse. Neither signal fires at threshold. The tag is not applied even though the cumulative omission across the document is analytically significant. The rule is calibrated for local section-level signals and misses distributed omissions.

Kill switch: If a section of a document that has been independently verified as complete — meaning historians have confirmed no material was omitted at that point — triggers both the max_drift_index and the Zipfian content-vocabulary anomaly at threshold, the signal calibration is wrong and this rule is falsified for that document type.

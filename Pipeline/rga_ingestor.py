"""
RGA Ingestor — Automated Ingestion and Classification Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Takes raw claim text and a set of anchor documents (primary source texts
already processed through the RGA pipeline) and produces a partial_record
in the standard RGA schema that can be fed directly into run_pipeline.

No LLM is used at any stage. Classification is deterministic:

  1. AdaptiveChunker segments the claim text into coherent proposition-
     sized units using density-aware splitting.

  2. SRLPipeline extracts predicate-argument structure from each chunk:
     agent, action, target, entities, memory value type.

  3. RGAClassifier maps each chunk to a classification against the anchor
     corpus using three signals in priority order:

       a. Entity overlap — named entities in the chunk vs. entities
          mentioned in the anchor propositions. High overlap = this chunk
          is about the same subject matter as the anchor.

       b. Lexical alignment — TF-IDF cosine similarity between the
          chunk's action-target text and the anchor's verbatim proposition
          text. Measures how close the claim language is to the anchor
          language.

       c. Polarity detection — the chunk's root verb lemma and its
          negation markers are compared against the anchor proposition's
          root verb. Agreement in polarity -> anchored_true candidate.
          Disagreement -> anchored_false candidate.

  4. Classification thresholds:

       entity_overlap >= 0.4  AND  lexical_sim >= 0.35  AND  polarity AGREE
           -> anchored_true

       entity_overlap >= 0.4  AND  lexical_sim >= 0.35  AND  polarity DISAGREE
           -> anchored_false

       entity_overlap >= 0.4  AND  lexical_sim < 0.35
           -> bridge_narrative  (same subject, different claim)

       entity_overlap < 0.4  AND  lexical_sim >= 0.25
           -> inferentially_true or inferentially_false (polarity-driven)

       entity_overlap < 0.4  AND  lexical_sim < 0.25
           -> ambiguous

  5. The output partial_record includes:
       - document_metadata  (populated from the claim source info)
       - anchor_registry    (carried from the anchor corpus)
       - proposition_table  (one entry per classified chunk)
       - confidence_flags   (low-confidence classifications flagged)

     This record is then passed to run_pipeline, which computes geodesic
     distances, derives stress, builds the Laplacian, and returns the
     tension score.

Anchor corpus format:

  The anchor corpus is a list of run_pipeline output dicts (the same JSON
  files already produced for the ten historical documents). The ingestor
  reads anchor_registry and proposition_table from each to build its
  matching index.

Usage:

    from rga_ingestor import RGAIngestor, IngestorConfig
    import json

    # Load your anchor documents
    anchors = [
        json.load(open("georgia_secession.json")),
        json.load(open("mississippi_secession.json")),
    ]

    ingestor = RGAIngestor(anchor_corpus=anchors)
    partial_record = ingestor.ingest(
        text=open("claim_text.txt").read(),
        source_label="1960s_textbook_chapter_4",
        document_id="CLAIM-001",
    )

    # Feed directly into the pipeline
    from rga_pipeline import run_pipeline
    result = run_pipeline(partial_record)
    print(result["geometric_summary"]["tension_score"])
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ── Optional dependencies — same graceful degradation as the SRL pipeline ─────

try:
    from veritas_memoria.core.write_path.adaptive_chunker import AdaptiveChunker
    _CHUNKER_AVAILABLE = True
except ImportError:
    try:
        from adaptive_chunker import AdaptiveChunker
        _CHUNKER_AVAILABLE = True
    except ImportError:
        _CHUNKER_AVAILABLE = False
        logger.warning("RGAIngestor: AdaptiveChunker unavailable — using sentence splitter fallback.")

try:
    from srl import SRLPipeline, MemoryValue, SemanticAnnotation
    _SRL_AVAILABLE = True
except ImportError:
    _SRL_AVAILABLE = False
    logger.warning("RGAIngestor: SRLPipeline unavailable — using regex extraction fallback.")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("RGAIngestor: sklearn unavailable — using token overlap for lexical similarity.")


# ── Negation markers — used for polarity detection ────────────────────────────

_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "nobody", "none", "cannot", "cant", "cant", "wont", "didnt",
    "wasnt", "werent", "isnt", "arent", "doesnt", "dont", "hadnt",
    "hasnt", "havent", "shouldnt", "wouldnt", "couldnt", "without",
})

# Verbs whose meaning strongly implies assertion of causation or motivation.
# When a claim uses these against the same subject as an anchor, it is
# making a positive claim about causation — relevant for "it was/wasn't about X".
_CAUSAL_VERBS = frozenset({
    "cause", "drive", "motivate", "stem", "result", "produce", "lead",
    "fight", "rebel", "secede", "declare", "preserve", "protect", "defend",
    "maintain", "uphold", "concern", "involve", "relate", "center",
})


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class IngestorConfig:
    """
    Thresholds and settings for the deterministic classifier.

    These are not magic numbers — they are classification boundaries
    expressed as minimum similarity scores. The correct long-term approach
    is to derive these from the empirical distribution of similarity scores
    across the anchor corpus, but that requires a labeled evaluation set.
    Until that exists, these are documented starting points with explicit
    audit entries when they are applied.
    """

    # Entity overlap: fraction of chunk entities that appear in the anchor
    entity_overlap_threshold: float = 0.40

    # Lexical similarity: TF-IDF cosine sim between chunk text and anchor text
    lexical_sim_threshold_strong: float = 0.35   # -> direct classification
    lexical_sim_threshold_weak:   float = 0.25   # -> inferential classification

    # Minimum similarity to assign any anchor mapping at all
    lexical_sim_floor:            float = 0.12

    # Confidence below which a classification gets a CF-LOW-CONFIDENCE flag
    confidence_flag_threshold:    float = 0.55

    # Chunker settings
    chunker_mode: str = "thorough"
    chunker_min:  int = 120
    chunker_max:  int = 600

    # SRL settings
    srl_fast_mode: bool = True   # full mode reserved for background enricher


# ── Anchor index ───────────────────────────────────────────────────────────────

@dataclass
class AnchorEntry:
    """
    A single anchor from the corpus, with its text and extracted entities
    pre-indexed for fast matching.
    """
    anchor_id:    str
    document_id:  str
    common_name:  str
    verbatim:     str          # verbatim source text from anchor proposition
    normalized:   str          # normalized claim text
    entities:     List[str]    # named entities extracted by SRL
    root_verb:    Optional[str]  # root verb lemma from SRL action field
    anchor_entities: List[str]  # entities from the anchor_registry entry itself


class AnchorIndex:
    """
    Pre-built index of all anchors across the corpus for fast matching.

    Built once from the anchor corpus at ingestor construction time.
    """

    def __init__(self, anchor_corpus: List[Dict], srl: Optional[Any] = None):
        self.entries: List[AnchorEntry] = []
        self._vectorizer: Optional[Any] = None
        self._tfidf_matrix: Optional[Any] = None
        self._build(anchor_corpus, srl)

    def _build(self, anchor_corpus: List[Dict], srl: Optional[Any]) -> None:
        all_texts: List[str] = []

        for doc_record in anchor_corpus:
            meta      = doc_record.get("document_metadata", {})
            doc_id    = meta.get("document_id", "UNKNOWN")
            doc_name  = meta.get("common_name", doc_id)

            # Index anchor_registry entries — these are the fixed governance nodes
            for anchor in doc_record.get("anchor_registry", []):
                anchor_entities = _extract_entity_tokens(anchor.get("verbatim_text", ""))
                # Also grab any entity text from the anchor fields themselves
                for field_key in ("anchor_concept", "anchor_text", "verbatim_text"):
                    val = anchor.get(field_key, "")
                    if val:
                        anchor_entities.extend(_extract_entity_tokens(val))
                anchor_entities = list(set(anchor_entities))

                entry = AnchorEntry(
                    anchor_id=anchor["anchor_id"],
                    document_id=doc_id,
                    common_name=doc_name,
                    verbatim=anchor.get("verbatim_text", ""),
                    normalized=anchor.get("anchor_concept", anchor.get("verbatim_text", "")),
                    entities=anchor_entities,
                    root_verb=None,
                    anchor_entities=anchor_entities,
                )
                self.entries.append(entry)
                all_texts.append(f"{entry.verbatim} {entry.normalized}")

            # Also index propositions classified as anchored_true —
            # they carry verbatim anchor text and are the closest thing
            # to the anchor's stated reasoning.
            for prop in doc_record.get("proposition_table", []):
                if prop.get("primary_classification") != "anchored_true":
                    continue
                verbatim   = prop.get("verbatim_source_text", "") or ""
                normalized = prop.get("normalized_claim", "") or ""
                if not verbatim and not normalized:
                    continue

                # Use SRL entities if available, otherwise token extraction
                entities = prop.get("entities", []) or []
                if not entities:
                    entities = _extract_entity_tokens(verbatim)

                entry = AnchorEntry(
                    anchor_id=prop.get("anchor_mappings", [""])[0] if prop.get("anchor_mappings") else "",
                    document_id=doc_id,
                    common_name=doc_name,
                    verbatim=verbatim,
                    normalized=normalized,
                    entities=entities,
                    root_verb=None,
                    anchor_entities=entities,
                )
                self.entries.append(entry)
                all_texts.append(f"{verbatim} {normalized}")

        if not self.entries:
            logger.warning("AnchorIndex: no anchor entries found in corpus.")
            return

        # Build TF-IDF index over all anchor texts for fast cosine similarity
        if _SKLEARN_AVAILABLE and all_texts:
            try:
                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=10000,
                    sublinear_tf=True,
                )
                self._tfidf_matrix = self._vectorizer.fit_transform(all_texts)
                logger.info(
                    "AnchorIndex: built TF-IDF index over %d anchor entries.",
                    len(self.entries)
                )
            except Exception as e:
                logger.warning("AnchorIndex: TF-IDF build failed (%s) — using token overlap.", e)

    def find_matches(
        self,
        chunk_text: str,
        chunk_entities: List[str],
        top_k: int = 5,
    ) -> List[Tuple[AnchorEntry, float, float]]:
        """
        Find the top-k anchor entries most similar to the chunk.

        Returns list of (AnchorEntry, entity_overlap, lexical_sim) tuples,
        sorted by combined score descending.
        """
        if not self.entries:
            return []

        # Entity overlap scores
        chunk_entity_set = set(e.lower() for e in chunk_entities if len(e) > 2)
        entity_scores: List[float] = []
        for entry in self.entries:
            anchor_set = set(e.lower() for e in entry.entities if len(e) > 2)
            if not anchor_set and not chunk_entity_set:
                entity_scores.append(0.0)
            elif not anchor_set or not chunk_entity_set:
                entity_scores.append(0.0)
            else:
                overlap = len(chunk_entity_set & anchor_set)
                entity_scores.append(overlap / max(len(chunk_entity_set), len(anchor_set)))

        # Lexical similarity scores
        if _SKLEARN_AVAILABLE and self._vectorizer is not None and self._tfidf_matrix is not None:
            try:
                chunk_vec   = self._vectorizer.transform([chunk_text])
                sim_scores  = cosine_similarity(chunk_vec, self._tfidf_matrix).flatten()
                lexical_scores = sim_scores.tolist()
            except Exception:
                lexical_scores = [_token_overlap(chunk_text, e.verbatim) for e in self.entries]
        else:
            lexical_scores = [_token_overlap(chunk_text, e.verbatim) for e in self.entries]

        # Combined score: weighted sum, entity overlap weighted higher for
        # subject matching, lexical for content matching
        combined = [
            0.55 * entity_scores[i] + 0.45 * lexical_scores[i]
            for i in range(len(self.entries))
        ]

        # Sort and return top_k
        ranked = sorted(
            zip(self.entries, entity_scores, lexical_scores, combined),
            key=lambda x: x[3],
            reverse=True,
        )

        return [
            (entry, ent_score, lex_score)
            for entry, ent_score, lex_score, _ in ranked[:top_k]
            if lex_score >= 0.0  # filter zero-signal matches
        ]


# ── Deterministic classifier ───────────────────────────────────────────────────

class RGAClassifier:
    """
    Deterministic classifier that maps a chunk + its SRL output to an
    RGA classification using entity overlap, lexical similarity, and
    polarity detection.

    No LLM. No probabilistic model. Transparent and auditable.
    """

    def __init__(self, anchor_index: AnchorIndex, config: IngestorConfig):
        self.index  = anchor_index
        self.config = config

    def classify(
        self,
        chunk_text: str,
        srl_annotation: Optional[Any],   # SemanticAnnotation or None
        chunk_entities: List[str],
    ) -> Dict[str, Any]:
        """
        Classify a single chunk against the anchor corpus.

        Returns a dict with:
            primary_classification  str
            anchor_mappings         List[str]
            confidence              float
            classification_basis    str   (audit trail)
            normalized_claim        str
            verbatim_source_text    str
            stress_energy_contribution  float  (placeholder — pipeline computes this)
            geodesic_distance_from_anchor  None  (pipeline computes this)
            is_compound             bool
            is_relational           bool
        """
        matches = self.index.find_matches(chunk_text, chunk_entities, top_k=3)

        if not matches:
            return self._unmatched(chunk_text)

        best_entry, best_entity_overlap, best_lexical_sim = matches[0]

        # Polarity: does this chunk agree or disagree with the anchor?
        polarity_agree, polarity_confidence = _detect_polarity(
            chunk_text,
            srl_annotation,
            best_entry,
        )

        # Classification decision tree
        cfg = self.config
        anchor_ids = [
            m[0].anchor_id for m in matches
            if m[0].anchor_id and (m[1] >= cfg.entity_overlap_threshold * 0.5
                                   or m[2] >= cfg.lexical_sim_floor)
        ]

        if best_entity_overlap >= cfg.entity_overlap_threshold:
            if best_lexical_sim >= cfg.lexical_sim_threshold_strong:
                if polarity_agree:
                    classification = "anchored_true"
                    confidence     = 0.6 + 0.3 * best_lexical_sim + 0.1 * polarity_confidence
                else:
                    classification = "anchored_false"
                    confidence     = 0.6 + 0.3 * best_lexical_sim + 0.1 * polarity_confidence
                basis = (
                    f"entity_overlap={best_entity_overlap:.2f} "
                    f"lexical_sim={best_lexical_sim:.2f} "
                    f"polarity={'AGREE' if polarity_agree else 'DISAGREE'}"
                )

            else:
                # Same subject matter, but claim text diverges from anchor text
                classification = "bridge_narrative"
                confidence     = 0.45 + 0.2 * best_entity_overlap
                basis = (
                    f"entity_overlap={best_entity_overlap:.2f} "
                    f"lexical_sim={best_lexical_sim:.2f} (below strong threshold) "
                    f"-> bridge_narrative"
                )

        elif best_lexical_sim >= cfg.lexical_sim_threshold_weak:
            # Weak entity match but some lexical overlap — inferential
            if polarity_agree:
                classification = "inferentially_true"
            else:
                classification = "inferentially_false"
            confidence = 0.35 + 0.2 * best_lexical_sim
            basis = (
                f"entity_overlap={best_entity_overlap:.2f} (below threshold) "
                f"lexical_sim={best_lexical_sim:.2f} -> inferential"
            )

        else:
            classification = "ambiguous"
            confidence     = 0.25
            basis          = (
                f"entity_overlap={best_entity_overlap:.2f} "
                f"lexical_sim={best_lexical_sim:.2f} "
                f"both below thresholds"
            )
            anchor_ids = []

        return {
            "primary_classification":       classification,
            "anchor_mappings":              anchor_ids,
            "confidence":                   round(min(confidence, 0.95), 4),
            "classification_basis":         basis,
            "normalized_claim":             _normalize_text(chunk_text),
            "verbatim_source_text":         chunk_text,
            "stress_energy_contribution":   0.3,   # placeholder; pipeline computes from geodesic
            "geodesic_distance_from_anchor": None,
            "geodesic_distance_per_anchor":  {},
            "is_compound":                  False,
            "is_relational":                False,
            "matched_anchor_document":      best_entry.common_name,
        }

    def _unmatched(self, chunk_text: str) -> Dict[str, Any]:
        return {
            "primary_classification":       "out_of_scope",
            "anchor_mappings":              [],
            "confidence":                   0.1,
            "classification_basis":         "no anchor matches found",
            "normalized_claim":             _normalize_text(chunk_text),
            "verbatim_source_text":         chunk_text,
            "stress_energy_contribution":   0.0,
            "geodesic_distance_from_anchor": None,
            "geodesic_distance_per_anchor":  {},
            "is_compound":                  False,
            "is_relational":                False,
            "matched_anchor_document":      None,
        }


# ── Main ingestor ──────────────────────────────────────────────────────────────

class RGAIngestor:
    """
    Automated ingestion pipeline.

    Takes raw claim text and a pre-processed anchor corpus, and produces
    a partial_record ready for run_pipeline.

    Args:
        anchor_corpus:  List of run_pipeline output dicts (the JSON result
                        files for the anchor documents).
        config:         IngestorConfig. Defaults to the standard thresholds.
    """

    def __init__(
        self,
        anchor_corpus: List[Dict],
        config: Optional[IngestorConfig] = None,
    ):
        self.config = config or IngestorConfig()

        # Initialize chunker
        if _CHUNKER_AVAILABLE:
            self._chunker = AdaptiveChunker(
                min_chunk_size=self.config.chunker_min,
                max_chunk_size=self.config.chunker_max,
                srl_enabled=False,   # SRL runs separately after chunking
                mode=self.config.chunker_mode,
            )
        else:
            self._chunker = None

        # Initialize SRL
        if _SRL_AVAILABLE:
            self._srl = SRLPipeline()
        else:
            self._srl = None

        # Build anchor index
        logger.info("RGAIngestor: building anchor index from %d document(s)...", len(anchor_corpus))
        self._index = AnchorIndex(anchor_corpus, self._srl)

        # Classifier
        self._classifier = RGAClassifier(self._index, self.config)

        # Carry anchor_registry from corpus for inclusion in output record
        self._anchor_registry = _merge_anchor_registries(anchor_corpus)

        logger.info(
            "RGAIngestor: ready. %d anchors indexed, %d anchor entries.",
            len(self._anchor_registry),
            len(self._index.entries),
        )

    def ingest(
        self,
        text: str,
        source_label: str,
        document_id: Optional[str] = None,
        date_of_document: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest raw claim text and produce a partial_record for run_pipeline.

        Args:
            text:             Raw claim text (transcript, document, speech, etc.)
            source_label:     Human-readable label for this claim source.
            document_id:      Optional ID. Auto-generated if not provided.
            date_of_document: Optional date string for the claim document.

        Returns:
            partial_record dict compatible with run_pipeline.
        """
        if not document_id:
            digest = hashlib.sha256(text[:500].encode()).hexdigest()[:8].upper()
            document_id = f"CLAIM-{digest}"

        logger.info(
            "RGAIngestor: ingesting '%s' (%d chars) as %s",
            source_label, len(text), document_id,
        )

        # ── Step 1: Chunk ─────────────────────────────────────────────────────
        chunks = self._chunk(text)
        logger.info("RGAIngestor: produced %d chunks.", len(chunks))

        # ── Step 2: SRL batch ─────────────────────────────────────────────────
        chunk_texts  = [c["content"] for c in chunks]
        srl_docs     = self._run_srl_batch(chunk_texts)

        # ── Step 3: Classify each chunk ───────────────────────────────────────
        proposition_table: List[Dict] = []
        confidence_flags:  List[Dict] = []

        for idx, (chunk, srl_doc) in enumerate(zip(chunks, srl_docs)):
            pid = f"P-{document_id}-{idx+1:03d}"

            # Get the best sentence from the SRL doc for role extraction
            best_annotation = None
            if srl_doc and srl_doc.sentences:
                kept = srl_doc.kept_sentences()
                if kept:
                    # Prefer high-signal M&A labels; fall back to old general labels
                    for mv in ("PROVISION", "FINDING", "OBLIGATION", "DEFINITION",
                               "DEADLINE", "FACT", "DECISION", "ACTION"):
                        for s in kept:
                            if hasattr(s, 'value') and s.value.value == mv:
                                best_annotation = s
                                break
                        if best_annotation:
                            break
                    if not best_annotation:
                        best_annotation = kept[0]

            # Gather entities from all kept sentences in the doc
            chunk_entities: List[str] = []
            if srl_doc and srl_doc.sentences:
                for sent in srl_doc.kept_sentences():
                    chunk_entities.extend(getattr(sent, 'entities', []) or [])
            # Supplement with token-level extraction for proper nouns
            chunk_entities.extend(_extract_entity_tokens(chunk["content"]))
            chunk_entities = list(set(chunk_entities))

            result = self._classifier.classify(
                chunk_text=chunk["content"],
                srl_annotation=best_annotation,
                chunk_entities=chunk_entities,
            )

            # Determine chain_length from classification
            chain_length = _chain_length_for_classification(result["primary_classification"])

            prop = {
                "proposition_id":              pid,
                "verbatim_source_text":        result["verbatim_source_text"],
                "normalized_claim":            result["normalized_claim"],
                "primary_classification":      result["primary_classification"],
                "anchor_mappings":             result["anchor_mappings"],
                "stress_energy_contribution":  result["stress_energy_contribution"],
                "geodesic_distance_from_anchor": result["geodesic_distance_from_anchor"],
                "geodesic_distance_per_anchor":  result["geodesic_distance_per_anchor"],
                "is_compound":                 result["is_compound"],
                "is_relational":               result["is_relational"],
                "chain_length_from_governance_zone": chain_length,
                "classification_confidence":   result["confidence"],
                "classification_basis":        result["classification_basis"],
                "matched_anchor_document":     result.get("matched_anchor_document"),
                "srl_value_type": (
                    best_annotation.value.value
                    if best_annotation and hasattr(best_annotation, 'value')
                    else None
                ),
                "srl_agent":  getattr(best_annotation, 'agent',  None) if best_annotation else None,
                "srl_action": getattr(best_annotation, 'action', None) if best_annotation else None,
                "srl_target": getattr(best_annotation, 'target', None) if best_annotation else None,
            }
            proposition_table.append(prop)

            # Flag low-confidence classifications
            if result["confidence"] < self.config.confidence_flag_threshold:
                confidence_flags.append({
                    "flag_id":   f"CF-LOW-CONFIDENCE-{pid}",
                    "flag_type": "low_confidence_classification",
                    "proposition_id": pid,
                    "classification": result["primary_classification"],
                    "confidence":     result["confidence"],
                    "basis":          result["classification_basis"],
                    "note": (
                        f"Classification confidence {result['confidence']:.2f} below "
                        f"threshold {self.config.confidence_flag_threshold}. "
                        f"Manual review recommended."
                    ),
                    "severity": "advisory",
                })

        # ── Step 4: Build partial_record ──────────────────────────────────────
        partial_record = {
            "schema_version": "0.4",
            "document_metadata": {
                "document_id":      document_id,
                "common_name":      source_label,
                "date_of_document": date_of_document or "unknown",
                "document_type":    "claim_corpus",
                "ingested_at":      datetime.datetime.utcnow().isoformat(),
                "anchor_corpus_size": len(self._anchor_registry),
                "ingestor_version": "1.0",
            },
            "anchor_registry":   self._anchor_registry,
            "proposition_table": proposition_table,
            "confidence_flags":  confidence_flags,
            "falsifiability_record": [],
            "graph_structure":   {},
            "geometric_summary": {},
            "audit_trail":       [{
                "module":    "rga_ingestor",
                "status":    "complete",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "note": (
                    f"Ingested {len(proposition_table)} propositions from '{source_label}'. "
                    f"Classifications: "
                    + ", ".join(
                        f"{cls}={sum(1 for p in proposition_table if p['primary_classification']==cls)}"
                        for cls in [
                            "anchored_true","anchored_false","bridge_narrative",
                            "inferentially_true","inferentially_false","ambiguous","out_of_scope"
                        ]
                        if any(p['primary_classification']==cls for p in proposition_table)
                    )
                ),
            }],
        }

        logger.info(
            "RGAIngestor: partial_record ready for run_pipeline. "
            "%d propositions, %d confidence flags.",
            len(proposition_table), len(confidence_flags),
        )
        return partial_record

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> List[Dict]:
        if self._chunker is not None:
            try:
                return self._chunker.chunk_document(text)
            except Exception as e:
                logger.warning("RGAIngestor: chunker failed (%s) — using sentence fallback.", e)
        return _sentence_chunk_fallback(text, self.config.chunker_max)

    def _run_srl_batch(self, texts: List[str]) -> List[Optional[Any]]:
        if self._srl is None or not texts:
            return [None] * len(texts)
        try:
            return self._srl.process_batch(texts, fast_mode=self.config.srl_fast_mode)
        except Exception as e:
            logger.warning("RGAIngestor: SRL batch failed (%s) — skipping SRL.", e)
            return [None] * len(texts)


# ── Helper functions ───────────────────────────────────────────────────────────

def _extract_entity_tokens(text: str) -> List[str]:
    """
    Extract candidate named entity tokens from text without spaCy.

    Looks for sequences of capitalized words (2+ chars) that are not
    at the start of a sentence. This catches proper nouns, place names,
    and institution names without requiring NER.

    Also extracts key thematic terms relevant to the civil war domain
    specifically, since those are the anchors we are initially testing against.
    """
    if not text:
        return []

    tokens: List[str] = []

    # Capitalized proper noun sequences (not sentence-initial)
    proper_nouns = re.findall(r'(?<!\.\s)(?<![.!?]\s)\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text)
    tokens.extend(proper_nouns)

    # Domain-specific thematic terms for the civil war corpus
    # These are the concepts the secession declarations explicitly name.
    _DOMAIN_TERMS = [
        "slavery", "slave", "slaves", "enslaved", "institution", "abolition",
        "secession", "secede", "confederacy", "confederate", "union", "federal",
        "states rights", "state rights", "sovereignty", "nullification",
        "cotton", "plantation", "property", "negro", "african", "race",
        "tariff", "economy", "agriculture", "war", "rebellion", "civil war",
        "states war", "war between the states", "war of northern aggression",
        "freedom", "liberty", "constitution", "amendment",
    ]
    text_lower = text.lower()
    for term in _DOMAIN_TERMS:
        if term in text_lower:
            tokens.append(term)

    return list(set(tokens))


def _token_overlap(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity between two text strings.
    Fallback when sklearn is unavailable.
    """
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(re.findall(r'\b[a-z]{3,}\b', text_a.lower()))
    tokens_b = set(re.findall(r'\b[a-z]{3,}\b', text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _detect_polarity(
    chunk_text: str,
    srl_annotation: Optional[Any],
    anchor_entry: AnchorEntry,
) -> Tuple[bool, float]:
    """
    Determine whether the chunk agrees or disagrees with the anchor.

    Returns (agrees: bool, confidence: float).

    Polarity is determined by:
    1. Negation marker presence in the chunk near the root verb
    2. Root verb lemma comparison (same causal verb = direct comparison)
    3. Presence of reframing language ("actually", "really", "in fact",
       "not about", "wasn't about", "states rights", "war between")
    """
    chunk_lower = chunk_text.lower()

    # Explicit disagreement / reframing markers
    _REFRAME_PATTERNS = [
        r"\bnot\s+(?:really\s+)?about\b",
        r"\bwasn'?t\s+(?:really\s+)?about\b",
        r"\bweren'?t\s+(?:really\s+)?about\b",
        r"\bwar\s+between\s+the\s+states\b",
        r"\bwar\s+of\s+northern\s+aggression\b",
        r"\bstates'?\s+rights?\b",
        r"\beconomic\s+(?:issue|reason|cause|factor)\b",
        r"\bnot\s+the\s+(?:main\s+)?(?:cause|reason|issue)\b",
        r"\bactually\s+about\b",
        r"\breally\s+about\b",
    ]
    for pattern in _REFRAME_PATTERNS:
        if re.search(pattern, chunk_lower):
            return False, 0.80   # disagrees with anchor, high confidence

    # Explicit agreement / corroboration markers
    _AGREE_PATTERNS = [
        r"\bwas\s+about\s+slavery\b",
        r"\babout\s+(?:the\s+)?preservation\s+of\s+slavery\b",
        r"\bslaver[y|s]\s+was\s+(?:the\s+)?(?:central|primary|main|key)\b",
        r"\bto\s+(?:preserve|protect|maintain|expand)\s+slavery\b",
        r"\benslave[d|ment]\b",
        r"\binstitution\s+of\s+slavery\b",
    ]
    for pattern in _AGREE_PATTERNS:
        if re.search(pattern, chunk_lower):
            return True, 0.80

    # Negation check: count negation tokens near the root verb
    words       = chunk_lower.split()
    neg_count   = sum(1 for w in words if w.strip(".,;:?!") in _NEGATION_TOKENS)
    total_words = max(len(words), 1)

    # SRL root verb comparison
    if srl_annotation and hasattr(srl_annotation, 'action') and srl_annotation.action:
        chunk_verb = srl_annotation.action.lower()
        if anchor_entry.root_verb:
            anchor_verb = anchor_entry.root_verb.lower()
            # Same causal verb family: direct comparison, polarity from negation
            if chunk_verb in _CAUSAL_VERBS or anchor_verb in _CAUSAL_VERBS:
                agrees = neg_count == 0
                return agrees, 0.60

    # Weak signal: negation density heuristic
    neg_density = neg_count / total_words
    if neg_density > 0.05:
        return False, 0.40
    else:
        return True, 0.35   # default to agree, low confidence


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip artifacts from chunk text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text[:500]   # cap normalized claim length


def _chain_length_for_classification(classification: str) -> int:
    """
    Map classification to chain_length_from_governance_zone.
    Mirrors the zone structure: governance=1, boundary=4.
    """
    return {
        "anchored_true":        1,
        "bridge_narrative":     2,
        "inferentially_true":   2,
        "ambiguous":            3,
        "inferentially_false":  3,
        "anchored_false":       4,
        "out_of_scope":         4,
    }.get(classification, 3)


def _merge_anchor_registries(anchor_corpus: List[Dict]) -> List[Dict]:
    """
    Merge anchor_registry entries from all anchor documents into one list,
    deduplicating by anchor_id.
    """
    seen:   set     = set()
    merged: List[Dict] = []
    for doc in anchor_corpus:
        for anchor in doc.get("anchor_registry", []):
            aid = anchor.get("anchor_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                merged.append(anchor)
    return merged


def _sentence_chunk_fallback(text: str, max_size: int) -> List[Dict]:
    """
    Minimal sentence-boundary chunker used when AdaptiveChunker is unavailable.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[Dict] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > max_size and current:
            content = " ".join(current)
            chunks.append({"content": content, "size": len(content), "metadata": {}})
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent)

    if current:
        content = " ".join(current)
        chunks.append({"content": content, "size": len(content), "metadata": {}})

    return chunks


# ── Convenience: one-shot tension measurement ──────────────────────────────────

def print_tension_report(result: Dict[str, Any]) -> None:
    """
    Print a human-readable report of every value the pipeline computed.

    This is the transparency layer. Every number, every classification
    decision, every signal that fed into the final tension score is shown
    explicitly so calibration issues and incorrect assumptions are visible
    without reading the code.
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72

    meta = result.get("document_metadata", {})
    geo  = result.get("geometric_summary", {})
    ts   = geo.get("tension_score", {})
    props = result.get("proposition_table", [])
    pairs = result.get("graph_structure", {}).get("contradiction_pairs", [])
    flags = result.get("confidence_flags", [])

    print()
    print(SEP)
    print("  RGA TENSION REPORT")
    print(SEP)
    print(f"  Source:   {meta.get('common_name', 'unknown')}")
    print(f"  Doc ID:   {meta.get('document_id', 'unknown')}")
    print(f"  Anchors:  {meta.get('anchor_corpus_size', '?')} anchor(s) in corpus")
    print(f"  Ingested: {meta.get('ingested_at', 'unknown')}")
    print()

    # ── Tension score ─────────────────────────────────────────────────────────
    print(SEP2)
    print("  TENSION SCORE")
    print(SEP2)
    score = ts.get("score", "n/a")
    band  = ts.get("band", "unknown")
    interp = ts.get("interpretation", "")
    components = ts.get("components", {})

    print(f"  Final score:  {score}  [{band.upper().replace('_', ' ')}]")
    print(f"  {interp}")
    print()
    print("  Score components:")
    print(f"    Fiedler deficit      {components.get('fiedler_deficit', 'n/a'):>8}  "
          f"(weight 50%)  — how far spectral coherence dropped from baseline 1.0")
    print(f"    Contradiction load   {components.get('contradiction_load', 'n/a'):>8}  "
          f"(weight 30%)  — normalized active contradiction pairs in Laplacian")
    print(f"    False attractor ratio {components.get('false_attractor_ratio', 'n/a'):>7}  "
          f"(weight 20%)  — proportion of propositions opposing the anchor")
    print()

    # ── Spectral geometry ─────────────────────────────────────────────────────
    print(SEP2)
    print("  SPECTRAL GEOMETRY  (Poincare disk / HSH Laplacian)")
    print(SEP2)
    fiedler_note = next(
        (e["note"] for e in result.get("audit_trail", []) if "lambda2=" in e.get("note", "")),
        "Fiedler value not computed"
    )
    print(f"  {fiedler_note}")
    print(f"  Manifold description:")
    for line in geo.get("manifold_description", "Not available.").split(". "):
        line = line.strip()
        if line:
            print(f"    {line}.")
    print()
    zb = geo.get("zone_boundaries", {})
    if zb:
        print("  Zone boundaries (geodesic distance from governance origin):")
        print(f"    Governance cluster:   {zb.get('governance_geodesic', 'n/a')}")
        print(f"    False attractor zone: {zb.get('false_attractor_geodesic', 'n/a')}")
    print()

    # ── Contradiction pairs ───────────────────────────────────────────────────
    print(SEP2)
    print(f"  CONTRADICTION PAIRS  ({len(pairs)} active negative edges in Laplacian)")
    print(SEP2)
    if pairs:
        for pair in pairs:
            print(f"  Anchor: {pair.get('anchor', 'unknown')}")
            print(f"    Node A (true):  {pair.get('node_a')}")
            print(f"    Node B (false): {pair.get('node_b')}")
            print(f"    Negative edge weight: {pair.get('negative_weight')}  "
                  f"(stress_a={pair.get('stress_a')}, stress_b={pair.get('stress_b')})")
            print()
    else:
        print("  None detected.")
        print()

    # ── Propositions ──────────────────────────────────────────────────────────
    print(SEP2)
    print(f"  PROPOSITIONS  ({len(props)} total)")
    print(SEP2)
    for prop in props:
        pid   = prop.get("proposition_id", "?")
        cls   = prop.get("primary_classification", "unknown")
        conf  = prop.get("classification_confidence", "?")
        d     = prop.get("geodesic_distance_from_anchor")
        stress = prop.get("stress_energy_contribution")
        basis = prop.get("classification_basis", "")
        anchor_doc = prop.get("matched_anchor_document", "")
        verbatim = prop.get("verbatim_source_text", "")[:120].replace("\n", " ")

        print(f"  [{pid}]")
        print(f"    Classification:  {cls}  (confidence={conf})")
        if anchor_doc:
            print(f"    Matched anchor:  {anchor_doc}")
        if prop.get("anchor_mappings"):
            print(f"    Anchor mappings: {', '.join(prop['anchor_mappings'])}")
        if d is not None:
            print(f"    Geodesic dist:   {d}  (Green's function stress={stress})")
        else:
            print(f"    Geodesic dist:   not computed  (stress={stress})")
        print(f"    Basis:           {basis}")
        if prop.get("srl_action"):
            srl_parts = []
            if prop.get("srl_agent"):  srl_parts.append(f"agent={prop['srl_agent']}")
            srl_parts.append(f"action={prop['srl_action']}")
            if prop.get("srl_target"): srl_parts.append(f"target={prop['srl_target'][:60]}")
            print(f"    SRL:             {', '.join(srl_parts)}")
        print(f"    Text:            \"{verbatim}\"")
        print()

    # ── Confidence flags ──────────────────────────────────────────────────────
    if flags:
        print(SEP2)
        print(f"  CONFIDENCE FLAGS  ({len(flags)} total)")
        print(SEP2)
        for flag in flags:
            ftype = flag.get("flag_type", "unknown")
            note  = flag.get("note", "")
            sev   = flag.get("severity", "")
            print(f"  [{sev.upper()}] {ftype}")
            print(f"    {note}")
            print()

    # ── Audit trail ───────────────────────────────────────────────────────────
    print(SEP2)
    print("  PIPELINE AUDIT TRAIL")
    print(SEP2)
    for entry in result.get("audit_trail", []):
        status = entry.get("status", "?").upper()
        module = entry.get("module", "?")
        note   = entry.get("note", "")
        print(f"  [{status:12}] {module}: {note[:80]}")
    print()
    print(SEP)
    print()


def measure_tension(
    claim_text: str,
    anchor_corpus: List[Dict],
    source_label: str = "claim",
    config: Optional[IngestorConfig] = None,
    n_passes: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    One-shot convenience function: ingest a claim and run the full pipeline.

    Returns the final run_pipeline result after n_passes of geometric
    convergence, including the tension_score in geometric_summary.

    Args:
        claim_text:     Raw text of the claim being evaluated.
        anchor_corpus:  List of run_pipeline output dicts for anchor documents.
        source_label:   Human-readable label for the claim source.
        config:         Optional IngestorConfig.
        n_passes:       Number of pipeline passes for geometric convergence.
                        3 is sufficient for all current documents.
        verbose:        Print the full tension report to stdout. Default True.

    Returns:
        Final run_pipeline output dict.
    """
    from rga_pipeline import run_pipeline

    ingestor = RGAIngestor(anchor_corpus=anchor_corpus, config=config)
    partial  = ingestor.ingest(claim_text, source_label=source_label)

    result = partial
    for i in range(n_passes):
        result = run_pipeline(result)
        logger.info(
            "measure_tension: pass %d/%d complete. Fiedler=%s",
            i + 1, n_passes,
            next(
                (e["note"].split("lambda2=")[1][:6]
                 for e in result["audit_trail"]
                 if "lambda2=" in e.get("note", "")),
                "n/a"
            ),
        )

    if verbose:
        print_tension_report(result)

    return result


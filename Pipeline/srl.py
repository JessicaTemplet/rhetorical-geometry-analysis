"""
VeritasMemoria - Semantic Role Labeling Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-extraction structural analysis pipeline. Runs before the Distiller's
chunking pass to give the extraction LLM a pre-parsed skeleton rather than
raw prose.

The core problem it solves: LLMs fail to distill messy content (brainstorming
logs, informal chats, transcripts) not because they lack intelligence but
because the task is too broad — they must simultaneously figure out what type
of thing each piece of content is, whether it's worth keeping, how it relates
to other things, and how to express it concisely. Ambiguous input plus an
undivided task produces mush.

SRL narrows the problem by first answering "what is structurally happening
in each statement" before any judgment about value is made. This is the key
distinction the extraction pass regularly misses:

    "we decided to use BM25"   ->  AGENT: we  ACTION: decided  TARGET: use BM25
    "someone asked about BM25" ->  AGENT: someone  ACTION: asked  TARGET: about BM25

Same entities, completely different memory value. Flat extraction conflates them.
SRL separates them structurally before the LLM ever sees the content.

Pipeline:
    Raw text
        -> prompt stripping (prompt-free data zone enforcement)
        -> sentence segmentation
        -> spaCy structural parse (dependency tree + NER)
        -> role annotation (agent / action / target / circumstance)
        -> optional 1.5B judgment pass (value classification)
        -> ParsedDocument with annotated sentences
        -> Distiller receives ParsedDocument instead of raw text

Two-tier architecture:
    Tier 1 — spaCy (always runs if installed)
        Fast, deterministic, local, zero model calls.
        Handles: dependency parse, NER, sentence segmentation, basic roles.
        Falls back to regex segmentation if spaCy not installed.

    Tier 2 — local 1.5B via llama-cpp-python (optional)
        Runs after spaCy has produced the structural skeleton.
        Task is narrow: given this pre-parsed statement with known roles,
        classify its memory value (DECISION / FACT / PREFERENCE / ACTION /
        CORRECTION / NOISE).
        Much easier than asking it to do everything at once.
        Falls back to heuristic classification if not available.

Graceful degradation:
    spaCy installed, 1.5B available  -> full pipeline
    spaCy installed, no 1.5B         -> structural parse + heuristic classification
    no spaCy, 1.5B available         -> regex segmentation + 1.5B classification
    neither available                -> regex segmentation + heuristic classification
    (Distiller still works in all cases — SRL is additive, not required)

Installation:
    pip install spacy
    python -m spacy download en_core_web_trf   # best for informal text
    # or
    python -m spacy download en_core_web_sm    # lighter, still useful

    # For 1.5B local model support:
    pip install llama-cpp-python
    # Place Qwen2.5-1.5B-Instruct GGUF in your models directory
    # Set VM_SRL_MODEL_PATH env var or pass model_path to SRLPipeline()
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import logging; get_logger = logging.getLogger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Optional imports — graceful degradation if not installed
# ─────────────────────────────────────────────────────────────

try:
    import spacy  # type: ignore[import]
    from spacy.language import Language as SpacyLanguage  # type: ignore[import]
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.info(
        "srl_parser: spaCy not installed — using regex segmentation. "
        "Install with: pip install spacy && python -m spacy download en_core_web_trf"
    )

try:
    from llama_cpp import Llama  # type: ignore[import]
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.info(
        "srl_parser: llama-cpp-python not installed — "
        "using heuristic value classification. "
        "Install with: pip install llama-cpp-python"
    )


# ─────────────────────────────────────────────────────────────
# Prompt injection patterns
# Stripped before any model touches the content.
# ─────────────────────────────────────────────────────────────

_PROMPT_PATTERNS = [
    # XML-style model addressing tags
    re.compile(r"<\s*(?:system|human|assistant|user|prompt|instruction)\s*>.*?</\s*(?:system|human|assistant|user|prompt|instruction)\s*>", re.IGNORECASE | re.DOTALL),
    # Role declarations
    re.compile(r"^\s*(?:system|human|assistant|user)\s*:\s*", re.IGNORECASE | re.MULTILINE),
    # Instruction imperatives directed at a model
    re.compile(r"(?:ignore\s+(?:previous|all|prior)\s+instructions?|forget\s+(?:everything|all)|you\s+are\s+now|disregard\s+(?:your|all)\s+(?:previous|prior|earlier))", re.IGNORECASE),
    # Common injection scaffolding
    re.compile(r"\[\s*(?:INST|\/INST|SYS|\/SYS|SYSTEM|END\s*SYSTEM)\s*\]", re.IGNORECASE),
    # Jinja/template markers that suggest prompt construction
    re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL),
]

# Speaker/turn labels in chatlogs — stripped because they're noise, not content
# Keeps the speaker name for context but removes structural prefix
_CHATLOG_TURN_PREFIX = re.compile(
    r"^(?:(?P<speaker>[A-Za-z0-9_\- ]{1,40})\s*[:\|]\s*)",
    re.MULTILINE,
)


# ─────────────────────────────────────────────────────────────
# Value classification
# ─────────────────────────────────────────────────────────────

class MemoryValue(str, Enum):
    DECISION    = "DECISION"    # something was explicitly decided
    FACT        = "FACT"        # a factual statement about the world/system
    PREFERENCE  = "PREFERENCE"  # explicit preference or want
    ACTION      = "ACTION"      # commitment or task to be done
    CORRECTION  = "CORRECTION"  # reversal or correction of prior content
    CONTEXT     = "CONTEXT"     # useful background, not a discrete fact
    NOISE       = "NOISE"       # not worth storing


# Heuristic keywords for fallback classification
_VALUE_HEURISTICS: List[Tuple[MemoryValue, List[str]]] = [
    (MemoryValue.CORRECTION, [
        "actually", "correction", "wait no", "scratch that", "i was wrong",
        "not that", "instead", "changed my mind", "revised", "update:",
    ]),
    (MemoryValue.DECISION, [
        "decided", "we will", "going to", "we're using", "chosen", "settled on",
        "agreed", "final decision", "we chose", "the plan is", "we'll use",
    ]),
    (MemoryValue.PREFERENCE, [
        "i want", "i prefer", "i like", "i don't want", "i hate",
        "i'd rather", "we prefer", "priority is", "important to me",
    ]),
    (MemoryValue.ACTION, [
        "todo", "to do", "action item", "need to", "will do", "i'll",
        "we need to", "must", "should", "follow up", "next step",
    ]),
    (MemoryValue.FACT, [
        "is", "are", "was", "were", "has", "have", "means", "defined as",
        "because", "therefore", "results in", "causes",
    ]),
]

_NOISE_PATTERNS = re.compile(
    r"^(?:ok|okay|sure|yes|yeah|no|nope|thanks|thank you|got it|"
    r"sounds good|great|nice|cool|lol|haha|hmm|uh|um|right|exactly|"
    r"makes sense|agreed|understood|i see|of course|absolutely)[\s.!?]*$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SemanticAnnotation:
    """
    Semantic role annotation for a single sentence.

    Roles follow a simplified PropBank-style schema:
        agent       — who is doing/deciding/saying (A0)
        action      — what is being done/decided (predicate)
        target      — what the action applies to (A1)
        circumstance — when/where/how/why (AM)
        entities    — named entities found (NER)
        value       — memory value classification
        raw         — original sentence text
    """
    raw:          str
    agent:        Optional[str]       = None
    action:       Optional[str]       = None
    target:       Optional[str]       = None
    circumstance: Optional[str]       = None
    entities:     List[str]           = field(default_factory=list)
    value:        MemoryValue         = MemoryValue.CONTEXT
    confidence:   float               = 0.5
    # True if this sentence survived the noise filter
    keep:         bool                = True


@dataclass
class ParsedDocument:
    """
    Result of running the SRL pipeline over a text block.

    The Distiller's extraction pass receives this instead of raw text
    when the SRL pipeline is active. The pre-parsed skeleton lets
    the extraction LLM focus on value judgment rather than parsing.
    """
    raw_text:          str
    sentences:         List[SemanticAnnotation]
    # Sentences worth keeping after noise filtering
    signal_count:      int  = 0
    noise_count:       int  = 0
    # Which tier handled the analysis
    spacy_used:        bool = False
    local_model_used:  bool = False

    def to_extraction_input(self) -> str:
        """
        Render the document in a format optimized for the extraction LLM.

        Instead of raw prose, the extraction LLM sees pre-parsed statements
        with their roles annotated. This dramatically reduces the cognitive
        load on the extraction pass and improves signal extraction on
        messy informal content.
        """
        lines = []
        for ann in self.sentences:
            if not ann.keep:
                continue

            parts = []
            if ann.agent:
                parts.append("AGENT: %s" % ann.agent)
            if ann.action:
                parts.append("ACTION: %s" % ann.action)
            if ann.target:
                parts.append("TARGET: %s" % ann.target)
            if ann.circumstance:
                parts.append("WHEN/HOW: %s" % ann.circumstance)
            if ann.entities:
                parts.append("ENTITIES: %s" % ", ".join(ann.entities))
            parts.append("TYPE: %s" % ann.value.value)
            parts.append("RAW: %s" % ann.raw)

            lines.append("---\n" + "\n".join(parts))

        return "\n".join(lines) if lines else self.raw_text

    def kept_sentences(self) -> List[SemanticAnnotation]:
        return [s for s in self.sentences if s.keep]


# ─────────────────────────────────────────────────────────────
# Prompt stripping
# ─────────────────────────────────────────────────────────────

def strip_prompts(text: str) -> str:
    """
    Remove prompt-like content from raw text before any model sees it.

    This is the prompt-free data zone enforcement layer. Runs on all
    content at ingestion time, regardless of source. Prompts only exist
    at the system level where we control them. They never live in the
    data layer.

    Two categories removed:
    1. Injection-style content — anything that could redirect model behavior
    2. Chatlog structural noise — turn prefixes, role labels, UI metadata

    Speaker names in chatlogs are preserved inline when the content is
    worth keeping — "Alice: we decided X" becomes "we decided X" with
    the agent inference left to the SRL pass.
    """
    # Remove injection patterns
    for pattern in _PROMPT_PATTERNS:
        text = pattern.sub(" ", text)

    # Strip chatlog turn prefixes (keep content, drop structural prefix)
    text = _CHATLOG_TURN_PREFIX.sub("", text)

    # Normalize whitespace introduced by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# Sentence segmentation fallback (no spaCy)
# ─────────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)")


def _segment_sentences_regex(text: str) -> List[str]:
    """
    Rough sentence segmentation without spaCy.
    Good enough for chunked distillation input.
    """
    sentences = _SENTENCE_SPLIT.split(text)
    out = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10:  # skip fragments
            out.append(s)
    return out or [text]


# ─────────────────────────────────────────────────────────────
# Heuristic value classification (no local model)
# ─────────────────────────────────────────────────────────────

def _classify_heuristic(text: str) -> Tuple[MemoryValue, float]:
    """
    Fast heuristic classification when the 1.5B is not available.
    Returns (MemoryValue, confidence).
    """
    lower = text.lower().strip()

    # Noise check first — common filler responses
    if _NOISE_PATTERNS.match(lower) or len(lower) < 8:
        return MemoryValue.NOISE, 0.9

    # Walk priority-ordered heuristics
    for value, keywords in _VALUE_HEURISTICS:
        if any(kw in lower for kw in keywords):
            return value, 0.6

    # Default to CONTEXT — it exists and might be useful
    return MemoryValue.CONTEXT, 0.4


# ─────────────────────────────────────────────────────────────
# spaCy role extraction
# ─────────────────────────────────────────────────────────────

def _extract_roles_spacy(sent) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract (agent, action, target, circumstance) from a spaCy Span.

    Uses dependency parse labels:
        nsubj / nsubjpass -> agent
        ROOT (verb)       -> action
        dobj / attr / xcomp / ccomp -> target
        advmod / prep + pobj        -> circumstance
    """
    agent       = None
    action      = None
    target      = None
    circumstance_parts = []

    root = None
    for token in sent:
        if token.dep_ == "ROOT":
            root = token
            action = token.lemma_
            break

    if root is None:
        return None, None, None, None

    for token in sent:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head == root:
            # Include compound modifiers in the agent span
            agent_tokens = [t.text for t in token.subtree
                            if t.dep_ in ("compound", "amod", "det") or t == token]
            agent = " ".join(agent_tokens)

        elif token.dep_ in ("dobj", "attr", "xcomp", "ccomp", "pcomp") and token.head == root:
            target_tokens = [t.text for t in token.subtree]
            target = " ".join(target_tokens)[:120]  # cap length

        elif token.dep_ in ("advmod", "prep") and token.head == root:
            circ = " ".join(t.text for t in token.subtree)
            if len(circ) < 80:
                circumstance_parts.append(circ)

    circumstance = "; ".join(circumstance_parts) if circumstance_parts else None
    return agent, action, target, circumstance


# ─────────────────────────────────────────────────────────────
# Local model classification prompt
# ─────────────────────────────────────────────────────────────

_LOCAL_MODEL_SYSTEM = """You classify statements for a memory system.
Given a pre-parsed statement, output ONLY one of these labels:
DECISION FACT PREFERENCE ACTION CORRECTION CONTEXT NOISE

Rules:
DECISION  = something was explicitly chosen or decided
FACT      = a factual statement about the world or system
PREFERENCE = an explicit want, preference, or priority
ACTION    = a commitment or task to be completed
CORRECTION = reversal or correction of prior information
CONTEXT   = useful background but not a discrete fact
NOISE     = filler, pleasantry, or content with no memory value

Output the label only. No explanation."""


def _classify_local_model(llm, annotation: SemanticAnnotation) -> Tuple[MemoryValue, float]:
    """
    Use the 1.5B local model to classify memory value for a pre-parsed statement.

    The model receives the structured role annotation, not raw prose.
    This is a narrow classification task well within the 1.5B's capability.
    """
    input_text = "RAW: %s" % annotation.raw
    if annotation.agent:
        input_text += "\nAGENT: %s" % annotation.agent
    if annotation.action:
        input_text += "\nACTION: %s" % annotation.action
    if annotation.target:
        input_text += "\nTARGET: %s" % annotation.target

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _LOCAL_MODEL_SYSTEM},
                {"role": "user", "content": input_text},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        label = response["choices"][0]["message"]["content"].strip().upper()

        # Validate against known values
        try:
            return MemoryValue(label), 0.75
        except ValueError:
            # Model output an unexpected label — fall back to heuristic
            return _classify_heuristic(annotation.raw)

    except Exception as exc:
        logger.debug("srl_parser: local model classification failed: %s", exc)
        return _classify_heuristic(annotation.raw)


# ─────────────────────────────────────────────────────────────
# SRL Pipeline
# ─────────────────────────────────────────────────────────────

class SRLPipeline:
    """
    Semantic Role Labeling pre-extraction pipeline.

    Sits between raw content ingestion and the Distiller's chunking pass.
    Produces an ParsedDocument with annotated sentences that the extraction
    LLM receives instead of raw prose.

    Args:
        model_name:     spaCy model to load.
                        "en_core_web_trf" recommended for informal text.
                        "en_core_web_sm" as a lighter fallback.
                        If the model is not installed, falls back to regex.
        model_path:     Path to a GGUF file for the local 1.5B model.
                        Reads from VM_SRL_MODEL_PATH env var if not provided.
                        If not set or not found, skips local model classification.
        n_gpu_layers:   GPU layers for the local model. 0 = CPU only.
                        Reads from VM_SRL_GPU_LAYERS env var (default 0).
        noise_threshold: Annotations with confidence below this for NOISE
                         classification are dropped. Default 0.7.
        strip_prompts_enabled: Run prompt stripping pass. Default True.
                               Disable only for testing.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_trf",
        model_path: Optional[str] = None,
        n_gpu_layers: int = 0,
        noise_threshold: float = 0.7,
        strip_prompts_enabled: bool = True,
    ):
        self.noise_threshold = noise_threshold
        self.strip_prompts_enabled = strip_prompts_enabled

        # ── spaCy setup ───────────────────────────────────────
        self._nlp: Optional["SpacyLanguage"] = None
        if _SPACY_AVAILABLE:
            self._nlp = self._load_spacy(model_name)

        # ── Local model setup ─────────────────────────────────
        self._llm = None
        if _LLAMA_AVAILABLE:
            path = model_path or os.environ.get("VM_SRL_MODEL_PATH")
            if path and os.path.isfile(path):
                gpu_layers = int(
                    os.environ.get("VM_SRL_GPU_LAYERS", str(n_gpu_layers))
                )
                self._llm = self._load_local_model(path, gpu_layers)
            elif path:
                logger.warning(
                    "srl_parser: VM_SRL_MODEL_PATH set but file not found: %s "
                    "— falling back to heuristic classification",
                    path,
                )

        tier = []
        if self._nlp:
            tier.append("spaCy(%s)" % model_name)
        else:
            tier.append("regex-segmentation")
        if self._llm:
            tier.append("local-1.5B")
        else:
            tier.append("heuristic-classification")

        logger.info("srl_parser: initialized [%s]", " + ".join(tier))

    # ── Public API ────────────────────────────────────────────

    def process(self, text: str, fast_mode: bool = False) -> ParsedDocument:
        """
        Run the SRL pipeline over a text block.

        Args:
            text:       Raw text to analyse.
            fast_mode:  When True, skip the local 1.5B model and use the
                        heuristic classifier instead. The dependency parse
                        (spaCy) still runs in full so action_rate and role
                        extraction are accurate. Use this at chunk time where
                        you need the structural density signal quickly and can
                        defer precise value classification to a background pass.

        Returns an ParsedDocument ready to hand to the Distiller.
        """
        # Step 1 — prompt stripping
        if self.strip_prompts_enabled:
            text = strip_prompts(text)

        # Step 2 — parse or segment
        if self._nlp is not None:
            annotations = self._parse_spacy(text)
            spacy_used = True
        else:
            annotations = self._parse_regex(text)
            spacy_used = False

        # Step 3 — value classification
        # fast_mode forces heuristic classification regardless of whether the
        # 1.5B is loaded. Heuristic is keyword matching — effectively free.
        # The 1.5B is reserved for the background enrichment pass.
        use_local_model = self._llm is not None and not fast_mode
        for ann in annotations:
            if use_local_model:
                ann.value, ann.confidence = _classify_local_model(self._llm, ann)
            else:
                ann.value, ann.confidence = _classify_heuristic(ann.raw)

        # Step 4 — noise filtering
        signal_count = 0
        noise_count = 0
        for ann in annotations:
            if ann.value == MemoryValue.NOISE and ann.confidence >= self.noise_threshold:
                ann.keep = False
                noise_count += 1
            else:
                signal_count += 1

        doc = ParsedDocument(
            raw_text=text,
            sentences=annotations,
            signal_count=signal_count,
            noise_count=noise_count,
            spacy_used=spacy_used,
            local_model_used=use_local_model,
        )

        logger.debug(
            "srl_parser: processed %d sentences (%d signal, %d noise filtered)%s",
            len(annotations), signal_count, noise_count,
            " [fast_mode]" if fast_mode else "",
        )

        return doc

    def process_batch(
        self, texts: List[str], fast_mode: bool = False
    ) -> List[ParsedDocument]:
        """
        Process multiple texts efficiently using spaCy's nlp.pipe().

        Equivalent to calling process() on each text individually, but
        substantially faster when processing many texts at once because
        spaCy batches the computation internally — particularly significant
        with transformer models (en_core_web_trf) where GPU batching matters.

        Maintains the same fast_mode semantics as process(): when True,
        heuristic classification is used regardless of whether the 1.5B
        is loaded. Use fast_mode=True at chunk time; leave it False in
        the background SRLEnricher where latency is not a concern.

        Args:
            texts:      List of raw text blocks to process.
            fast_mode:  Skip local 1.5B model, use heuristic classification.

        Returns:
            List of ParsedDocuments in the same order as the input texts.
        """
        if not texts:
            return []

        if self.strip_prompts_enabled:
            texts = [strip_prompts(t) for t in texts]

        use_local_model = self._llm is not None and not fast_mode

        if self._nlp is not None:
            # Single pass through spaCy for all texts — the core speedup.
            all_annotations = [
                self._annotate_spacy_doc(doc)
                for doc in self._nlp.pipe(texts)
            ]
            spacy_used = True
        else:
            all_annotations = [self._parse_regex(t) for t in texts]
            spacy_used = False

        results = []
        for text, annotations in zip(texts, all_annotations):
            for ann in annotations:
                if use_local_model:
                    ann.value, ann.confidence = _classify_local_model(self._llm, ann)
                else:
                    ann.value, ann.confidence = _classify_heuristic(ann.raw)

            signal_count = noise_count = 0
            for ann in annotations:
                if ann.value == MemoryValue.NOISE and ann.confidence >= self.noise_threshold:
                    ann.keep = False
                    noise_count += 1
                else:
                    signal_count += 1

            results.append(ParsedDocument(
                raw_text=text,
                sentences=annotations,
                signal_count=signal_count,
                noise_count=noise_count,
                spacy_used=spacy_used,
                local_model_used=use_local_model,
            ))

        logger.debug(
            "srl_parser: batch processed %d texts%s",
            len(texts), " [fast_mode]" if fast_mode else "",
        )
        return results

    def process_chunks(self, chunks: List[str]) -> List[ParsedDocument]:
        """
        Process a list of pre-chunked text blocks.
        Used when the Distiller has already chunked content before SRL.
        """
        return [self.process(chunk) for chunk in chunks]

    # ── Internal: spaCy parse ─────────────────────────────────

    def _annotate_spacy_doc(self, doc) -> List[SemanticAnnotation]:
        """
        Extract SemanticAnnotations from an already-processed spaCy Doc.

        Separated from _parse_spacy so process_batch() can reuse it across
        documents produced by nlp.pipe() without re-entering the spaCy pipeline.
        """
        annotations = []
        for sent in doc.sents:
            raw = sent.text.strip()
            if len(raw) < 8:
                continue

            agent, action, target, circumstance = _extract_roles_spacy(sent)

            entities = [
                ent.text for ent in sent.ents
                if ent.label_ not in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT")
            ]

            annotations.append(SemanticAnnotation(
                raw=raw,
                agent=agent,
                action=action,
                target=target,
                circumstance=circumstance,
                entities=entities,
            ))

        return annotations

    def _parse_spacy(self, text: str) -> List[SemanticAnnotation]:
        assert self._nlp is not None
        return self._annotate_spacy_doc(self._nlp(text))

    def _parse_regex(self, text: str) -> List[SemanticAnnotation]:
        """Fallback when spaCy is not installed."""
        sentences = _segment_sentences_regex(text)
        return [
            SemanticAnnotation(raw=s.strip())
            for s in sentences
            if s.strip()
        ]

    # ── Internal: model loading ───────────────────────────────

    def _load_spacy(self, model_name: str) -> Optional["SpacyLanguage"]:
        """
        Load spaCy model with graceful fallback.

        Tries the requested model first, then falls back through
        en_core_web_sm -> blank English pipeline if nothing is installed.
        """
        for name in [model_name, "en_core_web_sm", "en_core_web_md"]:
            try:
                nlp = spacy.load(name)
                logger.info("srl_parser: loaded spaCy model %s", name)
                return nlp
            except OSError:
                continue

        logger.warning(
            "srl_parser: no spaCy model found. "
            "Install with: python -m spacy download %s",
            model_name,
        )
        return None

    def _load_local_model(self, path: str, n_gpu_layers: int):
        """Load local GGUF model via llama-cpp-python."""
        try:
            llm = Llama(
                model_path=path,
                n_ctx=2048,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info(
                "srl_parser: loaded local model %s (gpu_layers=%d)",
                os.path.basename(path), n_gpu_layers,
            )
            return llm
        except Exception as exc:
            logger.warning(
                "srl_parser: could not load local model %s: %s "
                "— falling back to heuristic classification",
                path, exc,
            )
            return None


# ─────────────────────────────────────────────────────────────
# Convenience: process text without instantiating SRLPipeline
# ─────────────────────────────────────────────────────────────

_default_srl_parser: Optional[SRLPipeline] = None


def get_default_srl_parser(
    model_name: str = "en_core_web_trf",
    model_path: Optional[str] = None,
) -> SRLPipeline:
    """
    Get or create the module-level default pipeline.

    Lazy-initialized on first call. Subsequent calls return the
    same instance, so the spaCy model and local model are only
    loaded once per process.
    """
    global _default_srl_parser
    if _default_srl_parser is None:
        _default_srl_parser = SRLPipeline(
            model_name=model_name,
            model_path=model_path,
        )
    return _default_srl_parser


def analyze(text: str, model_name: str = "en_core_web_trf") -> ParsedDocument:
    """
    One-line convenience wrapper. Useful for testing and one-off analysis.

        from veritas_memoria.srl_parser import analyze
        doc = analyze("We decided to use BM25 for lexical search.")
        print(doc.to_extraction_input())
    """
    return get_default_srl_parser(model_name=model_name).process(text)
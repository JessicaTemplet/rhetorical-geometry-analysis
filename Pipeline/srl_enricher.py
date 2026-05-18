"""
VeritasMemoria - Background SRL Enricher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decouples SRL value-classification from the ingestion write path.

At ingest time, AdaptiveChunker runs with srl_enabled=False so chunks
are sized using the fast heuristic density signal. After chunks are
stored, the SRLEnricher processes each chunk's text in a background
thread, runs SRL, and calls a provided callback with the enrichment
metadata so the caller can write it back to the store.

Why decouple?

    SRL (spaCy + optional 1.5B model) is expensive per sentence. Running
    it on every paragraph of a 3MB document during the ingestion critical
    path adds latency that doesn't improve chunking quality enough to
    justify it — the heuristic verb-density signal already captures the
    density band correctly for most content. The SRL value classifications
    (DECISION, FACT, ACTION, etc.) are most useful at retrieval time for
    ranking and filtering. They can be computed asynchronously.

Usage::

    from veritas_memoria.core.write_path.srl_enricher import SRLEnricher

    def update_metadata(memory_id, srl_meta):
        vm.update_memory_metadata(memory_id, srl_meta)

    enricher = SRLEnricher(on_enriched=update_metadata)
    enricher.submit(memory_id="abc123", chunk_text="We decided to use BM25...")
    enricher.submit(memory_id="def456", chunk_text="The index rebuilds nightly...")
    # ...
    enricher.shutdown(wait=True)   # block until queue drains at process exit

    # Or: fire and forget — daemon thread exits with the process automatically.
    enricher.shutdown(wait=False)

Enrichment metadata keys written per chunk::

    srl_action_rate:      float  — proportion of sentences with a parsed ROOT verb
    srl_high_value_ratio: float  — proportion of kept sentences classified as
                                   DECISION / FACT / ACTION / CORRECTION
    srl_signal_count:     int    — sentences that survived noise filtering
    srl_noise_count:      int    — sentences filtered as noise
    srl_density:          float  — computed density value (same formula as chunker)
    srl_value_types:      list   — MemoryValue strings for kept sentences, e.g.
                                   ["DECISION", "FACT", "CONTEXT"]
    srl_enriched:         bool   — True (marker so callers can detect enriched chunks)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel pushed to the queue to signal the worker to stop cleanly.
_STOP = object()


def _run_srl_on_chunk(chunk_text: str) -> Optional[Dict]:
    """
    Run SRL on a single chunk and return enrichment metadata.

    Returns None if SRL is unavailable or fails, so the caller can skip
    the update rather than writing empty/misleading metadata.
    """
    try:
        from srl import SRLPipeline, MemoryValue

        # Re-use the module-level singleton from adaptive_chunker if it has
        # already been constructed; otherwise build one here. Either way the
        # spaCy model is only loaded once per process.
        try:
            from adaptive_chunker import _get_srl
            srl = _get_srl()
        except Exception:
            srl = SRLPipeline()

        if srl is None:
            return None

        doc = srl.process(chunk_text)

        total = len(doc.sentences)
        if total == 0:
            return None

        action_rate = sum(
            1 for s in doc.sentences if s.action is not None
        ) / total

        _high_value_types = {
            MemoryValue.DECISION,
            MemoryValue.FACT,
            MemoryValue.ACTION,
            MemoryValue.CORRECTION,
        }
        kept = doc.kept_sentences()
        high_value_ratio = (
            sum(1 for s in kept if s.value in _high_value_types)
            / max(len(kept), 1)
        )

        # Same formula used in AdaptiveChunker._compute_density SRL branch,
        # excluding the graph-level terms that are always 0 for raw text.
        verb_density = action_rate * 6.0 + high_value_ratio * 6.0
        srl_density = 0.25 * 0.25 + verb_density * 0.3   # coherence_bonus=0.25, betti_1=0

        value_types: List[str] = [s.value.value for s in kept]

        return {
            "srl_action_rate":      round(action_rate, 4),
            "srl_high_value_ratio": round(high_value_ratio, 4),
            "srl_signal_count":     doc.signal_count,
            "srl_noise_count":      doc.noise_count,
            "srl_density":          round(srl_density, 4),
            "srl_value_types":      value_types,
            "srl_enriched":         True,
        }

    except Exception as exc:
        logger.debug("SRLEnricher: SRL failed on chunk (%s) — skipping.", exc)
        return None


class SRLEnricher:
    """
    Background SRL enrichment worker.

    Accepts (memory_id, chunk_text) submissions via .submit() and processes
    them in a single daemon thread, calling on_enriched(memory_id, metadata)
    for each result.

    Thread safety: submit() is safe to call from any thread. The worker
    serialises all SRL calls so spaCy's model is accessed from one thread.

    Args:
        on_enriched:    Callback invoked with (memory_id: str, metadata: dict)
                        for each successfully enriched chunk. Called from the
                        worker thread — make sure the callback is thread-safe
                        (or simply queue the result and handle it on the main
                        thread if your store is not thread-safe).
        maxsize:        Maximum queue depth before submit() blocks.
                        0 (default) means unbounded.
    """

    def __init__(
        self,
        on_enriched: Callable[[str, Dict], None],
        maxsize: int = 0,
    ):
        self._on_enriched = on_enriched
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._worker,
            name="srl-enricher",
            daemon=True,   # exits automatically with the process
        )
        self._thread.start()
        logger.info("SRLEnricher: background worker started.")

    def submit(self, memory_id: str, chunk_text: str) -> None:
        """
        Enqueue a chunk for background SRL enrichment.

        Returns immediately. If maxsize is set and the queue is full,
        blocks until space is available.
        """
        self._queue.put((memory_id, chunk_text))

    def submit_batch(self, items: List[tuple]) -> None:
        """
        Enqueue multiple (memory_id, chunk_text) pairs at once.

        Convenience wrapper around submit().
        """
        for memory_id, chunk_text in items:
            self.submit(memory_id, chunk_text)

    def shutdown(self, wait: bool = True) -> None:
        """
        Signal the worker to stop after draining the queue.

        Args:
            wait: If True, block until the worker thread exits.
                  If False, return immediately (daemon thread will exit
                  with the process).
        """
        self._queue.put(_STOP)
        if wait:
            self._thread.join()
            logger.info("SRLEnricher: worker stopped.")

    def pending(self) -> int:
        """Approximate number of items still waiting in the queue."""
        return self._queue.qsize()

    # ── Worker ────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    break

                memory_id, chunk_text = item
                metadata = _run_srl_on_chunk(chunk_text)

                if metadata is not None:
                    try:
                        self._on_enriched(memory_id, metadata)
                    except Exception as cb_exc:
                        logger.warning(
                            "SRLEnricher: on_enriched callback failed for %s: %s",
                            memory_id, cb_exc,
                        )
                else:
                    logger.debug(
                        "SRLEnricher: no SRL result for memory %s — skipping update.",
                        memory_id,
                    )
            except Exception as exc:
                logger.error("SRLEnricher: unexpected worker error: %s", exc)
            finally:
                self._queue.task_done()

"""
VeritasMemoria - Graph Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enums, dataclasses, and the SQLite schema constant used by the graph engine.

Importing this module alone is enough to work with graph data structures
without pulling in the full engine (sqlite3 connection, bloom filters, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class GraphZone(str, Enum):
    """
    The four HSH knowledge zones. Each is a separate traversable subgraph
    embedded in a hyperbolic Poincare disk. Lambda (zone inertia) determines
    how close to the governance anchor a zone sits — higher Lambda = closer
    to center = more stable and authoritative.

    GOVERNANCE         (lambda=1.0) — immutable standing directives and compliance
                                      obligations. Human-only writes, always signed.
                                      Sits at the gravitational center of the disk.

    RATIONALE          (lambda=0.65) — structured reasoning artifacts: statutes,
                                      precedents, protocols, lessons learned, and
                                      the agent's own recorded rationale chains.
                                      Authority source (internal vs external) is a
                                      metadata flag, not a zone distinction.
                                      Human-only writes; VG consolidation pass
                                      permitted (ORC-driven rewiring is valid for
                                      evolving rationale graphs).

    WORK_KNOWLEDGE     (lambda=0.55) — matter-scoped persistent facts. Client
                                      preferences, case facts, patient history.
                                      LLM writes during session; sleep pass promotes
                                      or prunes; human can always write.

    TEMPORAL_KNOWLEDGE (lambda=0.35) — time-bounded ephemera. Deadlines,
                                      appointments, active task lists. LLM writes
                                      freely; sleep pass prunes/expires nodes to
                                      shadow graph; nodes near the disk boundary
                                      where hyperbolic distance to governance is
                                      effectively infinite.

    Decision tree for zone assignment (evaluated in priority order):
        1. Has expiry date?             -> TEMPORAL_KNOWLEDGE
        2. Matter/client specific?      -> WORK_KNOWLEDGE
        3. System directive?            -> GOVERNANCE
        4. Reasoning chain / knowledge? -> RATIONALE
    """
    GOVERNANCE         = "governance"
    RATIONALE          = "rationale"
    WORK_KNOWLEDGE     = "work_knowledge"
    TEMPORAL_KNOWLEDGE = "temporal_knowledge"


class EdgeKind(str, Enum):
    """
    Typed relationships. The type is part of the edge's meaning,
    not just a label.
    """
    SUPPORTS                      = "supports"
    CONTRADICTS                   = "contradicts"
    DEPENDS_ON                    = "depends_on"
    IMPLEMENTS                    = "implements"
    ABOUT                         = "about"
    TEMPORAL_NEXT                 = "temporal_next"
    REFINES                       = "refines"
    DUPLICATE_OF                  = "duplicate_of"
    EVIDENCE_SUPPORTS_DECISION    = "evidence_supports_decision"
    DECISION_UPDATES_PREFERENCE   = "decision_updates_preference"
    FACT_UPDATES_BELIEF           = "fact_updates_belief"
    DECISION_REFINES_POLICY       = "decision_refines_policy"
    EVIDENCE_CONTRADICTS_PREF     = "evidence_contradicts_preference"
    DECISION_REQUIRES_EVIDENCE    = "decision_requires_evidence"
    # Temporary — only used in deep semantic mode, never persisted
    SEMANTIC_SIMILARITY           = "semantic_similarity"


class GateLevel(str, Enum):
    """
    Access control on edges.

    NONE                 — always traversable
    SUGGEST              — traversable but caller is informed it's a suggestion
    REVIEW_REQUIRED      — traversable only if caller explicitly opts in
    BLOCK_UNTIL_RESOLVED — NEVER traversable; human sign-off required first

    BLOCK_UNTIL_RESOLVED is set automatically when a contradiction is detected.
    It cannot be cleared by any automated process — only by human resolution
    with a signature recorded in the audit log.
    """
    NONE                 = "none"
    SUGGEST              = "suggest"
    REVIEW_REQUIRED      = "review_required"
    BLOCK_UNTIL_RESOLVED = "block_until_resolved"


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Edge:
    graph:      GraphZone
    src_id:     str
    dst_id:     str
    kind:       EdgeKind
    weight:     float          = 1.0
    gate:       GateLevel      = GateLevel.NONE
    confidence: float          = 1.0
    source_ids: List[str]      = field(default_factory=list)
    created_at: str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # For contradiction edges: snapshot of both versions at detection time
    snapshot:   Optional[str]  = None  # JSON string
    resolved:   bool           = False
    resolved_by: Optional[str] = None  # actor_id
    resolved_at: Optional[str] = None
    resolution_type: Optional[str] = None  # "corrected" | "disputed" | "evidential"


@dataclass
class BridgeEdge:
    from_graph:  GraphZone
    from_id:     str
    to_graph:    GraphZone
    to_id:       str
    kind:        EdgeKind
    weight:      float     = 1.0
    gate:        GateLevel = GateLevel.NONE
    confidence:  float     = 1.0
    source_ids:  List[str] = field(default_factory=list)
    created_at:  str       = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TempEdge:
    """
    Exists only during a deep semantic or overlay query.
    Never written to storage. Discarded after the call returns.
    """
    src_id:     str
    dst_id:     str
    similarity: float   # 0.0–1.0 cosine similarity
    zone:       GraphZone = GraphZone.RATIONALE


@dataclass
class BridgePolicy:
    allow_suggest:          bool = True
    allow_review_required:  bool = False
    allow_blocked:          bool = False  # NEVER True in normal operation


@dataclass
class ContradictionRecord:
    node_a_id:      str
    node_b_id:      str
    zone:           GraphZone
    snapshot_a:     str   # content of node A at detection time
    snapshot_b:     str   # content of node B at detection time
    detected_at:    str
    edge_id:        str   # the Contradicts edge that was created
    resolved:       bool  = False
    resolved_by:    Optional[str] = None
    resolved_at:    Optional[str] = None
    resolution_type: Optional[str] = None
    winner_id:      Optional[str] = None
    signature:      Optional[str] = None


@dataclass
class IlluminationResult:
    """
    Result from overlay/illumination mode.
    Nodes ranked by how many retrieval methods independently surfaced them.
    """
    node_id:          str
    zone:             GraphZone
    convergence_score: float        # how many methods found this, weighted
    bm25_score:       float = 0.0
    hierarchy_score:  float = 0.0
    graph_centrality: float = 0.0
    semantic_score:   float = 0.0
    direct_query_hits: int  = 0    # times this was the direct answer to a query
    # Low direct_query_hits + high convergence = the "nagging" signal
    nagging_score:    float = 0.0


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_edges (
    id                  TEXT PRIMARY KEY,
    graph               TEXT NOT NULL,
    src_id              TEXT NOT NULL,
    dst_id              TEXT NOT NULL,
    kind                TEXT NOT NULL,
    weight              REAL NOT NULL DEFAULT 1.0,
    gate                TEXT NOT NULL DEFAULT 'none',
    confidence          REAL NOT NULL DEFAULT 1.0,
    source_ids_json     TEXT NOT NULL DEFAULT '[]',
    snapshot_json       TEXT,
    resolved            INTEGER NOT NULL DEFAULT 0,
    resolved_by         TEXT,
    resolved_at         TEXT,
    resolution_type     TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE(graph, src_id, dst_id, kind)
);

CREATE TABLE IF NOT EXISTS bridge_edges (
    id                  TEXT PRIMARY KEY,
    from_graph          TEXT NOT NULL,
    from_id             TEXT NOT NULL,
    to_graph            TEXT NOT NULL,
    to_id               TEXT NOT NULL,
    kind                TEXT NOT NULL,
    weight              REAL NOT NULL DEFAULT 1.0,
    gate                TEXT NOT NULL DEFAULT 'none',
    confidence          REAL NOT NULL DEFAULT 1.0,
    source_ids_json     TEXT NOT NULL DEFAULT '[]',
    created_at          TEXT NOT NULL,
    UNIQUE(from_graph, from_id, to_graph, to_id, kind)
);

CREATE TABLE IF NOT EXISTS contradiction_records (
    id              TEXT PRIMARY KEY,
    node_a_id       TEXT NOT NULL,
    node_b_id       TEXT NOT NULL,
    zone            TEXT NOT NULL,
    snapshot_a      TEXT NOT NULL,
    snapshot_b      TEXT NOT NULL,
    detected_at     TEXT NOT NULL,
    edge_id         TEXT NOT NULL,
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolved_by     TEXT,
    resolved_at     TEXT,
    resolution_type TEXT,
    winner_id       TEXT,
    signature       TEXT
);

CREATE TABLE IF NOT EXISTS node_query_hits (
    node_id     TEXT NOT NULL,
    zone        TEXT NOT NULL,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    last_hit    TEXT,
    PRIMARY KEY (node_id, zone)
);

CREATE TABLE IF NOT EXISTS graph_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src   ON graph_edges(graph, src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst   ON graph_edges(graph, dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_gate  ON graph_edges(gate);
CREATE INDEX IF NOT EXISTS idx_bridges_from ON bridge_edges(from_graph, from_id);
CREATE INDEX IF NOT EXISTS idx_contradictions_unresolved
    ON contradiction_records(resolved) WHERE resolved = 0;
"""

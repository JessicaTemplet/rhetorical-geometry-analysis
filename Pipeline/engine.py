"""
VeritasMemoria - Graph Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-zone knowledge graph with typed edges, gate levels, bloom-filtered
retrieval, deep semantic expansion, and overlay illumination mode.

Use cases:
    - Representing a long-lived agent's identity, work state, and beliefs.
    - Enforcing hard governance constraints (no silent cross-zone leakage).
    - Surfacing "nagging" but unqueried nodes via illumination mode.

Architecture:
    - Four zones, each a separate subgraph:
        * Identity   — preferences, values, biography.
        * Work       — tasks, decisions, case files, project state.
        * Knowledge  — learned facts, precedents, research.
        * Governance — non-negotiable directives, policies, registries.
    - Bridge edges connect zones with explicit gate levels and provenance.
    - No cross-zone relationship is created without a declared reason.

Retrieval modes:
    - standard:
        BFS traversal within zones, plus optional bridge hops.
    - deep_semantic:
        standard + temporary similarity edges from the embedding index
        (never persisted; discarded after the query returns).
    - overlay / illumination:
        combines BM25/lexical, hierarchy, semantic, and graph centrality
        signals to surface high-convergence nodes that are rarely or never
        directly queried ("nagging feeling" mode).

Hard guarantees (hold for all traversal methods exposed by this engine;
does not cover direct DB access outside this module):
    - Contradictions within a zone:
        auto-create a CONTRADICTS edge with GateLevel.BLOCK_UNTIL_RESOLVED.
        Blocked nodes are excluded from traversal until human sign-off.
    - Gate checks on read:
        gate levels are checked at traversal time; blocked edges are never
        crossed regardless of caller intent.
    - Provenance:
        BLOCK_UNTIL_RESOLVED edges require source_ids (enforced in add_edge).
        Governed edge kinds require provenance even at softer gate levels.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


from veritas_memoria.core.graph.bloom_filter import BloomFilter
from graph_types import (
    BridgeEdge,
    BridgePolicy,
    ContradictionRecord,
    Edge,
    EdgeKind,
    GateLevel,
    GraphZone,
    GRAPH_SCHEMA,
    IlluminationResult,
    TempEdge,
)
from veritas_memoria.core.graph.shadow_graph import ShadowGraph, ArchiveReason

logger = logging.getLogger(__name__)


class GraphEngine:
    """
    The core graph system for VeritasMemoria.

    Manages four zone subgraphs, bridge edges between zones,
    contradiction detection and gating, and three retrieval modes.

    All writes go through the SQLite backend for persistence.
    In-memory bloom filters and adjacency caches are rebuilt on init
    and updated incrementally on every write.
    """

    def __init__(self, db_path: str, audit_logger=None):
        self.db_path = db_path
        self.audit = audit_logger
        self._conn: Optional[sqlite3.Connection] = None

        # Per-zone bloom filters for fast candidate gating
        self._blooms: Dict[GraphZone, BloomFilter] = {
            z: BloomFilter(capacity=200_000, error_rate=0.01)
            for z in GraphZone
        }

        # In-memory adjacency for fast traversal
        # zone -> src_id -> list of (dst_id, kind, weight, gate)
        self._adj: Dict[GraphZone, Dict[str, List[Tuple]]] = {
            z: defaultdict(list) for z in GraphZone
        }

        # Reverse adjacency (in-edges index)
        # zone -> dst_id -> list of (src_id, kind, weight, gate)
        # Maintained as the transpose of _adj so that directed curvature
        # and any algorithm needing "who points at this node?" can run in
        # O(in-degree) rather than O(N). Kept in sync with _adj at all
        # write points: _rebuild_cache() and add_edge().
        self._adj_in: Dict[GraphZone, Dict[str, List[Tuple]]] = {
            z: defaultdict(list) for z in GraphZone
        }

        # Bridge adjacency: (from_zone, from_id) -> list of (to_zone, to_id, weight, gate)
        self._bridges: Dict[Tuple, List[Tuple]] = defaultdict(list)

        # Unresolved contradictions cache (node_id -> ContradictionRecord)
        self._blocked_nodes: Dict[str, ContradictionRecord] = {}

        # Temp edge recurrence tracking — persists for the lifetime of the process.
        # Keys are (src_id, dst_id) tuples; values track count, running average
        # similarity, zone, and last-seen timestamp.
        # The Calibrator reads this via get_recurring_temp_edge_candidates() to
        # surface node pairs worth promoting to permanent bridge edges.
        self._temp_edge_counts: Dict[Tuple[str, str], Dict] = {}

        self._init_db()
        self._rebuild_cache()

        # ── Shadow graph (append-only audit/retention layer) ─────────────────
        # Shares the same SQLite connection so every shadow write is in the
        # same WAL journal. 7-year default retention; only purge_expired()
        # can hard-delete and it requires a signed call.
        self.shadow = ShadowGraph(
            conn=self._conn_get(),
            retention_years=7,
            audit_logger=audit_logger,
        )

    # ── Database ────────────────────────────────────────────

    def _conn_get(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        conn = self._conn_get()
        conn.executescript(GRAPH_SCHEMA)
        conn.commit()

    # Stale zone values that existed in earlier schema versions.
    # Map them to their current equivalents so old databases load cleanly.
    _ZONE_MIGRATIONS: dict = {
        "knowledge": "rationale",
        "identity": "work_knowledge",
        "work": "work_knowledge",
        "temporal": "temporal_knowledge",
    }

    def _coerce_zone(self, raw: str) -> GraphZone:
        """
        Convert a raw zone string to a GraphZone, handling stale values from
        older schema versions. Logs a warning and remaps rather than crashing.
        """
        remapped = self._ZONE_MIGRATIONS.get(raw, raw)
        if remapped != raw:
            logger.warning(
                "Stale GraphZone value %r found in DB — remapping to %r. "
                "Run a migration to update these rows permanently.",
                raw, remapped,
            )
        try:
            return GraphZone(remapped)
        except ValueError:
            logger.error(
                "Unknown GraphZone value %r (remapped from %r) — skipping edge. "
                "Add an entry to _ZONE_MIGRATIONS or run a schema migration.",
                remapped, raw,
            )
            return None

    def _rebuild_cache(self) -> None:
        """Rebuild in-memory adjacency and bloom filters from DB on startup."""
        conn = self._conn_get()

        for row in conn.execute("SELECT graph, src_id, dst_id, kind, weight, gate FROM graph_edges"):
            zone = self._coerce_zone(row["graph"])
            if zone is None:
                continue
            kind   = EdgeKind(row["kind"])
            weight = row["weight"]
            gate   = GateLevel(row["gate"])
            self._adj[zone][row["src_id"]].append((
                row["dst_id"], kind, weight, gate
            ))
            # Reverse index: dst -> [(src, kind, weight, gate)]
            self._adj_in[zone][row["dst_id"]].append((
                row["src_id"], kind, weight, gate
            ))
            self._blooms[zone].add(row["src_id"])
            self._blooms[zone].add(row["dst_id"])

        for row in conn.execute("SELECT from_graph, from_id, to_graph, to_id, weight, gate FROM bridge_edges"):
            from_zone = self._coerce_zone(row["from_graph"])
            to_zone   = self._coerce_zone(row["to_graph"])
            if from_zone is None or to_zone is None:
                continue
            key = (from_zone, row["from_id"])
            self._bridges[key].append((
                to_zone, row["to_id"],
                row["weight"], GateLevel(row["gate"])
            ))

        # Cache unresolved contradictions
        for row in conn.execute(
            "SELECT * FROM contradiction_records WHERE resolved=0"
        ):
            rec = ContradictionRecord(
                node_a_id=row["node_a_id"],
                node_b_id=row["node_b_id"],
                zone=GraphZone(row["zone"]),
                snapshot_a=row["snapshot_a"],
                snapshot_b=row["snapshot_b"],
                detected_at=row["detected_at"],
                edge_id=row["edge_id"],
            )
            self._blocked_nodes[row["node_a_id"]] = rec
            self._blocked_nodes[row["node_b_id"]] = rec

        logger.info(f"Graph cache rebuilt — blocked nodes: {len(self._blocked_nodes)}")

    def _edge_id(self, graph, src: str, dst, kind) -> str:
        g = graph.value if hasattr(graph, 'value') else str(graph)
        d = dst.value if hasattr(dst, 'value') else str(dst)
        k = kind.value if hasattr(kind, 'value') else str(kind)
        return hashlib.sha256(
            f"{g}:{src}:{d}:{k}".encode()
        ).hexdigest()[:16]

    # ── Edge writes ──────────────────────────────────────────

    # Edge kinds that carry governance weight — provenance required
    GOVERNED_KINDS: Set[EdgeKind] = {
        EdgeKind.DECISION_REFINES_POLICY,
        EdgeKind.DECISION_UPDATES_PREFERENCE,
        EdgeKind.FACT_UPDATES_BELIEF,
        EdgeKind.EVIDENCE_CONTRADICTS_PREF,
        EdgeKind.DECISION_REQUIRES_EVIDENCE,
    }

    def _validate_edge(self, edge: Edge) -> None:
        """
        Enforce provenance invariants before an edge is persisted.

        Rules:
            - BLOCK_UNTIL_RESOLVED edges MUST have source_ids.
            - Governed edge kinds (see GOVERNED_KINDS) MUST have source_ids
              at any gate level — these edges change beliefs or policies.
        """
        if edge.gate == GateLevel.BLOCK_UNTIL_RESOLVED and not edge.source_ids:
            raise ValueError(
                f"BLOCK_UNTIL_RESOLVED edges require provenance (source_ids). "
                f"Edge: {edge.src_id} -> {edge.dst_id} [{edge.kind.value}]"
            )
        if edge.kind in self.GOVERNED_KINDS and not edge.source_ids:
            raise ValueError(
                f"Governed edge kind '{edge.kind.value}' requires provenance (source_ids). "
                f"Edge: {edge.src_id} -> {edge.dst_id}"
            )

    def add_edge(self, edge: Edge) -> None:
        """
        Add or update an intra-zone edge.

        If kind is CONTRADICTS and gate is BLOCK_UNTIL_RESOLVED,
        a ContradictionRecord is also created and both nodes are blocked.

        Raises ValueError if provenance invariants are violated (see _validate_edge).
        """
        self._validate_edge(edge)

        if edge.confidence == 0.0:
            edge.confidence = 1.0

        eid = self._edge_id(edge.graph, edge.src_id, edge.dst_id, edge.kind)
        conn = self._conn_get()

        conn.execute("""
            INSERT INTO graph_edges
                (id, graph, src_id, dst_id, kind, weight, gate, confidence,
                 source_ids_json, snapshot_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(graph, src_id, dst_id, kind) DO UPDATE SET
                weight=excluded.weight, gate=excluded.gate,
                confidence=excluded.confidence,
                source_ids_json=excluded.source_ids_json,
                snapshot_json=excluded.snapshot_json
        """, (
            eid, edge.graph.value, edge.src_id, edge.dst_id,
            edge.kind.value, edge.weight, edge.gate.value,
            edge.confidence, json.dumps(edge.source_ids),
            edge.snapshot, edge.created_at
        ))
        conn.commit()

        # Update forward adjacency
        neighbors = self._adj[edge.graph][edge.src_id]
        neighbors = [(d, k, w, g) for d, k, w, g in neighbors
                     if not (d == edge.dst_id and k == edge.kind)]
        neighbors.append((edge.dst_id, edge.kind, edge.weight, edge.gate))
        self._adj[edge.graph][edge.src_id] = neighbors

        # Update reverse adjacency (in-edges index) — keep in sync with _adj
        in_neighbors = self._adj_in[edge.graph][edge.dst_id]
        in_neighbors = [(s, k, w, g) for s, k, w, g in in_neighbors
                        if not (s == edge.src_id and k == edge.kind)]
        in_neighbors.append((edge.src_id, edge.kind, edge.weight, edge.gate))
        self._adj_in[edge.graph][edge.dst_id] = in_neighbors

        self._blooms[edge.graph].add(edge.src_id)
        self._blooms[edge.graph].add(edge.dst_id)

        if self.audit:
            self.audit.log("edge_upsert", subject_id=edge.src_id, details={
                "graph": edge.graph.value, "dst": edge.dst_id,
                "kind": edge.kind.value, "gate": edge.gate.value
            })

    def add_bridge(self, bridge: BridgeEdge) -> None:
        """Add or update a cross-zone bridge edge."""
        bid = self._edge_id(
            bridge.from_graph, bridge.from_id,
            GraphZone(bridge.to_graph), bridge.to_id  # type: ignore
        )
        conn = self._conn_get()

        conn.execute("""
            INSERT INTO bridge_edges
                (id, from_graph, from_id, to_graph, to_id, kind, weight,
                 gate, confidence, source_ids_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(from_graph, from_id, to_graph, to_id, kind) DO UPDATE SET
                weight=excluded.weight, gate=excluded.gate,
                confidence=excluded.confidence,
                source_ids_json=excluded.source_ids_json
        """, (
            bid, bridge.from_graph.value, bridge.from_id,
            bridge.to_graph.value, bridge.to_id,
            bridge.kind.value, bridge.weight, bridge.gate.value,
            bridge.confidence, json.dumps(bridge.source_ids),
            bridge.created_at
        ))
        conn.commit()

        key = (bridge.from_graph, bridge.from_id)
        self._bridges[key] = [
            t for t in self._bridges[key]
            if not (t[0] == bridge.to_graph and t[1] == bridge.to_id)
        ]
        self._bridges[key].append((
            bridge.to_graph, bridge.to_id, bridge.weight, bridge.gate
        ))

        if self.audit:
            self.audit.log("bridge_upsert", subject_id=bridge.from_id, details={
                "from": bridge.from_graph.value, "to": bridge.to_graph.value,
                "kind": bridge.kind.value
            })

    # ── Contradiction detection ──────────────────────────────

    def detect_and_gate_contradiction(
        self,
        zone: GraphZone,
        node_a_id: str,
        node_a_content: str,
        node_b_id: str,
        node_b_content: str,
        detected_by: str = "system",
    ) -> ContradictionRecord:
        """
        Called when a contradiction between two nodes is detected.

        Creates a CONTRADICTS edge with BLOCK_UNTIL_RESOLVED gate.
        Snapshots both nodes at this moment for the audit record.
        Blocks both nodes from retrieval until a human resolves.

        This is a directive enforcement point — no code path can
        bypass the block once this is called.
        """
        now = datetime.now(timezone.utc).isoformat()

        snapshot = json.dumps({
            "node_a": {"id": node_a_id, "content": node_a_content},
            "node_b": {"id": node_b_id, "content": node_b_content},
            "detected_at": now,
            "detected_by": detected_by,
        })

        edge = Edge(
            graph=zone,
            src_id=node_a_id,
            dst_id=node_b_id,
            kind=EdgeKind.CONTRADICTS,
            weight=1.0,
            gate=GateLevel.BLOCK_UNTIL_RESOLVED,
            confidence=1.0,
            source_ids=[node_a_id, node_b_id],  # the two nodes are their own provenance
            snapshot=snapshot,
            created_at=now,
        )
        self.add_edge(edge)

        eid = self._edge_id(zone, node_a_id, node_b_id, EdgeKind.CONTRADICTS)

        rec = ContradictionRecord(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            zone=zone,
            snapshot_a=node_a_content,
            snapshot_b=node_b_content,
            detected_at=now,
            edge_id=eid,
        )

        cid = hashlib.sha256(f"{node_a_id}:{node_b_id}:{now}".encode()).hexdigest()[:16]
        conn = self._conn_get()
        conn.execute("""
            INSERT OR REPLACE INTO contradiction_records
                (id, node_a_id, node_b_id, zone, snapshot_a, snapshot_b,
                 detected_at, edge_id)
            VALUES (?,?,?,?,?,?,?,?)
        """, (cid, node_a_id, node_b_id, zone.value,
              node_a_content, node_b_content, now, eid))
        conn.commit()

        self._blocked_nodes[node_a_id] = rec
        self._blocked_nodes[node_b_id] = rec

        if self.audit:
            self.audit.log("contradiction_detected", subject_id=node_a_id, details={
                "node_b": node_b_id, "zone": zone.value,
                "edge_id": eid, "action": "blocked_until_resolved"
            })

        logger.warning(
            f"Contradiction detected in {zone.value}: "
            f"{node_a_id} ↔ {node_b_id} — both nodes BLOCKED until resolved"
        )
        return rec

    def resolve_contradiction(
        self,
        node_a_id: str,
        node_b_id: str,
        winner_id: str,
        resolution_type: str,   # "corrected" | "disputed" | "evidential"
        resolved_by: str,
        signature: str,
    ) -> bool:
        """
        Human resolution of a contradiction. Requires:
        - winner_id: which node is authoritative
        - resolution_type: what kind of contradiction this was
        - resolved_by: actor_id of the human who decided
        - signature: cryptographic signature of the resolution record

        The losing node is archived (important flag preserved if evidential).
        The CONTRADICTS edge gate is downgraded to NONE.
        Both nodes are unblocked.

        Returns True if resolution succeeded.
        """
        if node_a_id not in self._blocked_nodes:
            logger.warning(f"resolve_contradiction: {node_a_id} is not blocked")
            return False

        now = datetime.now(timezone.utc).isoformat()
        loser_id = node_b_id if winner_id == node_a_id else node_a_id
        zone = self._blocked_nodes[node_a_id].zone

        conn = self._conn_get()
        conn.execute("""
            UPDATE contradiction_records
            SET resolved=1, resolved_by=?, resolved_at=?,
                resolution_type=?, winner_id=?, signature=?
            WHERE node_a_id=? AND node_b_id=? AND resolved=0
        """, (resolved_by, now, resolution_type, winner_id,
              signature, node_a_id, node_b_id))

        # Downgrade the blocking edge
        conn.execute("""
            UPDATE graph_edges SET gate='none', resolved=1,
                resolved_by=?, resolved_at=?, resolution_type=?
            WHERE graph=? AND src_id=? AND dst_id=? AND kind='contradicts'
        """, (resolved_by, now, resolution_type,
              zone.value, node_a_id, node_b_id))
        conn.commit()

        # Shadow-archive the losing node's edges before unblocking.
        # The contradiction CONTRADICTS edge has already been downgraded above;
        # archiving captures the full edge context at resolution time.
        try:
            self.shadow.archive_edges_for_node(
                memory_id=loser_id,
                zone=zone,
                archived_by=resolved_by,
                reason=ArchiveReason.CONTRADICTION_LOSER,
            )
        except Exception:
            logger.warning(
                f"Shadow archival of contradiction edges for {loser_id} failed (non-fatal)"
            )

        # Unblock both nodes
        self._blocked_nodes.pop(node_a_id, None)
        self._blocked_nodes.pop(node_b_id, None)

        # Update in-memory adjacency to reflect gate downgrade
        self._rebuild_cache()

        if self.audit:
            self.audit.log("contradiction_resolved", subject_id=node_a_id, details={
                "node_b": node_b_id, "winner": winner_id,
                "loser": loser_id, "resolution_type": resolution_type,
                "resolved_by": resolved_by, "signature": signature
            })

        logger.info(
            f"Contradiction resolved: winner={winner_id}, "
            f"type={resolution_type}, by={resolved_by}"
        )
        return True

    def get_unresolved_contradictions(self) -> List[ContradictionRecord]:
        """Return all contradictions awaiting human resolution."""
        conn = self._conn_get()
        rows = conn.execute(
            "SELECT * FROM contradiction_records WHERE resolved=0"
        ).fetchall()
        return [
            ContradictionRecord(
                node_a_id=r["node_a_id"], node_b_id=r["node_b_id"],
                zone=GraphZone(r["zone"]), snapshot_a=r["snapshot_a"],
                snapshot_b=r["snapshot_b"], detected_at=r["detected_at"],
                edge_id=r["edge_id"],
            )
            for r in rows
        ]

    def verify_resolution(
        self,
        node_a_id: str,
        node_b_id: str,
        signer,
    ) -> Dict:
        """Verify the HMAC signature on a stored contradiction resolution.

        Reconstructs the canonical payload from the database row and checks it
        against the stored signature. A mismatch means the contradiction_records
        row was edited after the resolution was committed.

        Returns a dict with keys: verified (bool), reason (str), row data.
        """
        from veritas_memoria.core.audit.crypto_signing import SignatureBundle
        conn = self._conn_get()
        row = conn.execute(
            """SELECT node_a_id, node_b_id, winner_id, resolution_type,
                      resolved_by, resolved_at, signature
               FROM contradiction_records
               WHERE node_a_id=? AND node_b_id=? AND resolved=1""",
            (node_a_id, node_b_id),
        ).fetchone()

        if row is None:
            return {"verified": False, "reason": "no resolved record found for this pair"}

        bundle = SignatureBundle(
            key_id=signer.key_id,
            content_hash="",
            signature=row["signature"] or "",
        )
        ok, reason = signer.verify_resolution(
            node_a_id=row["node_a_id"],
            node_b_id=row["node_b_id"],
            winner_id=row["winner_id"],
            resolution_type=row["resolution_type"],
            resolved_by=row["resolved_by"],
            timestamp=row["resolved_at"],
            bundle=bundle,
        )
        return {
            "verified": ok,
            "reason": reason,
            "node_a_id": row["node_a_id"],
            "node_b_id": row["node_b_id"],
            "winner_id": row["winner_id"],
            "resolution_type": row["resolution_type"],
            "resolved_by": row["resolved_by"],
            "resolved_at": row["resolved_at"],
        }

    def is_blocked(self, node_id: str) -> bool:
        """Check if a node is blocked due to an unresolved contradiction."""
        return node_id in self._blocked_nodes

    # ── Traversal ────────────────────────────────────────────

    def _gate_allows(self, gate: GateLevel, policy: BridgePolicy) -> bool:
        """Check if a gate level is passable under the given policy."""
        if gate == GateLevel.NONE:
            return True
        if gate == GateLevel.BLOCK_UNTIL_RESOLVED:
            return False  # NEVER passable regardless of policy
        if gate == GateLevel.SUGGEST:
            return policy.allow_suggest
        if gate == GateLevel.REVIEW_REQUIRED:
            return policy.allow_review_required
        return False

    def traverse(
        self,
        zone: GraphZone,
        seeds: List[str],
        depth: int = 2,
        kind_filter: Optional[EdgeKind] = None,
        max_nodes: int = 200,
        policy: Optional[BridgePolicy] = None,
    ) -> List[str]:
        """
        BFS traversal within a single zone.

        Blocked nodes are never returned. Blocked edges are never crossed.
        """
        if policy is None:
            policy = BridgePolicy()

        visited: Set[str] = set(seeds)
        result: List[str] = []
        queue: deque = deque((s, 0) for s in seeds)

        while queue and len(result) < max_nodes:
            cur, d = queue.popleft()
            if d >= depth:
                continue

            for dst, kind, weight, gate in self._adj[zone].get(cur, []):
                if dst in visited:
                    continue
                if kind_filter and kind != kind_filter:
                    continue
                if not self._gate_allows(gate, policy):
                    continue
                if self.is_blocked(dst):
                    continue

                visited.add(dst)
                result.append(dst)
                queue.append((dst, d + 1))

        # Remove seeds from result
        return [r for r in result if r not in set(seeds)]

    def traverse_with_bridges(
        self,
        seeds: List[Tuple[GraphZone, str]],
        depth: int = 2,
        max_nodes: int = 200,
        follow_bridges: bool = True,
        policy: Optional[BridgePolicy] = None,
    ) -> List[Tuple[GraphZone, str]]:
        """
        Multi-zone BFS traversal with optional bridge crossing.

        Returns (zone, node_id) pairs in BFS order.
        """
        if policy is None:
            policy = BridgePolicy()

        visited: Set[Tuple[GraphZone, str]] = set(seeds)
        result: List[Tuple[GraphZone, str]] = []
        queue: deque = deque((z, i, 0) for z, i in seeds)

        while queue and len(result) < max_nodes:
            zone, cur, d = queue.popleft()
            if d >= depth:
                continue

            # Within-zone neighbors
            for dst, kind, weight, gate in self._adj[zone].get(cur, []):
                key = (zone, dst)
                if key in visited or not self._gate_allows(gate, policy):
                    continue
                if self.is_blocked(dst):
                    continue
                visited.add(key)
                result.append(key)
                queue.append((zone, dst, d + 1))

            # Bridge hops
            if follow_bridges:
                for to_zone, to_id, weight, gate in self._bridges.get((zone, cur), []):
                    key = (to_zone, to_id)
                    if key in visited or not self._gate_allows(gate, policy):
                        continue
                    if self.is_blocked(to_id):
                        continue
                    visited.add(key)
                    result.append(key)
                    queue.append((to_zone, to_id, d + 1))

        # Remove seeds from result
        seed_set = set(seeds)
        return [r for r in result if r not in seed_set]

    # ── Deep Semantic Mode ───────────────────────────────────

    def build_temp_edges(
        self,
        seed_ids: List[str],
        embedding_index,          # ProductionSemanticIndex or compatible
        similarity_threshold: float = 0.72,
        max_per_node: int = 5,
        default_zone: GraphZone = GraphZone.RATIONALE,
    ) -> List[TempEdge]:
        """
        Generate temporary similarity edges from the embedding index.

        These edges NEVER get written to storage. They exist only for
        the duration of a deep semantic query, extending the graph's
        reach to semantically related nodes that don't yet have
        permanent edges.

        The threshold matters:
          Legal/high-stakes deployment: 0.82+ (precision over recall)
          Creative/research deployment: 0.65+ (broader associations)
        """
        temp_edges: List[TempEdge] = []

        if embedding_index is None:
            logger.debug("No embedding index — deep semantic expansion skipped")
            return temp_edges

        for seed_id in seed_ids:
            if self.is_blocked(seed_id):
                continue

            try:
                # Ask the embedding index for nearest neighbors
                similar = embedding_index.find_similar_by_id(
                    seed_id,
                    limit=max_per_node * 2,  # over-fetch, filter below
                )
            except Exception as e:
                logger.warning(f"Embedding lookup failed for {seed_id}: {e}")
                continue

            count = 0
            for other_id, sim in similar:
                if other_id == seed_id or sim < similarity_threshold:
                    continue
                if self.is_blocked(other_id):
                    continue

                temp_edges.append(TempEdge(
                    src_id=seed_id,
                    dst_id=other_id,
                    similarity=sim,
                    zone=default_zone,
                ))
                count += 1
                if count >= max_per_node:
                    break

        logger.debug(f"Deep semantic: {len(temp_edges)} temp edges from {len(seed_ids)} seeds")
        return temp_edges

    def traverse_deep_semantic(
        self,
        seeds: List[Tuple[GraphZone, str]],
        embedding_index,
        depth: int = 2,
        max_nodes: int = 300,
        similarity_threshold: float = 0.72,
        follow_bridges: bool = True,
        policy: Optional[BridgePolicy] = None,
    ) -> List[Tuple[GraphZone, str]]:
        """
        Deep semantic traversal: permanent graph + temporary similarity edges.

        The temporary edges extend reach to semantically related nodes
        that don't yet have permanent relationships. After this call
        returns, the temp edges are gone.

        For the Calibrator: temp edges that keep recurring across many
        queries are candidates for promotion to permanent bridge edges
        pending human review.
        """
        if policy is None:
            policy = BridgePolicy()

        # Standard traversal first
        base_results = self.traverse_with_bridges(
            seeds, depth, max_nodes, follow_bridges, policy
        )

        # Build temp edges from all seeds
        seed_ids = [sid for _, sid in seeds]
        temp_edges = self.build_temp_edges(
            seed_ids, embedding_index, similarity_threshold
        )

        if not temp_edges:
            return base_results

        # Track recurrence for Calibrator — running average similarity per pair.
        # Pairs that keep appearing across many queries become bridge candidates.
        _now = datetime.now(timezone.utc).isoformat()
        for te in temp_edges:
            _key = (te.src_id, te.dst_id)
            _existing = self._temp_edge_counts.get(_key)
            if _existing is None:
                self._temp_edge_counts[_key] = {
                    "recurrence_count": 1,
                    "avg_similarity": te.similarity,
                    "zone": te.zone,
                    "last_seen": _now,
                }
            else:
                _n = _existing["recurrence_count"]
                _existing["avg_similarity"] = (
                    (_existing["avg_similarity"] * _n + te.similarity) / (_n + 1)
                )
                _existing["recurrence_count"] = _n + 1
                _existing["last_seen"] = _now

        # Extend results with temp edge targets not already found
        existing = set(base_results) | set(seeds)
        for te in temp_edges:
            key = (te.zone, te.dst_id)
            if key not in existing and not self.is_blocked(te.dst_id):
                base_results.append(key)
                existing.add(key)
                if len(base_results) >= max_nodes:
                    break

        return base_results

    # ── Overlay / Illumination Mode ──────────────────────────

    def illuminate(
        self,
        zone: GraphZone,
        lexical_scores: Dict[str, float],
        hierarchy_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
        top_k: int = 20,
        nagging_weight: float = 2.0,
    ) -> List[IlluminationResult]:
        """
        Overlay mode — all retrieval methods illuminate the graph
        simultaneously. Surfaces nodes that multiple methods agree
        on, especially nodes with high convergence but low direct
        query history.

        This is the "nagging feeling" detector. A node that scores
        well across lexical (BM25/keyword), hierarchy, semantic, AND
        graph centrality but has never been directly queried is likely
        something important that's been missed.

        Parameters:
            zone             — which zone to illuminate
            lexical_scores   — {node_id: score} from BM25/keyword index
            hierarchy_scores — {node_id: score} from hierarchy index
            semantic_scores  — {node_id: score} from embedding index
            top_k            — how many results to return
            nagging_weight   — multiplier for nodes with low query hits
                               but high convergence. Higher = more
                               aggressive surfacing of missed nodes.

        Returns IlluminationResult list sorted by nagging_score desc.
        """
        # Graph centrality: degree centrality within the zone
        # (number of unique connections, normalized)
        all_zone_nodes: Set[str] = set()
        degree: Dict[str, int] = defaultdict(int)

        for src, neighbors in self._adj[zone].items():
            all_zone_nodes.add(src)
            for dst, kind, weight, gate in neighbors:
                if gate != GateLevel.BLOCK_UNTIL_RESOLVED:
                    all_zone_nodes.add(dst)
                    degree[src] += 1
                    degree[dst] += 1

        max_degree = max(degree.values(), default=1)

        # Collect direct query hit counts
        conn = self._conn_get()
        hit_rows = conn.execute(
            "SELECT node_id, hit_count FROM node_query_hits WHERE zone=?",
            (zone.value,)
        ).fetchall()
        query_hits = {r["node_id"]: r["hit_count"] for r in hit_rows}

        # All candidate nodes: union of everything all methods found
        all_candidates = (
            set(lexical_scores)
            | set(hierarchy_scores)
            | set(semantic_scores)
            | all_zone_nodes
        )

        results: List[IlluminationResult] = []

        for node_id in all_candidates:
            if self.is_blocked(node_id):
                continue

            b   = lexical_scores.get(node_id, 0.0)
            h   = hierarchy_scores.get(node_id, 0.0)
            s   = semantic_scores.get(node_id, 0.0)
            g   = degree.get(node_id, 0) / max_degree  # 0–1 normalized
            hits = query_hits.get(node_id, 0)

            # Convergence: how many methods independently surfaced this node
            methods_hit = sum([b > 0, h > 0, s > 0, g > 0.1])
            convergence = (b + h + s + g) / 4.0

            # Nagging score: high convergence, low query history
            # This is the "something I should have looked at but didn't" signal
            if hits == 0 and methods_hit >= 2:
                nagging = convergence * nagging_weight
            elif hits < 3 and methods_hit >= 3:
                nagging = convergence * (nagging_weight * 0.6)
            else:
                nagging = convergence

            results.append(IlluminationResult(
                node_id=node_id,
                zone=zone,
                convergence_score=convergence,
                bm25_score=b,
                hierarchy_score=h,
                graph_centrality=g,
                semantic_score=s,
                direct_query_hits=hits,
                nagging_score=nagging,
            ))

        results.sort(key=lambda r: r.nagging_score, reverse=True)
        return results[:top_k]

    def record_query_hit(self, node_id: str, zone: GraphZone) -> None:
        """
        Record that a node was the direct answer to a user query.
        Used by the illumination mode to distinguish "found often via
        traversal" from "directly queried."
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn_get()
        conn.execute("""
            INSERT INTO node_query_hits (node_id, zone, hit_count, last_hit)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(node_id, zone) DO UPDATE SET
                hit_count = hit_count + 1,
                last_hit  = excluded.last_hit
        """, (node_id, zone.value, now))
        conn.commit()

    # ── Calibrator support ───────────────────────────────────

    def get_recurring_temp_edge_candidates(
        self,
        min_recurrence: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return node pairs whose temporary similarity edges have recurred enough
        times to be worth promoting to permanent bridge edges.

        Recurrence is tracked in _temp_edge_counts by traverse_deep_semantic().
        Results are sorted by recurrence count descending so the strongest
        candidates appear first.

        The Calibrator presents these to a human for sign-off before any
        permanent edge is written — this method is read-only.

        Parameters
        ----------
        min_recurrence : int
            Minimum number of times the temp edge must have appeared across
            deep semantic queries before it's returned as a candidate.
            Default 3 is a reasonable signal threshold for process-lifetime
            tracking; adjust lower for high-traffic deployments or higher
            when false positives are a concern.
        """
        candidates = []
        for (src_id, dst_id), data in self._temp_edge_counts.items():
            if data["recurrence_count"] < min_recurrence:
                continue
            candidates.append({
                "from_id": src_id,
                "to_id": dst_id,
                "from_zone": data["zone"],
                "to_zone": data["zone"],
                "recurrence_count": data["recurrence_count"],
                "avg_similarity": round(data["avg_similarity"], 4),
                "last_seen": data["last_seen"],
                "suggested_kind": EdgeKind.SEMANTIC_SIMILARITY,
            })
        candidates.sort(key=lambda c: c["recurrence_count"], reverse=True)
        return candidates

    # ── Introspection ────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        conn = self._conn_get()
        edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        bridge_count = conn.execute("SELECT COUNT(*) FROM bridge_edges").fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM contradiction_records WHERE resolved=0"
        ).fetchone()[0]
        return {
            "edges": edge_count,
            "bridges": bridge_count,
            "unresolved_contradictions": blocked,
            "bloom_counts": {z.value: self._blooms[z].count for z in GraphZone},
            "cached_blocked_nodes": len(self._blocked_nodes),
        }

    def backfill_chunk_edges(self) -> int:
        """
        One-time idempotent backfill: wire memories that share a source_file
        into sequential SUPPORTS chains using chunk_index from their metadata.

        Gated by a row in graph_migrations so it fires exactly once per DB,
        regardless of how many edges exist from other sources (contradictions,
        manual add_edge calls, etc.). Safe across multi-user, multi-project
        deployments — each project has its own SQLite, so backfills are isolated.

        Returns the number of edges created (0 if already done).
        """
        # v2: fixed guard to check `memories` table instead of `node_query_hits`.
        # node_query_hits is only populated on recall, not on save, so it was
        # always 0 on first startup — causing the migration to mark itself done
        # before any edges were created. Bumped to v2 so existing DBs re-run.
        MIGRATION = "backfill_chunk_edges_v2"
        conn = self._conn_get()

        # Ensure the migrations table exists (older DBs pre-date this table)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        conn.commit()

        already = conn.execute(
            "SELECT 1 FROM graph_migrations WHERE name=?", (MIGRATION,)
        ).fetchone()
        if already:
            return 0   # migration already ran for this DB

        nodes = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if nodes == 0:
            # Nothing ingested yet — mark done so we don't re-check every startup
            conn.execute(
                "INSERT OR IGNORE INTO graph_migrations (name, applied_at) VALUES (?,?)",
                (MIGRATION, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return 0

        # Read every memory's metadata from the memories table
        import json as _json
        from collections import defaultdict as _dd

        by_file: Dict[str, list] = _dd(list)
        for row in conn.execute("SELECT memory_id, metadata FROM memories"):
            try:
                meta = _json.loads(row["memory_id"]) if False else _json.loads(row["metadata"])
            except Exception:
                continue
            sf = meta.get("source_file")
            ci = meta.get("chunk_index")
            zone = meta.get("zone", "rationale")
            if sf is None or ci is None:
                continue
            by_file[sf].append((int(ci), row["memory_id"], zone))

        now = datetime.now(timezone.utc).isoformat()
        added = 0
        for sf, chunks in by_file.items():
            chunks.sort(key=lambda x: x[0])
            for (_, mid_a, zone_a), (_, mid_b, zone_b) in zip(chunks, chunks[1:]):
                zone = zone_a or zone_b or "rationale"
                # Coerce to valid GraphZone; fall back to rationale
                try:
                    gz = GraphZone(self._ZONE_MIGRATIONS.get(zone, zone))
                except ValueError:
                    gz = GraphZone.RATIONALE
                edge = Edge(
                    graph=gz,
                    src_id=mid_a,
                    dst_id=mid_b,
                    kind=EdgeKind.SUPPORTS,
                    weight=0.8,
                    gate=GateLevel.NONE,
                    confidence=0.9,
                    source_ids=[],
                )
                eid = self._edge_id(gz, mid_a, mid_b, edge.kind)
                conn.execute("""
                    INSERT OR IGNORE INTO graph_edges
                        (id, graph, src_id, dst_id, kind, weight, gate,
                         confidence, source_ids_json, snapshot_json, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    eid, gz.value, mid_a, mid_b, EdgeKind.SUPPORTS.value,
                    0.8, GateLevel.NONE.value, 0.9, "[]", None, now,
                ))
                # Keep in-memory adjacency in sync
                self._adj[gz][mid_a].append((mid_b, EdgeKind.SUPPORTS, 0.8, GateLevel.NONE))
                self._adj_in[gz][mid_b].append((mid_a, EdgeKind.SUPPORTS, 0.8, GateLevel.NONE))
                self._blooms[gz].add(mid_a)
                self._blooms[gz].add(mid_b)
                added += 1

        # Mark migration complete before committing edges so a crash mid-write
        # doesn't leave the flag set with partial edges. On next startup the
        # INSERT OR IGNORE on graph_edges handles any gaps cleanly.
        conn.execute(
            "INSERT OR IGNORE INTO graph_migrations (name, applied_at) VALUES (?,?)",
            (MIGRATION, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.info(
            "GraphEngine.backfill_chunk_edges: created %d edges across %d sources",
            added, len(by_file),
        )
        return added

    # ── Zone-corpus regex patterns (compiled once at class load) ─────────
    import re as _re
    _SURGICAL_SIGNALS = _re.compile(
        r'\b(surgic|operation|incision|dissect|ligat|sutur|excis|amputat|resect|'
        r'procedure|practition|surgeon|patient|treatment|prognos|clinical|'
        r'case\s+of|cases\s+of|the\s+student\s+should|in\s+practice|'
        r'operative|post.?operat|pre.?operat)\b',
        _re.IGNORECASE,
    )
    _ANATOMICAL_SIGNALS = _re.compile(
        r'\b(is\s+attached|is\s+inserted|arises\s+from|is\s+bounded|'
        r'is\s+composed\s+of|consists?\s+of|articulates?\s+with|'
        r'is\s+supplied\s+by|receives?\s+its|is\s+divided\s+into|'
        r'lies\s+(in|on|between|beneath|anterior|posterior)|'
        r'origin|insertion|innervat|vascular|foramen|fossa|'
        r'anterior|posterior|medial|lateral|superior|inferior|'
        r'artery|vein|nerve|muscle|bone|ligament|tendon|fascia|cartilage|'
        r'vertebr|thorac|abdomin|pelv|crani|femor|tibial|radial|ulnar|'
        r'carotid|jugular|subclavian|aorta|vena\s+cava)\b',
        _re.IGNORECASE,
    )
    _MEDICAL_CORPUS = _re.compile(
        r'\b(anatomy|anatomical|surgical|dissection|artery|vein|nerve|muscle|bone|ligament)\b',
        _re.IGNORECASE,
    )

    @classmethod
    def _infer_medical_zone(cls, content: str, current_zone: str) -> str:
        """
        Re-evaluate zone for medical reference content.

        Only acts on memories currently in rationale / work_knowledge / governance.
        Returns the same zone string if no reclassification is warranted.

        Classification rules (Gray's Anatomy corpus):
          governance   — objective anatomical structure (non-negotiable facts)
          work_knowledge — surgical / clinical application content
          rationale    — non-medical or ambiguous chunks; stays as-is
        """
        if current_zone not in ("rationale", "work_knowledge", "governance"):
            return current_zone
        if not cls._MEDICAL_CORPUS.search(content[:3000]):
            return current_zone
        sample = content[:1200]
        sh = len(cls._SURGICAL_SIGNALS.findall(sample))
        ah = len(cls._ANATOMICAL_SIGNALS.findall(sample))
        if sh > ah and sh >= 2:
            return "work_knowledge"
        return "governance"

    def rezone_existing_memories(self) -> int:
        """
        One-time idempotent migration: re-evaluate zones for every memory
        in the DB using _infer_medical_zone.

        Updates both memories.metadata and node_query_hits.zone so the
        graph visualization reflects the corrected zone colours immediately
        after the next server restart.

        Gated by graph_migrations row 'rezone_memories_v1' — runs exactly
        once per DB. Safe for multi-user / multi-project setups (each
        project is its own SQLite file).

        Returns the number of memories re-zoned (0 if already done).
        """
        MIGRATION = "rezone_memories_v1"
        import json as _json
        conn = self._conn_get()

        # Migrations table may not exist on very old DBs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        conn.commit()

        if conn.execute(
            "SELECT 1 FROM graph_migrations WHERE name=?", (MIGRATION,)
        ).fetchone():
            return 0  # already ran

        updates = []
        try:
            for row in conn.execute("SELECT memory_id, content, metadata FROM memories"):
                try:
                    meta = _json.loads(row["metadata"])
                except Exception:
                    continue
                current = meta.get("zone", "rationale")
                new_zone = self._infer_medical_zone(row["content"] or "", current)
                if new_zone != current:
                    meta["zone"] = new_zone
                    updates.append((_json.dumps(meta), new_zone, row["memory_id"]))
        except Exception as _e:
            logger.warning("rezone_existing_memories: read stopped early — %s", _e)

        now = datetime.now(timezone.utc).isoformat()
        for new_meta, new_zone, mid in updates:
            conn.execute(
                "UPDATE memories SET metadata=? WHERE memory_id=?",
                (new_meta, mid),
            )
            conn.execute(
                "UPDATE node_query_hits SET zone=? WHERE node_id=?",
                (new_zone, mid),
            )
            # Also fix any graph_edges that reference this node
            conn.execute(
                "UPDATE graph_edges SET graph=? WHERE src_id=? OR dst_id=?",
                (new_zone, mid, mid),
            )
            # Keep in-memory adjacency bloom in sync (node itself is unchanged)
            try:
                gz = GraphZone(self._ZONE_MIGRATIONS.get(new_zone, new_zone))
                self._blooms[gz].add(mid)
            except (ValueError, KeyError):
                pass

        conn.execute(
            "INSERT OR IGNORE INTO graph_migrations (name, applied_at) VALUES (?,?)",
            (MIGRATION, now),
        )
        conn.commit()

        if updates:
            logger.info(
                "GraphEngine.rezone_existing_memories: re-zoned %d memories",
                len(updates),
            )
        return len(updates)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

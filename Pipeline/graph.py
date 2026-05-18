"""
Backward-compatibility shim.

All graph symbols now live in submodules:
    veritas_memoria.core.graph.types        — enums, dataclasses, GRAPH_SCHEMA
    veritas_memoria.core.graph.bloom_filter — BloomFilter
    veritas_memoria.core.graph.engine       — GraphEngine

This file re-exports everything so existing imports of the form
    from veritas_memoria.core.graph.graph import GraphEngine, GraphZone, ...
continue to work without modification.
"""

from graph_types import (  
    GraphZone,
    EdgeKind,
    GateLevel,
    Edge,
    BridgeEdge,
    TempEdge,
    BridgePolicy,
    ContradictionRecord,
    IlluminationResult,
    GRAPH_SCHEMA,
)
from veritas_memoria.core.graph.bloom_filter import BloomFilter  # noqa: F401
from engine import GraphEngine  # noqa: F401

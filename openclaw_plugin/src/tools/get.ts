import { Type } from "@sinclair/typebox";

export const get = {
  name: "memory_get",
  description: "Fetch memories by ID with optional graph context. graph_depth: 1-3 hops. traverse: BFS walk. graph_mode 'summary' returns flattened strings, 'full' returns nodes+edges.",
  method: "memory_get",
  parameters: Type.Object({
    memory_ids: Type.Union([Type.Integer(), Type.Array(Type.Integer())]),
    detail_level: Type.Optional(Type.String({ default: "verbose" })),
    synthesize: Type.Optional(Type.Boolean({ default: false })),
    include_graph: Type.Optional(Type.Boolean({ default: true })),
    graph_depth: Type.Optional(Type.Integer({ default: 1 })),
    traverse: Type.Optional(Type.Boolean({ default: false })),
    max_depth: Type.Optional(Type.Integer({ default: 3 })),
    direction: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    relation_types: Type.Optional(Type.Union([Type.Array(Type.String()), Type.Null()])),
    min_strength: Type.Optional(Type.Union([Type.Number(), Type.Null()])),
    graph_mode: Type.Optional(Type.String({ default: "summary" })),
  }),
};

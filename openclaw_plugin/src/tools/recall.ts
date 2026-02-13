import { Type } from "@sinclair/typebox";

export const recall = {
  name: "memory_recall",
  description: "Semantic search (keyword fallback). Centrality-weighted ranking. memory_type supports wildcards like 'code_*'. synthesize: True=LLM synthesis, False=raw objects.",
  method: "memory_recall",
  parameters: Type.Object({
    query: Type.String(),
    instance_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    project: Type.Optional(Type.Union([Type.String(), Type.Array(Type.String()), Type.Null()])),
    memory_type: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    subject: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    min_foundational: Type.Optional(Type.Union([Type.Boolean(), Type.Null()])),
    include_archived: Type.Optional(Type.Boolean({ default: false })),
    limit: Type.Optional(Type.Integer({ default: 20 })),
    detail_level: Type.Optional(Type.String({ default: "summary" })),
    synthesize: Type.Optional(Type.Boolean({ default: true })),
    include_graph: Type.Optional(Type.Boolean({ default: true })),
    graph_top_n: Type.Optional(Type.Integer({ default: 5 })),
    graph_depth: Type.Optional(Type.Integer({ default: 1 })),
    graph_mode: Type.Optional(Type.String({ default: "summary" })),
  }),
};

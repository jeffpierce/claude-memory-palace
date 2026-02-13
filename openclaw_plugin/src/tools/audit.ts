import { Type } from "@sinclair/typebox";

export const audit = {
  name: "memory_audit",
  description: "Audit palace health. Checks: duplicates, stale, orphan_edges, missing_embeddings, contradictions. Foundational never stale.",
  method: "memory_audit",
  parameters: Type.Object({
    checks: Type.Optional(Type.Union([Type.Array(Type.String()), Type.Null()])),
    thresholds: Type.Optional(Type.Union([Type.Record(Type.String(), Type.Any()), Type.Null()])),
    project: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    limit_per_category: Type.Optional(Type.Integer({ default: 20 })),
  }),
};

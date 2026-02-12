import { Type } from "@sinclair/typebox";

export const archive = {
  name: "memory_archive",
  description: "Archive memories. Foundational always protected. dry_run: True by default (preview). centrality_protection: Protect high-centrality (default True, threshold=5).",
  method: "memory_archive",
  parameters: Type.Object({
    memory_ids: Type.Optional(Type.Union([Type.Array(Type.Integer()), Type.Null()])),
    older_than_days: Type.Optional(Type.Union([Type.Integer(), Type.Null()])),
    max_access_count: Type.Optional(Type.Union([Type.Integer(), Type.Null()])),
    memory_type: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    project: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    centrality_protection: Type.Optional(Type.Boolean({ default: true })),
    min_centrality_threshold: Type.Optional(Type.Integer({ default: 5 })),
    dry_run: Type.Optional(Type.Boolean({ default: true })),
    reason: Type.Optional(Type.Union([Type.String(), Type.Null()])),
  }),
};

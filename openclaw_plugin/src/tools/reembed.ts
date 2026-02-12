import { Type } from "@sinclair/typebox";

export const reembed = {
  name: "memory_reembed",
  description: "Regenerate embeddings. missing_only: Backfill NULL embeddings. dry_run: True by default.",
  method: "memory_reembed",
  parameters: Type.Object({
    older_than_days: Type.Optional(Type.Union([Type.Integer(), Type.Null()])),
    memory_ids: Type.Optional(Type.Union([Type.Array(Type.Integer()), Type.Null()])),
    project: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    all_memories: Type.Optional(Type.Boolean({ default: false })),
    missing_only: Type.Optional(Type.Boolean({ default: false })),
    batch_size: Type.Optional(Type.Integer({ default: 50 })),
    dry_run: Type.Optional(Type.Boolean({ default: true })),
  }),
};

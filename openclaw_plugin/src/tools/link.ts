import { Type } from "@sinclair/typebox";

export const link = {
  name: "memory_link",
  description: "Create relationship edge. Standard types: supersedes, relates_to, derived_from, contradicts, exemplifies, refines (custom OK). archive_old: True + supersedes archives target.",
  method: "memory_link",
  parameters: Type.Object({
    source_id: Type.Integer(),
    target_id: Type.Integer(),
    relation_type: Type.String(),
    strength: Type.Optional(Type.Number({ default: 1.0 })),
    bidirectional: Type.Optional(Type.Boolean({ default: false })),
    metadata: Type.Optional(Type.Union([Type.Record(Type.String(), Type.Any()), Type.Null()])),
    created_by: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    archive_old: Type.Optional(Type.Boolean({ default: false })),
  }),
};

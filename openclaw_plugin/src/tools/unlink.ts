import { Type } from "@sinclair/typebox";

export const unlink = {
  name: "memory_unlink",
  description: "Remove edge(s) between memories. relation_type: Specific type or null=ALL edges between source and target.",
  method: "memory_unlink",
  parameters: Type.Object({
    source_id: Type.Integer(),
    target_id: Type.Integer(),
    relation_type: Type.Optional(Type.Union([Type.String(), Type.Null()])),
  }),
};

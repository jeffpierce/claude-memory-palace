import { Type } from "@sinclair/typebox";

export const recent = {
  name: "memory_recent",
  description: "Last X memories, newest first. verbose: False=title-card, True=full details. memory_type supports wildcards. limit max 200.",
  method: "memory_recent",
  parameters: Object.assign(Type.Object({
    limit: Type.Optional(Type.Integer({ default: 20 })),
    verbose: Type.Optional(Type.Boolean({ default: false })),
    project: Type.Optional(Type.Union([Type.String(), Type.Array(Type.String()), Type.Null()])),
    memory_type: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    instance_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    include_archived: Type.Optional(Type.Boolean({ default: false })),
  }), { required: [] }),
};

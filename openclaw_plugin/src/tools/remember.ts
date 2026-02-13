import { Type } from "@sinclair/typebox";

export const remember = {
  name: "memory_remember",
  description: "Store new memory in palace. AUTO-LINKING: >=0.75 similarity auto-creates edges. memory_type is open-ended (fact, preference, event, insight, architecture, gotcha, solution, design_decision). foundational memories are never archived.",
  method: "memory_remember",
  parameters: Type.Object({
    instance_id: Type.String(),
    memory_type: Type.String(),
    content: Type.String(),
    subject: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    keywords: Type.Optional(Type.Union([Type.Array(Type.String()), Type.Null()])),
    tags: Type.Optional(Type.Union([Type.Array(Type.String()), Type.Null()])),
    foundational: Type.Optional(Type.Boolean({ default: false })),
    project: Type.Optional(Type.Union([Type.String(), Type.Array(Type.String())])),
    source_type: Type.Optional(Type.String({ default: "explicit" })),
    source_context: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    source_session_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    supersedes_id: Type.Optional(Type.Union([Type.Integer(), Type.Null()])),
    auto_link: Type.Optional(Type.Union([Type.Boolean(), Type.Null()])),
  }),
};

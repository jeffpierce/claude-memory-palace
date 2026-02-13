import { Type } from "@sinclair/typebox";

export const reflect = {
  name: "memory_reflect",
  description: "Extract memories from transcript (JSONL or TOON format). dry_run: Report without writing.",
  method: "memory_reflect",
  timeout: 180_000,
  parameters: Type.Object({
    instance_id: Type.String(),
    transcript_path: Type.String(),
    session_id: Type.Optional(Type.Union([Type.String(), Type.Null()])),
    dry_run: Type.Optional(Type.Boolean({ default: false })),
  }),
};

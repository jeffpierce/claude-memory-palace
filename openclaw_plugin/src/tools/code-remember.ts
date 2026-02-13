import { Type } from "@sinclair/typebox";

export const codeRemember = {
  name: "code_remember_tool",
  description: "Index source file for NL search. Creates prose (embedded) + raw code. force: Re-index if already indexed.",
  method: "code_remember_tool",
  parameters: Type.Object({
    code_path: Type.String(),
    project: Type.String(),
    instance_id: Type.String(),
    force: Type.Optional(Type.Boolean({ default: false })),
  }),
};

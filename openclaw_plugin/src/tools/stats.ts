import { Type } from "@sinclair/typebox";

export const stats = {
  name: "memory_stats",
  description: "Overview statistics: total memories, counts by type/instance/project, foundational, most accessed, recent activity.",
  method: "memory_stats",
  parameters: Type.Object({}),
};

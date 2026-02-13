import { remember } from "./remember";
import { recall } from "./recall";
import { get } from "./get";
import { recent } from "./recent";
import { archive } from "./archive";
import { link } from "./link";
import { unlink } from "./unlink";
import { message } from "./message";
import { codeRemember } from "./code-remember";
import { audit } from "./audit";
import { reembed } from "./reembed";
import { stats } from "./stats";
import { reflect } from "./reflect";

/** Tool definition shape */
export interface ToolDefinition {
  name: string;
  description: string;
  method: string;
  parameters: Record<string, any>;
  timeout?: number;
}

/** All 13 palace tools */
export const tools: ToolDefinition[] = [
  remember,
  recall,
  get,
  recent,
  archive,
  link,
  unlink,
  message,
  codeRemember,
  audit,
  reembed,
  stats,
  reflect,
];

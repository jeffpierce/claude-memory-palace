/**
 * Memory Palace — OpenClaw native tool plugin.
 *
 * Registers palace operations as first-class agent tools,
 * calling into the Python service layer via a persistent bridge subprocess.
 * Eliminates MCP protocol overhead for OpenClaw users.
 */

import { PalaceBridge, BridgeConfig } from "./bridge";
import { tools } from "./tools";

/** Plugin config from openclaw.plugin.json */
interface PluginConfig {
  bridgeCommand?: string;
  pythonPath?: string;
  instanceId?: string;
}

/** OpenClaw plugin context (provided by the gateway) */
interface PluginContext {
  config: PluginConfig;
  logger: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
  };
  /** Register a tool with the gateway */
  registerTool: (tool: ToolDefinition) => void;
  /** Optional: notify the agent of an event */
  notify?: (event: { type: string; data: any }) => void;
}

/** Tool definition structure for OpenClaw */
interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
  execute: (toolCallId: string, params: Record<string, any>, signal?: AbortSignal) => Promise<any>;
}

let bridge: PalaceBridge | null = null;

/**
 * Plugin initialization — called by OpenClaw gateway on load.
 *
 * MUST be synchronous — OpenClaw ignores async registration.
 * Bridge connects lazily on first tool call via auto-reconnect.
 */
export function register(context: any): void {
  const logger = context.logger;
  // Plugin-specific config is in pluginConfig, NOT config (which is the full OpenClaw config)
  const pluginCfg = context.pluginConfig ?? context.config ?? {};

  logger.info(`Plugin config keys: ${Object.keys(pluginCfg).join(", ")}`);

  // Build bridge config
  const bridgeConfig: BridgeConfig = {};
  if (pluginCfg.bridgeCommand) bridgeConfig.bridgeCommand = pluginCfg.bridgeCommand;
  if (pluginCfg.pythonPath) bridgeConfig.pythonPath = pluginCfg.pythonPath;
  if (pluginCfg.instanceId) bridgeConfig.instanceId = pluginCfg.instanceId;

  // Create bridge — do NOT start yet, lazy connect on first tool call
  bridge = new PalaceBridge(bridgeConfig);

  bridge.on("exit", (code: number | null, signal: string | null) => {
    logger.warn(`Palace bridge exited (code=${code}, signal=${signal}) — will auto-reconnect on next tool call`);
  });

  bridge.on("error", (err: Error) => {
    logger.error(`Palace bridge error: ${err.message}`);
  });

  // Listen for real-time message events
  bridge.on("new_message", (data: any) => {
    logger.info(`Palace notification: ${data.type} from ${data.from} — ${data.subject || "(no subject)"}`);
    if (context.notify) {
      context.notify({
        type: "palace.new_message",
        data,
      });
    }
  });

  // Register all tools synchronously — bridge connects on first call
  for (const tool of tools) {
    // Ensure required is always an array — OpenClaw calls .some() on it
    const parameters = { ...tool.parameters };
    if (!parameters.required) {
      parameters.required = [];
    }

    context.registerTool({
      name: tool.name,
      description: tool.description,
      parameters,
      execute: async (toolCallId: string, params: Record<string, any>) => {
        const result = await bridge!.call(tool.method, params, tool.timeout || 30_000);
        // OpenClaw expects Anthropic content block format: { content: [{type, text}] }
        const text = typeof result === "string" ? result : JSON.stringify(result, null, 2);
        return { content: [{ type: "text", text }] };
      },
    });
  }

  logger.info(`Registered ${tools.length} palace tools (bridge connects on first call)`);

  // Kick off bridge connection in background — non-blocking
  bridge.start().then(() => {
    logger.info(`Palace bridge connected (v${bridge!.version}, ${bridge!.tools} tools)`);
  }).catch((err: any) => {
    logger.warn(`Bridge pre-connect failed: ${err.message} — will retry on first tool call`);
  });
}

/**
 * Plugin shutdown — called by OpenClaw gateway on unload.
 */
export async function deactivate(): Promise<void> {
  if (bridge) {
    await bridge.stop();
    bridge = null;
  }
}

// Default export for OpenClaw plugin loader
export default {
  register,
  deactivate
};

export { PalaceBridge } from "./bridge";
export { tools } from "./tools";

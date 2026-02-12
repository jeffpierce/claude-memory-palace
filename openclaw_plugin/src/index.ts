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
  execute: (params: Record<string, any>) => Promise<any>;
}

let bridge: PalaceBridge | null = null;

/**
 * Plugin initialization — called by OpenClaw gateway on load.
 */
export async function activate(context: PluginContext): Promise<void> {
  const { config, logger } = context;

  // Build bridge config
  const bridgeConfig: BridgeConfig = {};
  if (config.bridgeCommand) bridgeConfig.bridgeCommand = config.bridgeCommand;
  if (config.pythonPath) bridgeConfig.pythonPath = config.pythonPath;
  if (config.instanceId) bridgeConfig.instanceId = config.instanceId;

  // Create and start bridge
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

  try {
    await bridge.start();
    logger.info(`Palace bridge connected (v${bridge.version}, ${bridge.tools} tools)`);
  } catch (err: any) {
    logger.error(`Failed to start palace bridge: ${err.message}`);
    logger.warn("Tools will attempt auto-reconnect on first call");
  }

  // Register all tools
  for (const tool of tools) {
    context.registerTool({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      execute: async (params: Record<string, any>) => {
        return bridge!.call(tool.method, params, tool.timeout || 30_000);
      },
    });
  }

  logger.info(`Registered ${tools.length} palace tools`);
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

export { PalaceBridge } from "./bridge";
export { tools } from "./tools";

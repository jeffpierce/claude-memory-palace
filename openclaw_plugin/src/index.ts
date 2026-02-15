/**
 * Memory Palace — OpenClaw native tool plugin.
 *
 * Registers palace operations as first-class agent tools,
 * calling into the Python service layer via a persistent bridge subprocess.
 * Eliminates MCP protocol overhead for OpenClaw users.
 *
 * NOTE: Requires live-patch to extensionAPI.js — see PATCH_NOTES.md.
 * Without the patch, runtime.system.enqueueSystemEvent and requestHeartbeatNow
 * use extensionAPI's module-scoped closures (isolated from the loader's copies),
 * meaning events go into a void queue and heartbeat has a null handler.
 * The patch rewires createPluginRuntime() to import from the loader chunk.
 */

import { PalaceBridge, BridgeConfig } from "./bridge";
import { tools } from "./tools";

/**
 * Self-trigger the gateway's palace-notify webhook hook.
 * This delivers via Discord, which wakes sleeping agent sessions
 * that enqueueSystemEvent + requestHeartbeatNow can't reach.
 */
async function triggerWebhookHook(
  port: number,
  hookToken: string,
  payload: Record<string, any>,
  logger: any,
): Promise<void> {
  try {
    const url = `http://127.0.0.1:${port}/hooks/palace`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${hookToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      logger.info(`[palace/webhook-relay] Triggered hook for ${payload.to_instance} (${res.status})`);
    } else {
      logger.warn(`[palace/webhook-relay] Hook returned ${res.status}: ${await res.text()}`);
    }
  } catch (err: any) {
    logger.warn(`[palace/webhook-relay] Failed to self-trigger hook: ${err.message}`);
  }
}

/** Plugin config from openclaw.plugin.json */
interface PluginConfig {
  bridgeCommand?: string;
  pythonPath?: string;
  instanceId?: string;
}

/** Tool definition structure for OpenClaw */
interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
  execute: (toolCallId: string, params: Record<string, any>, signal?: AbortSignal) => Promise<any>;
}

/** Maps instance_id → { sessionKey, agentId?, lastSeen } for targeted dispatch */
const sessionRegistry = new Map<string, { sessionKey: string; agentId?: string; lastSeen: number }>();

/** Maps channel → Set<instanceId> for pubsub dispatch — O(subscribers) not O(all) */
const channelSubscriptions = new Map<string, Set<string>>();

/** Default channels every instance auto-subscribes to on registration */
const DEFAULT_CHANNELS = ["system", "sandy"];

/** Palace tool names for hook filtering — O(1) lookup instead of array scan on every tool call */
const palaceToolNames = new Set(tools.map(t => t.name));

/**
 * Subscribe an instance to a channel in the local dispatch registry.
 * Idempotent — safe to call multiple times.
 */
function channelSubscribe(instanceId: string, channel: string): void {
  let subs = channelSubscriptions.get(channel);
  if (!subs) {
    subs = new Set();
    channelSubscriptions.set(channel, subs);
  }
  subs.add(instanceId);
}

/**
 * Unsubscribe an instance from a channel in the local dispatch registry.
 */
function channelUnsubscribe(instanceId: string, channel: string): void {
  const subs = channelSubscriptions.get(channel);
  if (subs) {
    subs.delete(instanceId);
    if (subs.size === 0) channelSubscriptions.delete(channel);
  }
}

/**
 * Auto-subscribe an instance to its own DM channel + all default channels.
 */
function autoSubscribeInstance(instanceId: string): void {
  channelSubscribe(instanceId, instanceId); // DMs via channel routing
  for (const ch of DEFAULT_CHANNELS) {
    channelSubscribe(instanceId, ch);
  }
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

  // Access system event functions from the plugin runtime.
  // With the extensionAPI.js patch applied, these are the LOADER's versions —
  // they share closures with the heartbeat handler and the real event queue.
  // Without the patch, these are dead (isolated closures, null handler).
  const runtime = context.runtime;
  let _enqueueSystemEvent: ((text: string, options: { sessionKey: string }) => void) | null = null;
  let _requestHeartbeatNow: ((opts?: { reason?: string; coalesceMs?: number }) => void) | null = null;

  if (runtime) {
    logger.info(`[palace] runtime keys: ${Object.keys(runtime).join(", ")}`);
    if (runtime.system) {
      logger.info(`[palace] runtime.system keys: ${Object.keys(runtime.system).join(", ")}`);
    }
  }

  if (runtime?.system?.enqueueSystemEvent) {
    _enqueueSystemEvent = runtime.system.enqueueSystemEvent;
    logger.info(`[palace] enqueueSystemEvent — ready`);
  } else {
    logger.warn(`[palace] context.runtime.system.enqueueSystemEvent not available — notification dispatch disabled`);
  }

  if (runtime?.system?.requestHeartbeatNow) {
    _requestHeartbeatNow = runtime.system.requestHeartbeatNow;
    logger.info(`[palace] requestHeartbeatNow — ready`);
  } else {
    logger.warn(`[palace] requestHeartbeatNow not on runtime.system`);
  }

  // Plugin-specific config is in pluginConfig, NOT config (which is the full OpenClaw config)
  const pluginCfg = context.pluginConfig ?? context.config ?? {};

  // Extract gateway config for webhook self-trigger (wakes sleeping agent sessions via Discord)
  const fullConfig = context.config ?? {};
  const gatewayPort = fullConfig?.gateway?.port ?? 18789;
  const hookToken = fullConfig?.hooks?.token ?? "";

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

  // Auto-discover agent sessions from tool usage
  // When any agent calls a palace tool, we capture their sessionKey + instance_id
  context.on("before_tool_call", (event: { toolName: string; params: Record<string, any> }, ctx: { agentId?: string; sessionKey?: string; sessionId?: string }) => {
    // Only care about palace tools
    if (!palaceToolNames.has(event.toolName)) return;

    // Extract instance_id from tool params — most tools use instance_id,
    // but message tool uses from_instance. Also fall back to bridge config.
    const instanceId = event.params?.instance_id
      || event.params?.from_instance
      || pluginCfg.instanceId;
    // Try sessionKey first, fall back to sessionId — OpenClaw may provide either
    const sessionKey = ctx?.sessionKey || ctx?.sessionId;

    // Log what we're seeing for debugging session discovery
    if (event.params?.instance_id || event.params?.from_instance) {
      logger.info(`[palace/session] Hook fired: tool=${event.toolName} instanceId=${instanceId} sessionKey=${sessionKey} agentId=${ctx.agentId} ctx_keys=${Object.keys(ctx || {}).join(",")}`);
    }

    if (instanceId && sessionKey) {
      const existing = sessionRegistry.get(instanceId);
      const isNew = !existing || existing.sessionKey !== sessionKey;
      sessionRegistry.set(instanceId, {
        sessionKey,
        agentId: ctx.agentId,
        lastSeen: Date.now(),
      });
      if (isNew) {
        logger.info(`[palace/session] Auto-registered: instance=${instanceId} session=${sessionKey} agent=${ctx.agentId || "unknown"}`);

        // Auto-subscribe to default channels for dispatch routing
        autoSubscribeInstance(instanceId);
        logger.info(`[palace/session] Auto-subscribed ${instanceId} to channels: [${instanceId}, ${DEFAULT_CHANNELS.join(", ")}]`);

        // Auto-subscribe bridge to this instance's Postgres LISTEN channel
        // so real-time dispatch works for all agents, not just the bootstrap instanceId.
        // LISTEN is idempotent in Postgres — safe to call even if already subscribed.
        if (bridge?.isReady) {
          const pgChannel = `memory_palace_msg_${instanceId}`;
          bridge.subscribe(pgChannel).then(() => {
            logger.info(`[palace/session] Auto-subscribed bridge to ${pgChannel}`);
          }).catch((err: any) => {
            logger.warn(`[palace/session] Failed to auto-subscribe to ${pgChannel}: ${err.message}`);
          });
        }
      }
    }

    // Track channel subscribe/unsubscribe actions from the palace_message tool
    if (event.toolName === "palace_message" && instanceId) {
      const action = event.params?.action;
      const channel = event.params?.channel;
      if (channel && action === "subscribe") {
        channelSubscribe(instanceId, channel);
        logger.info(`[palace/channel] ${instanceId} subscribed to channel "${channel}"`);
      } else if (channel && action === "unsubscribe") {
        channelUnsubscribe(instanceId, channel);
        logger.info(`[palace/channel] ${instanceId} unsubscribed from channel "${channel}"`);
      }
    }
  });

  // Listen for real-time message events and dispatch via enqueueSystemEvent
  bridge.on("new_message", async (data: any) => {
    logger.info(`[palace/notify] ${data.type} from=${data.from} to=${data.to} — ${data.subject || "(no subject)"}`);

    if (!_enqueueSystemEvent) {
      logger.warn(`[palace/notify] Runtime system API not available — cannot dispatch`);
      return;
    }

    const targetInstance = data.to;

    if (!targetInstance) {
      logger.warn(`[palace/notify] No target instance in notification, skipping dispatch`);
      return;
    }

    // Build the system event text
    const eventText = `[Palace Notification] New ${data.type || "message"} from ${data.from || "unknown"}: ${data.subject || "(no subject)"}. Use palace_message tool with action="get" to read it.`;

    // Helper: enqueue + heartbeat for a set of instance IDs
    const dispatchToInstances = (instanceIds: Iterable<string>, label: string): number => {
      let dispatched = 0;
      for (const instId of instanceIds) {
        const session = sessionRegistry.get(instId);
        if (!session) continue;
        try {
          _enqueueSystemEvent!(eventText, { sessionKey: session.sessionKey });
          dispatched++;
          logger.info(`[palace/dispatch] Enqueued for instance=${instId} session=${session.sessionKey} (${label})`);
        } catch (err: any) {
          logger.warn(`[palace/dispatch] Failed to enqueue for instance=${instId}: ${err.message}`);
        }
      }
      // Ring the doorbell — trigger agent heartbeat to drain the queue
      // TODO: HERE THERE BE DRAGONS — "exec-event" bypasses the empty-heartbeat-file
      // check in runHeartbeatOnce(). Without this lie, the heartbeat runner reads the
      // heartbeat prompt file, finds it empty, and skips. Upstream fix: OpenClaw should
      // expose a proper plugin wake API that doesn't route through the heartbeat system.
      if (dispatched > 0 && _requestHeartbeatNow) {
        try {
          _requestHeartbeatNow({ reason: "exec-event", coalesceMs: 250 });
          logger.info(`[palace/dispatch] Heartbeat fired for ${label} (${dispatched} sessions)`);
        } catch (err: any) {
          logger.warn(`[palace/dispatch] requestHeartbeatNow EXPLODED: ${err.message} — event queued but agent won't wake`);
        }
      }
      return dispatched;
    };

    if (targetInstance !== "all") {
      // Path 1: Targeted DM — dispatch to specific instance
      const dispatched = dispatchToInstances([targetInstance], `dm→${targetInstance}`);
      if (dispatched === 0) {
        logger.warn(`[palace/dispatch] Instance "${targetInstance}" not in session registry (${sessionRegistry.size} known). Message stays in palace for polling.`);
      }
    } else if (data.channel) {
      // Path 2: Channel message — dispatch only to subscribers of this channel
      const subscribers = channelSubscriptions.get(data.channel);
      if (subscribers && subscribers.size > 0) {
        const dispatched = dispatchToInstances(subscribers, `channel=${data.channel}`);
        logger.info(`[palace/dispatch] Channel "${data.channel}" complete: ${dispatched}/${subscribers.size} subscribers, ${sessionRegistry.size} total sessions`);
      } else {
        logger.warn(`[palace/dispatch] Channel "${data.channel}" has no subscribers (${channelSubscriptions.size} channels tracked). Message stays in palace for polling.`);
      }
    } else {
      // Path 3: True broadcast (to="all", no channel) — dispatch to ALL sessions
      const dispatched = dispatchToInstances(sessionRegistry.keys(), "broadcast");
      logger.info(`[palace/dispatch] Broadcast complete: ${dispatched}/${sessionRegistry.size} sessions`);
    }

    // Also fire the webhook hook path for targeted messages.
    // enqueueSystemEvent + requestHeartbeatNow only wakes sessions with active conversations.
    // The webhook hook delivers via Discord, which wakes sleeping agent sessions (like public).
    // Belt and suspenders: fast path (system events) + reliable path (Discord delivery).
    if (hookToken && targetInstance !== "all") {
      triggerWebhookHook(gatewayPort, hookToken, {
        to_instance: targetInstance,
        from_instance: data.from,
        message_id: data.id,
        message_type: data.type,
        subject: data.subject,
        priority: data.priority ?? 0,
      }, logger);
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

  // Debug introspection: show session registry state
  if (context.registerGatewayMethod) {
    context.registerGatewayMethod("memoryPalace.sessions", (opts: any) => {
      const sessions: Record<string, any> = {};
      for (const [instId, session] of sessionRegistry) {
        sessions[instId] = {
          sessionKey: session.sessionKey,
          agentId: session.agentId,
          lastSeen: new Date(session.lastSeen).toISOString(),
          ageMs: Date.now() - session.lastSeen,
        };
      }
      const channels: Record<string, string[]> = {};
      for (const [ch, subs] of channelSubscriptions) {
        channels[ch] = Array.from(subs);
      }
      opts.respond({ sessions, count: sessionRegistry.size, channels });
    });
    logger.info(`[palace] Registered gateway method: memoryPalace.sessions`);
  }

  // Kick off bridge connection in background — non-blocking
  bridge.start().then(async () => {
    logger.info(`Palace bridge connected (v${bridge!.version}, ${bridge!.tools} tools)`);

    // Pre-seed sessionRegistry from palace config's instance_routes.
    // This eliminates the chicken-and-egg where agents need to make a tool
    // call before pubsub dispatch works. Bridge restart = fully self-healing.
    try {
      const routes = await bridge!.call("_get_instance_routes", {}, 5_000);
      if (routes?.instance_routes) {
        let seeded = 0;
        for (const [instanceId, route] of Object.entries(routes.instance_routes) as [string, any][]) {
          const sessionKey = route.session;
          if (sessionKey && !sessionRegistry.has(instanceId)) {
            sessionRegistry.set(instanceId, {
              sessionKey,
              agentId: undefined,
              lastSeen: Date.now(),
            });
            // Auto-subscribe to default channels for dispatch routing
            autoSubscribeInstance(instanceId);
            seeded++;
          }
        }
        logger.info(`[palace/session] Pre-seeded ${seeded} sessions from instance_routes (${sessionRegistry.size} total, ${channelSubscriptions.size} channels)`);
      }
    } catch (err: any) {
      logger.warn(`[palace/session] Failed to pre-seed from instance_routes: ${err.message} — agents will self-register on first tool call`);
    }
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

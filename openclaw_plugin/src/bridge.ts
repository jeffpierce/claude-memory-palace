/**
 * PalaceBridge — manages the persistent Python bridge subprocess.
 *
 * Protocol: newline-delimited JSON over stdin/stdout.
 * - Requests: { id, method, params } → { id, result } | { id, error }
 * - Events: { event, data } (server-initiated, no id)
 *
 * The bridge subprocess holds a long-lived DB connection and optionally
 * a Postgres LISTEN thread for real-time pubsub events.
 */

import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import { createInterface, Interface as ReadlineInterface } from "readline";

/** Bridge configuration */
export interface BridgeConfig {
  /** Command to spawn bridge (default: "memory-palace-bridge") */
  bridgeCommand?: string;
  /** Optional Python interpreter path — spawns as: pythonPath -m memory_palace.bridge */
  pythonPath?: string;
  /** Instance ID for auto-subscribe */
  instanceId?: string;
  /** Ready timeout in ms (default: 10000) */
  readyTimeout?: number;
}

/** Pending request tracker */
interface PendingRequest {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class PalaceBridge extends EventEmitter {
  private config: BridgeConfig;
  private process: ChildProcess | null = null;
  private readline: ReadlineInterface | null = null;
  private pending: Map<string, PendingRequest> = new Map();
  private nextId: number = 1;
  private ready: boolean = false;
  private readyPromise: Promise<void> | null = null;
  private bridgeVersion: string = "";
  private toolCount: number = 0;

  constructor(config: BridgeConfig = {}) {
    super();
    this.config = {
      bridgeCommand: config.bridgeCommand || "memory-palace-bridge",
      readyTimeout: config.readyTimeout || 10_000,
      ...config,
    };
  }

  /**
   * Start the bridge subprocess and wait for the ready handshake.
   */
  async start(): Promise<void> {
    if (this.ready && this.process) return;

    // Build spawn command
    let command: string;
    let args: string[];
    if (this.config.pythonPath) {
      command = this.config.pythonPath;
      args = ["-m", "memory_palace.bridge"];
    } else {
      command = this.config.bridgeCommand!;
      args = [];
    }

    // Set instance ID env var if configured
    const env = { ...process.env };
    if (this.config.instanceId) {
      env.MEMORY_PALACE_INSTANCE_ID = this.config.instanceId;
    }

    this.process = spawn(command, args, {
      stdio: ["pipe", "pipe", "inherit"],
      env,
    });

    this.process.on("exit", (code, signal) => {
      this.ready = false;
      this.readyPromise = null;
      // Reject all pending requests
      for (const [id, req] of this.pending) {
        clearTimeout(req.timer);
        req.reject(new Error(`Bridge process exited (code=${code}, signal=${signal})`));
      }
      this.pending.clear();
      this.process = null;
      this.readline = null;
      this.emit("exit", code, signal);
    });

    this.process.on("error", (err) => {
      this.emit("error", err);
    });

    // Set up line-based reader on stdout
    this.readline = createInterface({ input: this.process.stdout! });
    this.readline.on("line", (line: string) => this.handleLine(line));

    // Wait for ready handshake
    this.readyPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Bridge did not send ready handshake within ${this.config.readyTimeout}ms`));
      }, this.config.readyTimeout);

      const onReady = () => {
        clearTimeout(timeout);
        resolve();
      };

      this.once("_ready", onReady);
    });

    await this.readyPromise;
  }

  /**
   * Send a request to the bridge and wait for the response.
   */
  async call(method: string, params: Record<string, any> = {}, timeoutMs: number = 30_000): Promise<any> {
    // Auto-reconnect if bridge is dead
    if (!this.ready || !this.process) {
      await this.start();
    }

    const id = `req-${this.nextId++}`;
    const request = JSON.stringify({ id, method, params }) + "\n";

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Bridge request ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      this.process!.stdin!.write(request);
    });
  }

  /**
   * Gracefully stop the bridge.
   */
  async stop(): Promise<void> {
    if (!this.process || !this.ready) return;

    try {
      await this.call("_shutdown", {}, 5_000);
    } catch {
      // Force kill if shutdown request fails
    }

    // Give it a moment to exit gracefully
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        if (this.process) {
          this.process.kill("SIGTERM");
        }
        resolve();
      }, 3_000);

      if (this.process) {
        this.process.once("exit", () => {
          clearTimeout(timeout);
          resolve();
        });
      } else {
        clearTimeout(timeout);
        resolve();
      }
    });

    this.ready = false;
    this.process = null;
    this.readline = null;
  }

  /**
   * Subscribe to additional Postgres LISTEN channels.
   */
  async subscribe(channel: string): Promise<any> {
    return this.call("_subscribe", { channel });
  }

  /**
   * Unsubscribe from a Postgres LISTEN channel.
   */
  async unsubscribe(channel: string): Promise<any> {
    return this.call("_unsubscribe", { channel });
  }

  /** Whether the bridge is currently connected and ready */
  get isReady(): boolean {
    return this.ready;
  }

  /** Bridge version from handshake */
  get version(): string {
    return this.bridgeVersion;
  }

  /** Number of tools reported by bridge */
  get tools(): number {
    return this.toolCount;
  }

  // ── Internal ──────────────────────────────────────────────

  private handleLine(line: string): void {
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      // Ignore non-JSON lines (shouldn't happen but be safe)
      return;
    }

    // Ready handshake
    if (msg.ready === true) {
      this.ready = true;
      this.bridgeVersion = msg.version || "";
      this.toolCount = msg.tools || 0;
      this.emit("_ready");
      return;
    }

    // Server-initiated event (has "event" field, no "id")
    if (msg.event && !msg.id) {
      this.emit(msg.event, msg.data);
      return;
    }

    // Response to a pending request
    if (msg.id) {
      const pending = this.pending.get(msg.id);
      if (pending) {
        this.pending.delete(msg.id);
        clearTimeout(pending.timer);
        if (msg.error) {
          pending.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
        } else {
          pending.resolve(msg.result);
        }
      }
      return;
    }
  }
}

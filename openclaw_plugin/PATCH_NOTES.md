# OpenClaw extensionAPI.js Live-Patch for Pubsub Wake

## Overview

This document describes a **critical live-patch** required for the Memory Palace OpenClaw plugin to receive real-time wake notifications when palace messages arrive. Without this patch, the plugin cannot wake the agent automatically.

**Status**: FRAGILE HACK - Required until upstream fix lands

## The Problem

### Module Isolation Bug

OpenClaw's build process creates **4 independent copies** of core system functions across different bundle chunks:
- `loader-BpQdnOY1.js` (the REAL implementation with working closures)
- `reply-*.js`
- `extensionAPI.js`
- `pi-embedded-*.js`

The `createPluginRuntime()` function in `extensionAPI.js` creates its OWN isolated copies of:
- `enqueueSystemEvent()` - queues system events for processing
- `requestHeartbeatNow()` - triggers agent wake

**These copies don't share closures with the heartbeat handler.** Events go into a void queue, heartbeat calls hit a null handler, and the agent never wakes.

### Why This Matters

When a palace message arrives via PostgreSQL NOTIFY, the plugin needs to:
1. Enqueue a system event describing the message
2. Request an immediate heartbeat to wake the agent
3. Have the agent process the event and respond

Without the patch, steps 1-2 succeed silently but accomplish nothing.

## Patch Procedure

### Prerequisites

Locate your OpenClaw installation's `dist/` directory:
- Global npm: `~/.npm-global/lib/node_modules/openclaw/dist/`
- Local: `./node_modules/openclaw/dist/`
- Windows global: `%APPDATA%\npm\node_modules\openclaw\dist\`

### Step 0: Backup

**ALWAYS backup before patching:**

```bash
cd /path/to/openclaw/dist
cp extensionAPI.js extensionAPI.js.bak
```

### Step 1: Add Loader Import

At **line 1** of `extensionAPI.js`, add:

```javascript
import { Zn as _loaderEnqueueSystemEvent, er as _loaderRequestHeartbeatNow } from "./loader-BpQdnOY1.js";
```

**Note**: The filename hash `BpQdnOY1` and minified names `Zn`, `er` are **build-specific** and WILL change with OpenClaw updates.

### Step 2: Patch createPluginRuntime()

Find the `system:` block inside `createPluginRuntime()` (around line 60473). It looks like:

```javascript
system: {
  enqueueSystemEvent,
  requestHeartbeatNow,
  // ... other methods
}
```

Replace with:

```javascript
system: {
  enqueueSystemEvent: _loaderEnqueueSystemEvent,
  requestHeartbeatNow: _loaderRequestHeartbeatNow,
  // ... other methods
}
```

### Step 3: Verify

Restart OpenClaw and test that pubsub notifications wake the agent. You should see system events logged when palace messages arrive.

## Fragility Warnings

### This Patch WILL Break

**On every OpenClaw update** via `npm update openclaw` or reinstall, this patch is **wiped**. You must re-apply manually.

### Build-Specific Identifiers

The following change with **every OpenClaw build**:

1. **Loader filename hash**: `loader-BpQdnOY1.js` → `loader-XyZ123Ab.js`
2. **Minified export names**: `Zn`, `er` → `Aa`, `Bb` (arbitrary)

### Re-Application After Updates

When OpenClaw updates, you must:

1. **Find the new loader filename**:
   ```bash
   ls dist/loader-*.js
   ```

2. **Find the new minified export names**:
   - Open the loader file
   - Scroll to the **bottom** (export map)
   - Search for functions matching these signatures:
     - `enqueueSystemEvent`: Takes `text` + `options` with `sessionKey`
     - `requestHeartbeatNow`: Takes `opts` with `reason` + `coalesceMs`

   Example export map:
   ```javascript
   export { Zn as enqueueSystemEvent, er as requestHeartbeatNow, ... }
   ```

3. **Update the import** in Step 1 with new filename and export names
4. **Re-apply** Step 2 (should be unchanged)

## The Dragon Hack

### Heartbeat File Empty Check Bypass

The plugin uses `reason: "exec-event"` in `requestHeartbeatNow()` calls. This is a **deliberate lie**.

**Why**: OpenClaw's `runHeartbeatOnce()` checks if the heartbeat prompt file is empty. If empty, it **skips the heartbeat entirely**. Our pubsub wake mechanism needs the heartbeat to run **even when the file is empty**, because the event context comes from the system event, not the prompt file.

**Locations**: Both plugin invocations in `openclaw_plugin/src/index.ts` are marked:
```typescript
// HERE THERE BE DRAGONS
system.requestHeartbeatNow({ reason: "exec-event", coalesceMs: 50 });
```

**Alternative considered**: Writing to the heartbeat file would be cleaner but requires filesystem access and coordination. The `exec-event` lie is simpler and works.

## Upstream Fix

The **proper solution** requires OpenClaw changes:

1. **Expose plugin wake API**: Provide a dedicated `plugin.wake()` method that doesn't route through the heartbeat system
2. **Fix module isolation**: Ensure plugins receive the loader's real implementations instead of isolated dead copies
3. **Consolidate exports**: Use a single source of truth for system functions across all chunks

An upstream PR should be submitted to OpenClaw with these changes. Until then, this patch is required for Memory Palace real-time notifications.

## Testing

After applying the patch:

1. Start OpenClaw with Memory Palace plugin enabled
2. Send a test message via `message` tool to the plugin's instance ID
3. Verify the agent wakes automatically (check logs for "waking agent for palace message")
4. Confirm the message appears in system events

If the agent doesn't wake, the patch likely failed or was reverted by an update.

## Maintenance Checklist

- [ ] Backup `extensionAPI.js` before patching
- [ ] Apply Step 1: Add loader import with correct filename/exports
- [ ] Apply Step 2: Patch `createPluginRuntime()` system block
- [ ] Test pubsub wake functionality
- [ ] Document current OpenClaw version and minified names for future reference
- [ ] Set reminder to re-apply after OpenClaw updates

## Version History

| Date | OpenClaw Version | Loader Hash | Export Names | Notes |
|------|------------------|-------------|--------------|-------|
| 2026-02-13 | (current) | BpQdnOY1 | Zn, er | Initial patch |

Update this table when re-applying the patch to track identifier changes across versions.

---

**Maintained by**: Sandy Blake (jeff+sandy@spectrumgamecreations.com)
**Status**: Active Hack - Required for Production
**Upstream Issue**: TBD (create OpenClaw GitHub issue)

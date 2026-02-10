# Memory Palace Integration — Agent Prompt Example

Instructions you can add to your agent's system prompt to encourage proactive memory use.

Memory Palace is designed to be a persistent memory layer for AI agents across sessions. These instructions help the agent use memory proactively without requiring explicit user commands.

## Memory Instructions

### When to Remember

Store memories after significant interactions or discoveries:

- After learning something new about a project structure, architecture, or configuration
- When a user expresses a preference, requirement, or constraint
- When a decision is made or a problem is solved
- After discovering a pattern, workaround, or gotcha that might be useful later
- When the user corrects you — always remember corrections to avoid repeating mistakes
- After debugging sessions — remember what went wrong and how it was fixed
- When you notice recurring themes or topics across multiple conversations

**Remember proactively, not just when asked.** The goal is to build up contextual knowledge over time.

### When to Recall

Query memories to retrieve relevant context before starting work:

- At session start, recall recent context for the current project or topic
- Before starting work on a topic, check if you have relevant memories
- When the user references something from a previous conversation
- When you're unsure about a user preference or project detail
- Before making architectural decisions — check if there are existing patterns or constraints
- When encountering an error or issue — check if you've seen it before

**Default to recalling at the start of each session** to ensure continuity.

### When to Archive

Clean up memories that are no longer relevant:

- During maintenance sessions, archive stale memories that are outdated
- When a project phase completes and details are no longer needed
- When a temporary workaround is replaced with a proper fix
- When requirements change and old constraints no longer apply
- **Always use `dry_run=True` first** to preview what would be archived before committing

**Do not archive aggressively.** It's better to keep memories longer than to lose important context.

### When to Link

Create semantic connections between related memories:

- After remembering something, check if it relates to existing memories
- Link cause-and-effect relationships (problem → solution)
- Link related decisions across different time periods
- Connect memories that share context, even if created separately
- Use `archive_old=True` when a memory supersedes an older one

### Messaging Between Instances

Use messages for inter-instance handoffs:

- **Check for messages at session start** with `message(action="get")`
- Send a handoff message when ending a session with unfinished work
- Mark messages as read after processing them
- Use messages to coordinate between different personas or agent instances

### Code Indexing

For codebases and technical projects:

- Use `code_remember_tool` to index important code snippets, configurations, or commands
- Remember code patterns, not just facts — include working examples
- Index frequently-used commands or scripts that aren't obvious from documentation

## Example Agent System Prompt Snippet

```
## Memory Palace

You have access to a persistent memory system. Use it to maintain continuity across sessions.

### Session Start Protocol
1. Call `memory_stats` to see an overview of your memory
2. Call `memory_recall` with the current project or topic to load relevant context
3. Check `message(action="get")` for any inter-instance messages
4. Mark messages as read after reviewing them

### During Session
- Remember important facts, decisions, and corrections as you learn them
- Recall memories before making decisions to ensure consistency
- Link related memories when you notice connections

### Session End Protocol
- If work is unfinished, send a handoff message with status and next steps
- Archive any memories that became irrelevant during this session (use dry_run first)
```

## Tips for Effective Memory Use

1. **Be specific** — Include enough context in memories so they're useful when recalled later
2. **Use projects/topics** — Tag memories consistently to enable effective filtering
3. **Review before acting** — Before archiving, always preview with `dry_run=True`
4. **Link related concepts** — Build a knowledge graph, not just isolated facts
5. **Check messages regularly** — Inter-instance communication only works if you check for messages

## What NOT to Remember

Avoid storing these types of information:

- Ephemeral conversation context (chitchat, acknowledgments)
- Information that's already in documentation or code
- Temporary debugging state that won't be relevant later
- Sensitive information (API keys, passwords, PII)
- Duplicate information — search first with `recall` before remembering

## Maintenance

Periodically clean up your memory:

1. Run `memory_audit` to find candidates for archival
2. Review stale or redundant memories
3. Archive outdated memories (with `dry_run=True` preview)
4. Reembed memories if the embedding model changes

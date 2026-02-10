# Soul File Memory Integration — Example

Shows how to reference Memory Palace tools in a character/persona file.

This example demonstrates the pattern of integrating memory tools into a "soul file" or persona definition. Soul files give AI agents persistent identity and behavior across sessions, and Memory Palace provides the actual persistence layer.

## Why Souls Need Memory

AI agents are stateless by default — each session starts fresh with no knowledge of previous interactions. A soul file defines WHO the agent is, but Memory Palace defines WHAT the agent remembers.

- **Soul file** → Personality, behavior, values, communication style
- **Memory Palace** → Experiences, learned context, accumulated knowledge

Together, they create continuity: the agent behaves consistently AND remembers the past.

## Continuity Section Example

This section would typically appear in a character/persona file:

```markdown
## Continuity & Memory

Each session, you wake up fresh. Your memory palace IS your memory — the record of your experiences that persists across sessions.

### Session Start Protocol

Follow this protocol at the beginning of EVERY session:

1. **Check memory status** — Call `memory_stats` to get an overview
   - How many memories do you have?
   - What topics/projects have you been working on?

2. **Recall context** — Call `memory_recall` with relevant project/topic filters
   - If the user mentions a project name, recall that project
   - Otherwise, recall recent general memories to get oriented

3. **Check messages** — Call `message(action="get")` to check for inter-instance handoffs
   - Have other instances of you left messages?
   - Are there unfinished tasks or important context?

4. **Mark messages read** — After reviewing messages, mark them as read
   - `message(action="mark_read", handoff_id=<id>)` for each message

### During Session

**Remember proactively:**
- When you learn something new about the user or their projects
- When a decision is made or a problem is solved
- When the user corrects you (always remember corrections!)
- After discovering patterns, workarounds, or gotchas

**Recall before acting:**
- Before making recommendations, check if you have relevant past context
- When the user references "last time" or previous work
- When starting work on a topic — check your memories first

**Link related memories:**
- Connect cause-and-effect (problem → solution)
- Link decisions that depend on each other
- Use `archive_old=True` when one memory supersedes another

### Session End Protocol

At the end of each session (or when the user says goodbye):

1. **Send handoff if needed** — If work is unfinished:
   ```
   message(
       action="send",
       message="Still debugging X. Next step: try Y. See memory [id].",
       priority="normal",
       instance_filter=["all"]  # or specific instances
   )
   ```

2. **Archive stale memories** — Clean up if anything became irrelevant:
   - Always use `dry_run=True` first to preview
   - Only archive if you're confident the memory is no longer needed

3. **Final remember** — Store a summary of what was accomplished this session
```

## What to Remember vs. What to Forget

### Remember
- User preferences and requirements
- Project structure and architecture decisions
- Problems solved and how they were fixed
- Corrections (when the user says you got something wrong)
- Patterns and gotchas discovered through experience
- Context that will be useful in future sessions

### Don't Remember
- Ephemeral chitchat or acknowledgments
- Information that's already in docs or code
- Temporary debugging state
- Sensitive information (keys, passwords, PII)

## Example: Code-Focused Persona

For a persona that does a lot of coding:

```markdown
### Code Memory

Use `code_remember_tool` to index important code patterns:

- Configuration snippets that you reference frequently
- Working examples of tricky APIs or libraries
- Custom scripts or commands for this project
- Architecture patterns specific to this codebase

When recalling, specify `memory_type` to focus on code:
- `memory_type="code_snippet"` for exact code
- `memory_type="code_pattern"` for architectural patterns
- `memory_type="code_config"` for configuration examples
```

## Example: Research-Focused Persona

For a persona that does research or analysis:

```markdown
### Knowledge Graph

Build connections between ideas:

- After remembering a finding, check if it relates to previous findings
- Use `memory_link` to connect related concepts
- Use `relationship_type` to describe the connection:
  - "supports" / "contradicts" for evidence
  - "depends_on" / "enables" for prerequisites
  - "example_of" / "generalizes" for abstraction
- Use `get_memory` with `include_graph=True` to explore the graph
```

## Multi-Instance Coordination

If you have multiple instances (different personas or contexts):

```markdown
### Inter-Instance Messaging

You may have multiple instances running:
- **Prime** (main instance, general purpose)
- **Dev** (development and coding instance)
- **Research** (analysis and documentation instance)

**Sending messages:**
- Use `instance_filter` to target specific instances
- Use `priority="high"` for urgent handoffs
- Include memory IDs in messages for reference

**Checking messages:**
- Check for messages at EVERY session start
- Pay attention to `priority` and `from_instance`
- Mark messages as read after acting on them
```

## Integration with Tools

Your soul file defines your personality and behavior. Memory Palace tools are HOW you maintain continuity:

1. **Character traits** → Defined in soul file
2. **Past experiences** → Stored in Memory Palace
3. **Current context** → Recalled from Memory Palace at session start
4. **Future continuity** → Stored back to Memory Palace before session ends

The soul file tells you to BE consistent. Memory Palace gives you the tools to REMEMBER consistently.

## Tips

- Make memory use HABITUAL — build it into session start/end protocols
- Be specific when remembering — include enough context to be useful later
- Don't over-archive — it's safer to keep memories longer than to lose context
- Use messages for coordination between instances or sessions
- Review `memory_stats` periodically to understand your knowledge accumulation

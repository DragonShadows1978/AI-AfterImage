# AI-AfterImage

**Episodic memory for AI coding agents.**

Like the visual phenomenon where an image persists after you look away - AfterImage gives Claude Code persistent memory of code it has written across sessions.

## The Problem

Claude Code starts every session with amnesia. Even though transcripts exist with every Write/Edit ever made, Claude can't remember:
- What code it wrote yesterday
- How it solved a similar problem last week
- Patterns it has used before in this codebase

Users re-explain context. Claude rewrites similar solutions. Institutional knowledge is lost.

## The Solution

A Claude Code hook that:

1. **Pre-Write**: Searches KB for related past code before writing
2. **Injects**: "You wrote this before..." with relevant examples
3. **Post-Write**: Extracts and stores the diff for future recall

```
Write/Edit hook fires
        │
        ▼
   Is this code?
   (not .md/.json/etc)
        │
        ▼
   Search KB for similar
        │
   ┌────┴────┐
Found      Not Found
   │           │
   ▼           ▼
Inject     Just write
context
   │           │
   └─────┬─────┘
         ▼
   Claude writes
         │
         ▼
   Extract diff
   Store in KB
```

## Two Layers

### Public Layer (Open Source)
- Claude Code hook
- Personal SQLite + embeddings KB
- Individual developer memory
- Session-to-session continuity

### RDE Layer (Institutional)
- Mission-aware context
- Cross-mission learning
- Pattern recognition across missions
- Feeds into mission planning prompts

## Status

Specification phase.

## Name

**AI** = **A**fter **I**mage

The ghost of what was written, persisting across sessions.

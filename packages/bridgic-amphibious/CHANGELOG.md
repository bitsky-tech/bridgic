# Changelog

All notable changes to `bridgic-amphibious` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-04

Version 0.2.0 is a breaking redesign of the context, cognitive-worker, and
orchestration APIs. Applications built against 0.1.x must be migrated.

### Breaking changes

- Replaced the single cognitive-context model with two explicit loops:
  `OTAContext` owns a run's input, tools, and `OTARecord` history, while
  `Context` carries free-form longer-lived knowledge. Agents now declare both
  types as `AmphibiousAutoma[MyOTAContext, MyContext]`.
- Tools are no longer auto-injected. Declare them explicitly with
  `MyOTAContext.tool(...)` or provide them on a pre-built OTA context.
- Pass the default LLM to `arun(llm=...)`, use `user_input=` instead of
  `goal=`, and provide longer-lived state through `context=`. The active
  contexts are exposed as `ota_ctx` and `ctx`.
- Replaced `set_final_answer()` with `yield RETURN(value)` and replaced
  awaitable think-unit descriptors with `yield ThinkUnit("name")`.
- Agent, workflow, and hook templates now receive both contexts and must be
  async generators. Their valid yield primitives are checked at runtime.
- Simplified `CognitiveWorker` to the `thinking(ota_context, context=None)`
  template method. It returns the natural result of the selected LLM protocol,
  including `(tool_calls, content)` for tool selection.
- Replaced `AgentCall` with `EnterAgent`, which runs an isolated OTA episode
  using the class's `on_agent` strategy.
- Removed the former exposure hierarchy, cognitive-policy system,
  `CognitiveWorker.inline()`, `WorkflowDecision`, and the field-mutation
  `snapshot()` API.

### Added

- Added `LLMCall` for deterministic direct LLM calls from workflows and hooks.
- Added external Claude Code and Codex delegation with `AgentWorker`,
  `think_agent()`, and `ThinkAgent`, including an in-process MCP bridge for
  selected project tools.
- Added named human-input channels with `@human_channel`, shared by
  `HumanCall` and the opt-in `request_human_tool`.
- Added opt-in shell, filesystem, glob, grep, and human-input tool specs.

### Changed

- Rebuilt amphibious dispatch as a scope-aware state machine with bounded
  workflow recovery and configurable escalation to agent mode.
- Unified in-memory and persisted tracing. `trace=True` enables an
  `AgentTrace`; combining it with `workdir=` incrementally writes one
  `trace.json` under `<workdir>/runs/<run_id>/`.
- Updated the `bridgic-amphibious create` scaffold for the two-context,
  explicit-tool, yield-driven API.
- Rewrote the package README for the 0.2 API.

### Packaging

- Declared `uvicorn` as a direct runtime dependency.
- Declared the directly imported Pydantic 2.11 dependency explicitly.
- Declared support for Python 3.10 through 3.13. Python 3.14 is temporarily
  excluded because the current `bridgic-core` dependency requires Pydantic
  below 2.12; support can be restored after that constraint is upgraded and
  validated.

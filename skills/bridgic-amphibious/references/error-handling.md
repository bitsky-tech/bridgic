# Error Handling — `ErrorStrategy`

## What this controls

When the OTA cycle throws an exception, `ErrorStrategy` determines what happens next. Set it on `think_unit`:

```python
from bridgic.amphibious import ErrorStrategy

think_unit(worker, on_error=ErrorStrategy.RAISE)
think_unit(worker, on_error=ErrorStrategy.IGNORE)
think_unit(worker, on_error=ErrorStrategy.RETRY, max_retries=3)
```

## Strategies explained

### `RAISE` (default) — fail fast

```python
think_unit(worker, on_error=ErrorStrategy.RAISE)
```

**What happens**: Wraps the original exception in `RuntimeError` (via `raise RuntimeError(...) from e`) and re-raises immediately. The original exception is preserved as `__cause__`. The agent stops.

**What is discarded**: Everything from the failed cycle — the LLM's reasoning text, observation, and any partial tool results are lost.

**Use when**: Errors are unexpected and should halt execution. Good for development and for tasks where partial results are worse than no results.

### `IGNORE` — skip and continue

```python
think_unit(worker, on_error=ErrorStrategy.IGNORE)
```

**What happens**: The exception handler is literally `pass`. The failed cycle vanishes entirely:
- **No step is recorded** in `cognitive_history`
- **LLM reasoning text is discarded** — the `decision` object from the Think phase is lost
- **`finished` stays `False`** — the loop continues to the next attempt (if `max_attempts > 1`)
- Cleanup (tool/skill filter restoration, token accounting) still runs

**Use when**: Individual cycle failures are acceptable and the agent can recover by trying again with fresh reasoning. The next iteration starts a completely fresh observe-think-act cycle.

### `RETRY` — retry the entire cycle

```python
think_unit(worker, on_error=ErrorStrategy.RETRY, max_retries=3)
```

**What happens**: Re-runs the **entire** observe-think-act cycle (`_run_observe_think_act`), not just the failed phase. Each retry:
1. Calls `observation()` again (fresh observation)
2. Calls `worker.arun()` again (new LLM call)
3. Calls `_action()` again (new tool execution)

**Retry count**: Up to `max_retries + 1` additional attempts after the initial failure. Total executions = 1 (original) + `max_retries + 1` (retries). If all retries fail, raises `RuntimeError`.

**Use when**: Failures are transient (network timeouts, rate limits, flaky APIs).

## Choosing a strategy

| Scenario | Strategy | Why |
|----------|----------|-----|
| Development / debugging | `RAISE` | See errors immediately |
| Production agent, critical task | `RAISE` | Don't silently lose failures |
| Best-effort auxiliary step | `IGNORE` | Failure is acceptable, loop continues |
| Flaky external API calls | `RETRY` (max_retries=2-3) | Transient failures likely |
| Long ReAct loop, one step might fail | `IGNORE` or `RETRY` | Keep the loop running |

See also: [orchestration.md](orchestration.md) for `think_unit` parameters, [cognitive-worker.md](cognitive-worker.md) for worker hooks that run during the OTA cycle.

Confirmed. The only `.close()` calls are inside the observer/writer `close()` stubs themselves (capture.py:342, 582) which delegate to `r2_sink.close()`; nothing in `engine_workers.py` ever calls those stubs. All findings are grounded in the real commit. Writing the report.

# Async batched R2 upload review (commit `d972dd48`, branch `exp/async-r2-upload`)

## 1. Overall verdict

**Not yet sound for an unattended box-validation run. There is one structural blocker that defeats the entire point of the feature, plus three shutdown-path bugs that can hang or silently lose data.** The synchronous path (the default, `r2_async=False`) is unaffected and remains safe to ship; the opt-in async path is the problem surface.

The single most important defect is architectural, not a typo: the async failure-surfacing contract is **never wired in**. `CaptureWriter.close()` and `WeightTrajObserver.close()` exist and correctly delegate to `r2_sink.close()` -> `_raise_if_errors()`, but **no code in `engine_workers.py` ever calls them** (confirmed: the only `.close()` references at this commit are inside the stubs themselves, capture.py:342 and capture.py:582). The only run-end path that actually fires is the `atexit` handler, which by design swallows every exception. So in async mode a run can lose snapshots, write phantom manifest rows, and still exit `0` with the error buried in stderr. That is the exact silent-data-loss failure async mode was supposed to prevent.

Recommendation: land the fixes below (at minimum the **critical** four: wire `close()` into engine teardown, pass a bounded timeout from `_atexit_close`, fix the staged-bytes leak window, and bound the per-step `flush()`), then take it to a box. Until `close()` is called from teardown, do **not** run with `r2_async=True` on a real trajectory.

## 2. Confirmed issues, by severity

### Critical

**C1. Async upload failures never surface: `close()` defined but never called** (`silent-loss-async-no-close`, `no-final-close-barrier`)
`verl/workers/comm_eff/capture.py:334-342` (CaptureWriter.close stub), `:570-582` (WeightTrajObserver.close stub); root cause in `verl/workers/engine_workers.py` (zero `.close()` call sites on either observer).
The `_errors` list is only drained/raised by `_raise_if_errors()` via `flush()`/`close()`. With no run-end `close()`, the only path is `_atexit_close` (r2_sink.py:428-434), which catches and logs. A trajectory can drop `.pt` files and manifest entries, and the run still exits successfully.
**Fix:** Call `observer.close(timeout=...)` and `writer.close(timeout=...)` from the engine's teardown/shutdown path (the same place rollout/ref workers are finalized in `engine_workers.py`), in a `finally` so it runs on both success and exception. Let the raised `RuntimeError` propagate to fail the run loud.

**C2. `queue.join()` hangs forever during interpreter shutdown with dead daemon workers** (`critical-queue-join-deadlock`)
`verl/workers/comm_eff/r2_sink.py:398` (`flush()` -> `self._jobs.join()`), reached from `_atexit_close` -> `close()` with `timeout=None`.
Daemon workers (created `daemon=True`, r2_sink.py:321) are killed at interpreter shutdown. If one dies after `_jobs.get()` but before `task_done()`, `unfinished_tasks > 0` permanently and `join()` blocks forever. In multi-rank FSDP this stalls the rank and triggers collective timeouts.
**Fix:** Couple this with C3 so the atexit path always uses the bounded polling branch (r2_sink.py:400-410), never the unbounded `join()`.

**C3. `_atexit_close` passes no timeout, guaranteeing the unbounded path** (`missing-atexit-timeout`)
`verl/workers/comm_eff/r2_sink.py:433` (`self.close()` with no arg -> `close(timeout=None)` -> `flush(timeout=None)` -> `join()`).
The bounded fallback at r2_sink.py:400-410 is unreachable from atexit because `timeout` is always `None`.
**Fix:** Change line 433 to `self.close(timeout=<bounded>)` (e.g. 60-120s, ideally config-driven). This makes C2's hang impossible from the atexit path and bounds shutdown.

### High

**H1. Staged-bytes accounting leak: increment is inside the lock, `put()` is outside** (`R2SINK-001`)
`verl/workers/comm_eff/r2_sink.py:349-350`. `self._staged_bytes += nbytes` runs under `_staged_cond` (line 349); `self._jobs.put(...)` runs **after** the `with` block exits (line 350). A `KeyboardInterrupt`/`SystemExit`/signal in that window leaves `_staged_bytes` incremented for a job that was never enqueued. Since workers only decrement on jobs they actually dequeue, the counter is permanently inflated and every future producer blocks on backpressure against a near-empty queue. This is a deadlock-by-accounting.
**Fix:** Move `self._jobs.put(...)` **inside** the `with self._staged_cond:` block (after the increment, before lock release), so enqueue and accounting are atomic. `queue.Queue.put` on an unbounded queue does not block, so holding the lock across it is safe.

**H2. Manifest row written before R2 upload in async mode -> phantom entries** (`manifest-written-before-r2-upload`)
`verl/workers/comm_eff/capture.py:311-327` (CaptureWriter.dump), `:558-589` (WeightTrajObserver._dump_full). In async mode `upload()` enqueues and returns immediately, so the manifest row is committed before the upload is even attempted. If the upload later fails (and given C1 it is never surfaced), an analyst reads a manifest that attests to an artifact that only exists in local staging.
**Fix:** In async mode, do not let the local manifest row imply R2 durability. Either (a) have the worker write/patch the manifest row only after a verified upload (the sync path already does manifest-after-verify in `_do_upload`), or (b) add an explicit `r2_status: pending` field to the row that the worker flips to `verified` post-upload, plus a reconciliation step gated by C1's `close()`. Option (a) is cleaner and matches the sync semantics.

**H3. Per-step `flush()` has no timeout -> can block the training step indefinitely** (`per-step-flush-timeout-handling`)
`verl/workers/comm_eff/capture.py:556-567` (WeightTrajObserver.observe periodic flush). The per-`r2_flush_every_steps` flush calls `self.r2_sink.flush()` with no timeout, hitting the unbounded `_jobs.join()` (r2_sink.py:398). On a slow R2 endpoint with multi-GB snapshots this stalls the optimizer step until the queue drains, and a hung worker stalls it forever. Note this contradicts the same class's `close()`, which **does** pass `timeout` (capture.py:582) — the inconsistency is the tell.
**Fix:** Pass a bounded, config-driven timeout: `self.r2_sink.flush(timeout=self.r2_flush_timeout_s)`. On `TimeoutError`, decide policy explicitly — either fail the run (consistent with fail-loud) or log-and-continue while letting backpressure (H1, fixed) throttle the producer.

**H4. In-flight failures only surface on the *next* enqueue** (`enqueue-exception-on-error-not-idempotent`)
`verl/workers/comm_eff/r2_sink.py:341` (`_raise_if_errors()` at top of `_enqueue`). A failure at the last dump is never re-checked because no further enqueue happens, and with C1 unfixed the only catch is the swallowing atexit. This is a corollary of C1 rather than independent.
**Fix:** Resolved transitively by C1 (run-end `close()` raises) and H3 (bounded per-step flush raises). No separate change needed once those land; verify the last-step error is surfaced in the box-validation checklist.

### Medium

**M1. Thread creation during interpreter shutdown can raise** (`thread-creation-during-shutdown`)
`verl/workers/comm_eff/r2_sink.py:341` (`_ensure_workers()` from `_enqueue`) -> `:321` (`threading.Thread(...)`). If `upload()` is reached during shutdown (e.g. from a later-registered atexit handler or a `__del__`, before `_atexit_close` runs), `_closed` is still `False`, and `threading.Thread()` raises `RuntimeError: can't create new thread at interpreter shutdown`, which can break other cleanup handlers.
**Fix:** Guard `_ensure_workers()` with a shutdown check (e.g. `if sys.is_finalizing(): raise RuntimeError(...)` cleanly, or skip starting workers and route the artifact to the synchronous path). Low probability in this engine but cheap to harden.

### Low

**L1. Close-timeout returns success while workers are still hung** (`R2SINK-004`)
`verl/workers/comm_eff/r2_sink.py:413-426` (close). If `flush(timeout=T)` times out and the subsequent `t.join(timeout=T)` (line 426) also times out, `close()` returns normally with no `is_alive()` check; the daemon worker is then killed at exit possibly mid-manifest-write.
**Fix:** After the joins, check `any(t.is_alive() for t in self._workers)` and raise (or log loudly and propagate via the C1 path) so a timed-out shutdown is not reported as clean. Bounding the manifest write under `_manifest_lock` already limits the corruption window; an `is_alive()` signal closes the reporting gap.

**L2. `_manifest_lock` now wraps the synchronous manifest append** (`FINDING-001`)
`verl/workers/comm_eff/r2_sink.py:298-303` (`_do_upload`). The unconditional `_manifest_lock` adds an uncontended acquire/release to the sync path. Manifest bytes are identical; behavior is unchanged. This is expected collateral of making `_do_upload` thread-safe and is not a defect — noting it only so the "default-OFF is byte-identical" claim is qualified to *output*-identical, not *code-path*-identical.
**Fix:** None required. Keep as-is.

## 3. Residual risk that only a training box can validate

These cannot be settled on a laptop and must be checked during the gated box-validation run, ideally with the C1-C3 + H1/H3 fixes in place:

- **Real R2 throughput under N concurrent streams.** `upload_workers=4` against the real R2 endpoint: measure aggregate MB/s, whether concurrent `aws s3 cp` saturate the box NIC, and whether the workers keep up with dump cadence or chronically sit at the backpressure cap. Tune `upload_workers` and `max_staged_gb` from observed throughput.
- **Real disk backpressure under a ~492 GB trajectory.** Confirm the staged-bytes cap actually bounds local staging below box disk capacity end to end, that the producer blocks (rather than ENOSPC) when uploaders fall behind, and — critically after the H1 fix — that the staged-bytes counter returns to ~0 at the end with no leak inflating it over a long run. Watch for the single-artifact-larger-than-cap admit path (r2_sink.py:347) on the largest snapshot.
- **`atexit`/`close()` firing under real engine and Ray teardown.** Verify the C1 engine-level `close()` runs on both normal completion and on exception, on every rank, and that it fires *before* Ray/daemon-thread teardown rather than racing it. Confirm the bounded atexit (C3) actually terminates instead of hanging a rank, and that a deliberately injected upload failure on the last step propagates as a non-zero run exit rather than a buried stderr log. Multi-rank: confirm no rank blocks the collective at shutdown.

Files reviewed at commit `d972dd48a992056f6fec8f262519bc8fcc3a8661`: `/Users/shamane/Documents/verl/verl/workers/comm_eff/r2_sink.py`, `/Users/shamane/Documents/verl/verl/workers/comm_eff/capture.py`, `/Users/shamane/Documents/verl/verl/workers/engine_workers.py`.
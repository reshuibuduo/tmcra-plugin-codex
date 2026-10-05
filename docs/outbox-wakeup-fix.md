# Outbox wakeup handoff

The Windows E2E timeout revealed a genuine producer/worker race. Increasing the
test timeout from 8 to 20 seconds did not fix it, so that timeout change has been
reverted.

A worker could finish its last empty-queue check while still holding the drain
lock. A producer then enqueued a new record, saw that lock, wrote a request marker
and returned. The old worker released its lock and exited without consuming the
marker. The durable record remained local until another host event woke a worker.

The fix pairs two checks:

1. The producer rechecks the worker lock after writing its request. If the worker
   has already released it, the producer continues the launch path.
2. The worker releases its lock before its final request check. A pending signal
   causes another drain attempt; a competing active worker stops that attempt.

Concurrent request-marker creation is now idempotent and does not truncate or
delete another producer's signal.

`node tests/outbox_wakeup_mock.mjs` uses child-process-only filesystem gates to
exercise both exit interleavings deterministically, without changing production
timings or source files. The previous implementation reproduces `0 !== 1` on the
first race; the fix passes both interleavings and 20 concurrent signals. The test
is part of the full Codex contract suite. Only synthetic loopback memory is used.

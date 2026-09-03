import { open, mkdir, rm, stat, utimes } from "node:fs/promises";
import { join } from "node:path";

import {
  appendLog,
  clearOutboxCircuit,
  completeOutboxTurn,
  coalesceOutboxCheckpoints,
  getJob,
  getScopeRecovery,
  listOutboxTurns,
  loadConfig,
  markOutboxSubmitted,
  openOutboxCircuit,
  outboxCircuitForEntry,
  pluginDataDir,
  removeOutboxTurn,
  sortOutboxEntries,
  submitOutboxTurn,
} from "./tmcra_client.mjs";

const LOCK_STALE_MS = 5 * 60 * 1000;
const LOCK_HEARTBEAT_MS = 30 * 1000;
const MAX_CIRCUIT_SLEEP_MS = 60 * 1000;
const MAX_DRAIN_LIFETIME_MS = 6 * 60 * 60 * 1000;
const SUPPORT_RECOVERY_RECHECK_MS = Math.max(
  50,
  Number(process.env.TMCRA_OUTBOX_SUPPORT_RECHECK_MS || 60 * 1000),
);
const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
const TRANSIENT_BACKOFF_MIN_MS = 1000;
const TRANSIENT_BACKOFF_MAX_MS = 60 * 1000;
const JOB_POLL_MS = Math.max(500, Number(process.env.TMCRA_OUTBOX_JOB_POLL_MS || 2_000));
const MAX_IN_FLIGHT_JOBS = Math.max(
  1,
  Math.min(8, Number(process.env.TMCRA_OUTBOX_MAX_IN_FLIGHT_JOBS || 4)),
);
const outboxDirectory = join(pluginDataDir(), "outbox");
const lockPath = join(outboxDirectory, ".drain.lock");
const requestPath = join(outboxDirectory, ".drain.request");
const launchPath = join(outboxDirectory, ".drain.launch");

async function hasDrainRequest() {
  try {
    await stat(requestPath);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

function transientRetryDelay(error, consecutiveFailures) {
  const retryAfterMs = Number(error?.retryAfterSeconds) * 1000;
  if (Number.isFinite(retryAfterMs) && retryAfterMs > 0) {
    return Math.min(TRANSIENT_BACKOFF_MAX_MS, retryAfterMs);
  }
  return Math.min(
    TRANSIENT_BACKOFF_MAX_MS,
    TRANSIENT_BACKOFF_MIN_MS * (2 ** Math.min(6, Math.max(0, consecutiveFailures - 1))),
  );
}

function isTransientFailure(error) {
  if (Number.isInteger(error?.status)) return TRANSIENT_STATUS.has(error.status);
  return error?.name === "AbortError" || error?.code === "ECONNRESET" ||
    error?.code === "ECONNREFUSED" || error?.code === "ETIMEDOUT" ||
    error?.code === "EAI_AGAIN" || error instanceof TypeError;
}

async function acquireLock() {
  await mkdir(outboxDirectory, { recursive: true });
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const handle = await open(lockPath, "wx", 0o600);
      try {
        await handle.writeFile(`${process.pid}\n`, { encoding: "utf8" });
      } finally {
        await handle.close();
      }
      return true;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      let lock;
      try {
        lock = await stat(lockPath);
      } catch (statError) {
        if (statError?.code === "ENOENT") continue;
        throw statError;
      }
      if (Date.now() - lock.mtimeMs < LOCK_STALE_MS) return false;
      await rm(lockPath, { force: true });
    }
  }
  return false;
}

async function drain() {
  if (!(await acquireLock())) return;
  await rm(launchPath, { force: true });
  await rm(requestPath, { force: true });
  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    const now = new Date();
    void utimes(lockPath, now, now).catch(() => {});
  }, LOCK_HEARTBEAT_MS);
  let consecutiveTransientFailures = 0;
  try {
    const config = await loadConfig();
    for (;;) {
      let entries = await listOutboxTurns();
      if (!entries.length) {
        if (await hasDrainRequest()) {
          await rm(requestPath, { force: true });
          await new Promise((resolve) => setTimeout(resolve, 100));
          continue;
        }
        await new Promise((resolve) => setTimeout(resolve, 200));
        if ((await listOutboxTurns()).length || await hasDrainRequest()) continue;
        return;
      }
      entries = await coalesceOutboxCheckpoints(entries);
      entries = sortOutboxEntries(entries);
      entries = [
        ...entries.filter((entry) => Boolean(entry.jobId)),
        ...entries.filter((entry) => !entry.jobId),
      ];
      let earliestRetryAt = null;
      let sawQuarantineWait = false;
      let transientRetryAt = null;
      let jobPollAt = null;
      let inFlightJobs = entries.filter((entry) => Boolean(entry.jobId)).length;
      let sawPermanentFailure = false;
      const supportRecoveryChecks = new Map();
      const supportRecoveryFailuresLogged = new Set();
      for (const entry of entries) {
        if (entry.jobId) {
          try {
            const job = await getJob(entry.jobId, config);
            const status = String(job?.status || "unknown");
            if (["succeeded", "failed", "cancelled"].includes(status)) {
              const completed = await completeOutboxTurn(entry, job);
              inFlightJobs = Math.max(0, inFlightJobs - 1);
              await clearOutboxCircuit(entry);
              await appendLog(completed.state === "succeeded" ? "ingest_succeeded" : "ingest_failed", {
                host: entry.host,
                pluginVersion: entry.pluginVersion,
                lifecycleContractVersion: entry.lifecycleContractVersion || null,
                projectId: entry.projectId,
                jobId: entry.jobId,
                outboxId: entry.outboxId,
                status,
              });
              if (completed.state === "failed") sawPermanentFailure = true;
            } else {
              const pollAt = Date.now() + JOB_POLL_MS;
              jobPollAt = jobPollAt === null ? pollAt : Math.min(jobPollAt, pollAt);
            }
          } catch (error) {
            if (!isTransientFailure(error)) {
              sawPermanentFailure = true;
              await appendLog("ingest_job_status_failed", {
                outboxId: entry.outboxId,
                jobId: entry.jobId,
                status: error?.status || null,
                code: error?.code || null,
                requestId: error?.requestId || null,
              });
              continue;
            }
            consecutiveTransientFailures += 1;
            const retryAt = Date.now() + transientRetryDelay(error, consecutiveTransientFailures);
            transientRetryAt = transientRetryAt === null
              ? retryAt
              : Math.min(transientRetryAt, retryAt);
          }
          continue;
        }
        const circuit = await outboxCircuitForEntry(entry);
        if (circuit?.open) {
          if (circuit.requiresSupport) {
            let recoveryPromise = supportRecoveryChecks.get(circuit.key);
            if (!recoveryPromise) {
              recoveryPromise = getScopeRecovery(entry.scope, config);
              supportRecoveryChecks.set(circuit.key, recoveryPromise);
            }
            try {
              const recovery = await recoveryPromise;
              if (recovery?.writes_available === true) {
                await clearOutboxCircuit(entry);
                await appendLog("ingest_scope_recovered", {
                  outboxId: entry.outboxId,
                  state: recovery.state || null,
                  phase: recovery.phase || null,
                });
              } else {
                sawQuarantineWait = true;
                const retryAt = Date.now() + SUPPORT_RECOVERY_RECHECK_MS;
                earliestRetryAt = earliestRetryAt === null
                  ? retryAt
                  : Math.min(earliestRetryAt, retryAt);
                continue;
              }
            } catch (error) {
              sawQuarantineWait = true;
              const retryAt = Date.now() + SUPPORT_RECOVERY_RECHECK_MS;
              earliestRetryAt = earliestRetryAt === null
                ? retryAt
                : Math.min(earliestRetryAt, retryAt);
              if (!supportRecoveryFailuresLogged.has(circuit.key)) {
                supportRecoveryFailuresLogged.add(circuit.key);
                await appendLog("ingest_scope_recovery_status_unavailable", {
                  outboxId: entry.outboxId,
                  status: error?.status || null,
                  code: error?.code || null,
                  requestId: error?.requestId || null,
                });
              }
              continue;
            }
          } else if (circuit.retryAt) {
            const retryAt = Date.parse(circuit.retryAt);
            earliestRetryAt = earliestRetryAt === null
              ? retryAt
              : Math.min(earliestRetryAt, retryAt);
            sawQuarantineWait = true;
            continue;
          }
        }
        if (sawPermanentFailure) continue;
        if (inFlightJobs >= MAX_IN_FLIGHT_JOBS) {
          const pollAt = Date.now() + JOB_POLL_MS;
          jobPollAt = jobPollAt === null ? pollAt : Math.min(jobPollAt, pollAt);
          continue;
        }
        try {
          const result = await submitOutboxTurn(entry, config);
          const submitted = await markOutboxSubmitted(entry, result);
          await clearOutboxCircuit(entry);
          inFlightJobs += 1;
          const pollAt = Date.now() + JOB_POLL_MS;
          jobPollAt = jobPollAt === null ? pollAt : Math.min(jobPollAt, pollAt);
          consecutiveTransientFailures = 0;
          await appendLog("ingest_submitted", {
            host: entry.host,
            pluginVersion: entry.pluginVersion,
            lifecycleContractVersion: entry.lifecycleContractVersion || null,
            projectId: entry.projectId,
            jobId: submitted.jobId,
            outboxId: submitted.outboxId,
          });
        } catch (error) {
          if (error?.status === 422 && error?.code === "scope_quarantined") {
            try {
              error.recovery = await getScopeRecovery(entry.scope, config);
            } catch {
              // Older servers may not expose recovery progress yet. The
              // bounded circuit remains the safe compatibility behavior.
            }
            const circuitState = await openOutboxCircuit(entry, error);
            if (circuitState.newlyOpened) {
              await appendLog("ingest_paused_scope_recovery", {
                outboxId: entry.outboxId,
                status: error.status,
                code: error.code,
                retryAt: circuitState.retryAt,
                requiresSupport: circuitState.requiresSupport === true,
                requestId: error.requestId || null,
              });
            }
            if (circuitState.retryAt) {
              const retryAt = Date.parse(circuitState.retryAt);
              earliestRetryAt = earliestRetryAt === null
                ? retryAt
                : Math.min(earliestRetryAt, retryAt);
              sawQuarantineWait = true;
            }
            continue;
          }
          if (isTransientFailure(error)) {
            consecutiveTransientFailures += 1;
            const retryDelay = transientRetryDelay(error, consecutiveTransientFailures);
            transientRetryAt = Date.now() + retryDelay;
            await appendLog("ingest_retry_waiting", {
              outboxId: entry.outboxId,
              status: error.status || null,
              code: error.code || null,
              retryAt: new Date(transientRetryAt).toISOString(),
              requestId: error.requestId || null,
            });
            // Queue saturation and transport failures affect the batch. Stop
            // here so the remaining entries are not used as retry probes.
            break;
          }
          sawPermanentFailure = true;
          await appendLog("ingest_retry_pending", {
            outboxId: entry.outboxId,
            message: error.message,
            status: error.status || null,
            requestId: error.requestId || null,
          });
        }
      }
      if (sawPermanentFailure || Date.now() - startedAt >= MAX_DRAIN_LIFETIME_MS) return;
      const nextRetryAt = [earliestRetryAt, transientRetryAt, jobPollAt]
        .filter((value) => Number.isFinite(value))
        .sort((left, right) => left - right)[0] ?? null;
      if ((!sawQuarantineWait && transientRetryAt === null && jobPollAt === null) || nextRetryAt === null) return;
      const delay = Math.min(
        MAX_CIRCUIT_SLEEP_MS,
        Math.max(50, nextRetryAt - Date.now()),
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  } finally {
    clearInterval(heartbeat);
    await rm(lockPath, { force: true });
  }
}

await drain();

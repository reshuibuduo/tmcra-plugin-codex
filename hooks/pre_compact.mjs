import {
  checkpointTaskContinuity,
  continueNormally,
  failOpen,
  readHookInput,
  recordCompaction,
} from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("pre_compact", async () => {
  await checkpointTaskContinuity(input, {
    reason: `pre_compact_${String(input.trigger || "unknown")}`,
    force: true,
    queueRemote: true,
  });
  await recordCompaction(input, "started");
  continueNormally();
});

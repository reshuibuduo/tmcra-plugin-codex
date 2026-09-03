import {
  checkpointTaskContinuity,
  completeTaskContinuity,
  continueNormally,
  failOpen,
  ingestCompletedTurn,
  readHookInput,
  unregisterSubagentLifecycle,
} from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("subagent_stop", async () => {
  const result = await ingestCompletedTurn(input);
  if (result) {
    await completeTaskContinuity(input);
  } else {
    await checkpointTaskContinuity(input, {
      reason: "subagent_stop_without_complete_turn",
      force: true,
      queueRemote: true,
    });
  }
  await unregisterSubagentLifecycle(input);
  continueNormally();
});

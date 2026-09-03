import {
  checkpointTaskContinuity,
  completeTaskContinuity,
  continueNormally,
  failOpen,
  ingestCompletedTurn,
  hostName,
  readHookInput,
  recordFailedTurn,
} from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("stop", async () => {
  if (input.hook_event_name === "StopFailure") {
    // StopFailure.last_assistant_message is provider error text, not assistant memory.
    await recordFailedTurn(input);
    continueNormally();
    return;
  }
  const result = await ingestCompletedTurn(input);
  if (result) {
    const hasScheduledClaudeWork = hostName(input) === "claude-code" &&
      input.hook_event_name === "Stop" &&
      (Array.isArray(input.background_tasks) && input.background_tasks.length > 0 ||
        Array.isArray(input.session_crons) && input.session_crons.length > 0);
    if (hasScheduledClaudeWork) {
      await checkpointTaskContinuity(input, {
        reason: "stop_with_background_work",
        force: true,
        queueRemote: true,
      });
    } else {
      await completeTaskContinuity(input);
    }
  } else {
    await checkpointTaskContinuity(input, {
      reason: "stop_without_complete_turn",
      force: true,
      queueRemote: true,
    });
  }
  continueNormally();
});

import {
  emit,
  failOpen,
  readHookInput,
  recordSessionStart,
  resumeTaskContinuity,
} from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("session_start", async () => {
  await recordSessionStart(input);
  const source = String(input.source || "startup");
  if (!["compact", "resume"].includes(source)) {
    emit({ continue: true });
    return;
  }
  const context = await resumeTaskContinuity(input);
  if (!context) {
    emit({ continue: true });
    return;
  }
  emit({
    continue: true,
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: context,
    },
  });
});

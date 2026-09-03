import { emit, failOpen, readHookInput, recallForContext, rememberPrompt } from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("user_prompt_submit", async () => {
  const remembered = await rememberPrompt(input);
  const query = String(input.prompt || "").trim();
  if (!query) {
    emit({ continue: true });
    return;
  }
  const result = await recallForContext(
    remembered?.turnId ? { ...input, tmcra_turn_id: remembered.turnId } : input,
    query,
  );
  if (!result.context) {
    emit({ continue: true });
    return;
  }
  emit({
    continue: true,
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: result.context,
    },
  });
});

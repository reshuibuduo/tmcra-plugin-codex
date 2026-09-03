import { continueNormally, failOpen, readHookInput, recordToolUse } from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("post_tool_use", async () => {
  await recordToolUse(input);
  continueNormally();
});

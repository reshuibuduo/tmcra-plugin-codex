import { continueNormally, failOpen, readHookInput, recordCompaction } from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("post_compact", async () => {
  await recordCompaction(input, "completed");
  continueNormally();
});

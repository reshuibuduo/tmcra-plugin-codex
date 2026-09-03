import {
  continueNormally,
  failOpen,
  readHookInput,
  registerSubagentLifecycle,
} from "./hook_common.mjs";

const input = await readHookInput();
await failOpen("subagent_start", async () => {
  await registerSubagentLifecycle(input);
  continueNormally();
});

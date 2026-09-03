import { run } from "./run_hook.mjs";

// Claude Code has a separate plugin runtime. Set the platform explicitly so
// model/session fields can never make a Claude event look like a Codex event.
process.env.TMCRA_CLIENT_PLATFORM = "claude-code";

try {
  await run();
} catch (error) {
  process.stderr.write(`TMCRA Claude Code hook failed open: ${error.message}\n`);
  process.stdout.write('{"continue":true}\n');
}

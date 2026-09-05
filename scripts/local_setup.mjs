import { localSetupAction, startMemoryCenter } from './memory_center.mjs';

// This entry is available before any TMCRA account or API credential exists.
const center = await startMemoryCenter({ invoke: localSetupAction, open: !process.argv.includes('--no-open') });
process.stdout.write(JSON.stringify({url:center.url, localSetup:true, accountRequired:false})+'\n');
await new Promise(resolve => center.server.once('close', resolve));

import { createMemoryActions, startMemoryCenter } from './memory_center.mjs';

// This entry is available before any TMCRA account or API credential exists.
const invoke = createMemoryActions({
  config: { baseUrl: 'http://127.0.0.1:2009', apiKey: 'local-setup-placeholder' },
  scope: 'local-installation', sessionId: 'local-installation',
  request: async () => { throw Error('请先在模型配置页面完成本地安装；此设置入口不会连接 TMCRA 服务器。'); },
});
const center = await startMemoryCenter({ invoke, open: !process.argv.includes('--no-open') });
process.stdout.write(JSON.stringify({url:center.url, localSetup:true, accountRequired:false})+'\n');
await new Promise(resolve => center.server.once('close', resolve));

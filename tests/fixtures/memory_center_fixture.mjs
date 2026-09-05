import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { controlKey, memoryPolicy, updateTask, recordMemoryActivity } from "../../scripts/memory_controls.mjs";
import { createMemoryActions, startMemoryCenter } from "../../scripts/memory_center.mjs";
import { writeProviderConfig } from "../../scripts/provider_config.mjs";

// Isolated, explicitly labelled demo data. Never connect to production services.
export async function memoryCenterFixture({ empty = false, providerTestConfig } = {}) {
  const root = await mkdtemp(join(tmpdir(), "tmcra-workspace-preview-"));
  process.env.TMCRA_MEMORY_STATE_DIR = root;
  if (providerTestConfig) await writeProviderConfig({writer:{provider:'openai-compatible',...providerTestConfig},organizer:{inheritWriter:true}},join(root,'providers.json'));
  const config = { baseUrl: "https://example.invalid", apiKey: "never-visible-test-secret" };
  const scope = "tmcra / 插件研发";
  const sessionId = "local-design-preview";
  const key = controlKey(config, scope);
  const capture = await memoryPolicy(key, sessionId);
  if (!empty) {
    await updateTask(key, sessionId, { objective: "接入本地模型配置", summary: "Writer 与 Organizer 的配置入口已接通，密钥保护与连通性检查完成。", status: "completed" });
    await updateTask(key, sessionId, { objective: "完善跨会话任务接续", summary: "支持保存目标、最近结果与下一步。多任务并行时，由用户明确选择接续对象。", nextStep: "覆盖任务切换与上下文压缩后的恢复场景" });
    await updateTask(key, sessionId, { objective: "把记忆工作台打磨成日常工具", summary: "任务、记忆来源与会话控制已接通。让信息层级更清晰，让每一个操作都有明确的反馈。", nextStep: "核对来源详情，完成桌面与移动端的交互验收" });
    const source = (id, content, roles = ["user"]) => ({ memory_id: id, actor_roles: roles, timestamp: "2026-09-05T08:42:00Z", content });
    await recordMemoryActivity(capture, { kind: "recall", query: "模型密钥应该保存在哪里？", selection: { characters: 1260, estimatedTokens: 420, omitted: [] }, layers: [
      { scope: "个人偏好", status: "success", queryId: "demo-query-01", sources: [source("source-privacy-01", "用户的 Writer 和 Organizer 模型密钥保存在本机配置中。连接外部服务前，清楚展示用途与授权边界。")] },
    ] });
    await recordMemoryActivity(capture, { kind: "recall", query: "上一次做到哪了，接下来做什么？", selection: { characters: 2480, estimatedTokens: 827, omitted: [] }, layers: [
      { scope, status: "success", queryId: "demo-query-02", sources: [source("source-task-01", "任务接续、会话模式与纠错接口已完成。下一步验证真实页面交互与发布资源。"), source("source-safety-01", "来源内容必须作为纯文本展示。测试样本：<img src=x onerror=alert(1)>", ["assistant"])] },
    ] });
    await recordMemoryActivity(capture, { kind: "recall", query: "继续优化记忆工作台", selection: { characters: 3840, estimatedTokens: 1280, omitted: [{ scope, reason: "duplicate", characters: 640 }, { scope, reason: "budget", characters: 9800 }] }, layers: [
      { scope, status: "success", queryId: "demo-query-03", sources: [source("source-design-01", "工作台需要把当前任务、最近进展和下一步放在最容易看到的位置。记忆来源可以直接核对原文，纠错和恢复都要有清晰的操作反馈。"), source("source-contract-02", "会话关闭后，新的对话不参与记忆捕获。重新开启时，关闭期间的内容与旧代待发送记录都不会补写。")] },
      { scope: "个人偏好", status: "success", queryId: "demo-query-04", sources: [source("source-preference-03", "沟通聚焦重点，说明当前结果与实际边界。任务完成后保留必要的验证记录。", ["user"])] },
    ] });
    await recordMemoryActivity(capture, { kind: "write", state: "succeeded", jobId: "job_demo_01" });
    await recordMemoryActivity(capture, { kind: "write", state: "succeeded", jobId: "job_demo_02" });
    await recordMemoryActivity(capture, { kind: "write", state: "pending", jobId: "job_demo_03" });
  }
  const calls = [];
  const behavior = { failNext: false, legacyResponse: false };
  const concepts = [
    ['模型密钥留在本机','Writer 和后台整理通过本机配置访问模型服务。','preference','user'],
    ['聊天纠错先确认','每次修改展示原文、新内容和范围，用户确认后生效。','decision','user'],
    ['隔离待确认对话','等待确认与取消的纠错对话跳过自动写入。','solution','assistant'],
    ['记忆来源可追溯','知识条目与关系都能回到对应的原始证据。','requirement','user'],
    ['任务接续已接通','保存目标、进展和下一步，继续时核对实际状态。','result','assistant'],
    ['图谱与知识库联动','在同一个工作台中浏览个人知识和关系。','goal','user'],
  ];
  const nodes = empty ? [] : concepts.map(([label,summary,memory_type,actor_role],i)=>({id:`evidence-${i}`,level:'evidence',evidence_kind:'memory',memory_id:`memory-${i}`,source_record_ids:[`source-${i}`],label,summary,memory_type,actor_role}));
  const edges = empty ? [] : [[1,2,'leads_to'],[2,1,'reinforces'],[3,5,'applies_to'],[0,5,'applies_to'],[4,5,'related']].map(([a,b,type],i)=>({id:`edge-${i}`,source:`evidence-${a}`,target:`evidence-${b}`,type,origin:'agent',reason:'虚构演示关系，可通过原始证据核对。'}));
  const invoke = createMemoryActions({ config, scope, sessionId, globalScope: "个人偏好",
    status: async () => empty ? {} : { queued: 1, succeeded: 2, pending: [{ jobId: "job_demo_03", state: "pending" }] },
    request: async (path, options) => {
      if (options.method === 'GET') {
        if(path.includes('/knowledge-base'))return {projection_state:'ready',pages:empty?[]:[
          {page_id:'page-privacy',collection:'personal',title:'我的记忆使用偏好',abstract:'把隐私、确认与来源追溯作为日常记忆管理的边界。',claims:[{text:concepts[0][1],status:'confirmed',evidence_ids:['evidence-0']},{text:concepts[1][1],status:'confirmed',evidence_ids:['evidence-1']}],sections:[{heading:'如何处理纠错',body:'先展示修改，再确认提交。取消时保持原记忆；需要时可以继续核对来源。',evidence_ids:['evidence-2']}]},
          {page_id:'page-work',collection:'project',title:'记忆工作台的交付进展',abstract:'把任务、知识、关系和模型配置放进同一个工作空间。',claims:[{text:concepts[4][1],status:'confirmed',evidence_ids:['evidence-4']}],sections:[]},
        ],evidence_catalog:Object.fromEntries(nodes.map(n=>[n.id,n]))};
        if(path.includes('/visual-atlas'))return {projection_state:'ready',nodes,edges};
        if(path.includes('/evidence'))return {items:[{source_record_id:decodeURIComponent(path.split('/nodes/')[1].split('/')[0]),actor_role:'user',text:'这是一段虚构的演示证据：配置密钥保存在本机，修改记忆之前需要用户确认。'}],page:{has_more:false,next_cursor:null}};
        throw new Error('Unknown demo read endpoint');
      }
      calls.push({ path, ...options });
      if (behavior.failNext) { behavior.failNext = false; throw new Error("模拟响应丢失，请重试同一操作。"); }
      if (behavior.legacyResponse) return { feedback_id: "demo-note-only" };
      return { effective: true, correction_index_status: "pending", preview: true };
    },
  });
  const center = await startMemoryCenter({ open: false, providerConfigPath: join(root, 'providers.json'), invoke: async (action, args) => {
    const result = await invoke(action, args);
    return action === "dashboard" ? { ...result, preview: true } : result;
  } });
  return { ...center, key, sessionId, config, calls, behavior, async dispose() {
    if (center.server.listening) await new Promise(done => center.server.close(done));
    await rm(root, { recursive: true, force: true });
  } };
}

if (process.argv.includes("--serve")) {
  const fixture = await memoryCenterFixture({providerTestConfig: process.env.TMCRA_TEST_PROVIDER_JSON ? JSON.parse(process.env.TMCRA_TEST_PROVIDER_JSON) : undefined});
  process.stdout.write(JSON.stringify({ url: fixture.url, demoData: true, productionAccess: false }) + "\n");
  fixture.server.once("close", () => fixture.dispose());
}

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { memoryPolicy, memoryDashboard } from "../scripts/memory_controls.mjs";
import { memoryCenterFixture } from "./fixtures/memory_center_fixture.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.TMCRA_PLAYWRIGHT_MODULE || "playwright");
const output = resolve("test-artifacts");
await mkdir(output, { recursive: true });
let browser, fixture;
const errors = [];
try {
  fixture = await memoryCenterFixture();
  browser = await chromium.launch({ headless: true, ...(process.env.TMCRA_BROWSER_EXECUTABLE ? { executablePath: process.env.TMCRA_BROWSER_EXECUTABLE } : {}) });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1060 }, reducedMotion: "reduce" });
  const requests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => requests.push(request.url()));
  const navigate = async name => {
    if (await page.locator("#menuToggle").isVisible()) await page.locator("#menuToggle").click();
    await page.locator("#navigation").getByRole("button", { name, exact: true }).click();
  };
  await page.goto(fixture.url);
  await page.getByRole("heading", { name: "记忆工作台", exact: true }).waitFor();
  await page.getByText("把记忆工作台打磨成日常工具", { exact: true }).first().waitFor();
  assert.equal(await page.locator("body").innerText().then(text => text.includes(fixture.config.apiKey)), false);
  assert.equal(new URL(page.url()).hash, "");
  await page.locator('.brand-image-frame img').evaluate(img => img.decode());
  const logoResponse = await page.request.get(fixture.baseUrl + '/assets/tmcra-logo.png');
  assert.equal(logoResponse.headers()['content-type'], 'image/png');
  assert((await logoResponse.body()).length > 1000);
  await page.screenshot({ path: resolve(output, "memory-center.png"), fullPage: true });

  await navigate("记忆来源");
  await page.locator(".source-card").first().waitFor();
  await page.screenshot({ path: resolve(output, "memory-center-sources.png"), fullPage: true });
  fixture.behavior.failNext = true;
  await page.getByRole("button", { name: "纠正内容", exact: true }).first().click();
  assert.equal(await page.locator("#objectiveLabel").isVisible(), false);
  await page.locator("#replacement").fill("新的事实：Writer 和 Organizer 密钥保存在当前用户的本机配置中。");
  await page.screenshot({ path: resolve(output, "memory-center-correction.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("button", { name: "保存纠正", exact: true }).click();
  await page.locator("#editorError").getByText("模拟响应丢失，请重试同一操作。", { exact: true }).waitFor();
  await page.getByRole("button", { name: "保存纠正", exact: true }).click();
  await page.locator("#editor").waitFor({ state: "hidden" });
  assert.equal(fixture.calls.length, 2);
  assert.equal(fixture.calls[0].headers["Idempotency-Key"], fixture.calls[1].headers["Idempotency-Key"]);
  assert.equal(fixture.calls[0].body.memory_ids[0], "source-design-01");

  await page.getByRole("button", { name: "忽略", exact: true }).first().click();
  await page.getByRole("button", { name: "取消", exact: true }).click();
  assert.equal(fixture.calls.length, 2);
  fixture.behavior.legacyResponse = true;
  await page.getByRole("button", { name: "忽略", exact: true }).first().click();
  await page.getByRole("button", { name: "确认忽略", exact: true }).click();
  await page.locator("#editorError").getByText("服务端尚未确认规则生效，请检查服务版本后重试。", { exact: true }).waitFor();
  fixture.behavior.legacyResponse = false;
  await page.getByRole("button", { name: "确认忽略", exact: true }).click();
  await page.locator("#editor").waitFor({ state: "hidden" });
  assert.equal(fixture.calls[2].headers["Idempotency-Key"], fixture.calls[3].headers["Idempotency-Key"]);
  await page.getByRole("button", { name: "恢复召回", exact: true }).first().click();
  await page.locator("#editor").getByRole("button", { name: "恢复召回", exact: true }).click();
  await page.locator("#editor").waitFor({ state: "hidden" });
  assert.equal(fixture.calls.at(-1).body.action, "restore");

  await page.locator("#search").fill("onerror");
  await page.locator(".source-content").filter({ hasText: "onerror" }).waitFor();
  assert.equal(await page.locator(".source-card img").count(), 0);
  await page.locator("#search").fill("no-record-matches-this");
  await page.getByRole("heading", { name: "没有找到相关记录" }).waitFor();
  await page.getByRole("button", { name: "清空搜索" }).click();

  await navigate("任务接续");
  await page.locator("#newTaskSecondary").click();
  await page.locator("#objective").fill("验证新的任务交互");
  await page.locator("#taskSummary").fill("通过实际接口保存任务");
  await page.locator("#next").fill("核对任务绑定");
  await page.getByRole("button", { name: "确认保存", exact: true }).click();
  await page.locator("#editor").waitFor({ state: "hidden" });
  await page.getByRole("heading", { name: "验证新的任务交互", exact: true }).waitFor();
  let dashboard = await memoryDashboard(fixture.key, fixture.sessionId);
  const created = dashboard.tasks.find(task => task.objective === "验证新的任务交互");
  assert.equal(dashboard.currentTaskId, created.id);
  const createdCard = page.locator(".task-card").filter({ hasText: "验证新的任务交互" });
  await createdCard.getByRole("button", { name: "标记完成" }).click();
  await page.locator("#editor").getByRole("button", { name: "标记完成" }).click();
  await page.locator("#editor").waitFor({ state: "hidden" });
  await page.getByRole("tab", { name: "已完成", exact: true }).click();
  await page.getByRole("heading", { name: "验证新的任务交互" }).waitFor();
  await page.getByRole("tab", { name: "进行中", exact: true }).click();
  await page.locator(".task-card").filter({ hasText: "把记忆工作台打磨成日常工具" }).getByRole("button", { name: "在此会话继续" }).click();
  await page.locator(".task-card").filter({ hasText: "把记忆工作台打磨成日常工具" }).locator('.bound').waitFor();
  await page.screenshot({ path: resolve(output, "memory-center-tasks.png"), fullPage: true });

  await navigate("会话设置");
  await page.getByRole("radio", { name: /关闭记忆/u }).check();
  await page.getByRole("button", { name: "应用变更" }).click();
  await page.getByText("当前模式：关闭记忆", { exact: true }).waitFor();
  assert.equal((await memoryPolicy(fixture.key, fixture.sessionId)).read, false);
  assert.equal(await page.locator("#saveMode").isDisabled(), true);
  await navigate("任务接续");
  assert.equal(await page.locator("#newTaskSecondary").isDisabled(), true);
  await navigate("记忆来源");
  assert.equal(await page.getByRole("button", { name: "纠正内容", exact: true }).first().isDisabled(), true);
  await navigate("会话设置");
  await page.getByRole("radio", { name: /正常读写/u }).check();
  await page.getByRole("button", { name: "应用变更" }).click();
  await page.getByText("当前模式：正常读写", { exact: true }).waitFor();
  await page.getByRole("button", { name: "6,000 · 精简" }).click();
  await page.getByRole("button", { name: "保存预算" }).click();
  await page.getByText("召回预算已保存", { exact: true }).waitFor();
  assert.equal((await memoryDashboard(fixture.key, fixture.sessionId)).budgetChars, 6000);
  assert.equal(await page.locator("#saveBudget").isDisabled(), true);
  await page.getByRole('button', { name: '关闭提示', exact: true }).click();
  await page.screenshot({ path: resolve(output, "memory-center-settings.png"), fullPage: true });
  await navigate("任务接续");
  await page.locator(".task-card").filter({ hasText: "把记忆工作台打磨成日常工具" }).getByRole("button", { name: "在此会话继续" }).click();
  await page.locator(".task-card").filter({ hasText: "把记忆工作台打磨成日常工具" }).locator('.bound').waitFor();
  await page.getByRole('button', { name: '关闭提示', exact: true }).click();

  await navigate('知识库');
  await page.getByRole('heading',{name:'我的记忆使用偏好',exact:true}).waitFor();
  await page.locator('.knowledge-entry').filter({hasText:'记忆工作台的交付进展'}).click();
  await page.getByRole('heading',{name:'记忆工作台的交付进展',exact:true}).waitFor();
  await page.locator('.knowledge-entry').first().click();
  await page.getByRole('button',{name:'来源 1',exact:true}).first().click();
  await page.locator('.evidence-reader').getByText(/虚构的演示证据/).waitFor();
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:resolve(output,'memory-center-knowledge.png'),fullPage:true});
  await navigate('知识图谱');
  await page.locator('.graph-node').first().waitFor();
  assert.equal(await page.locator('.graph-node').count(),6);
  assert.equal(await page.locator('.graph-edge').count(),5);
  await page.locator('.graph-node').nth(1).click();
  await page.getByRole('heading',{name:'聊天纠错先确认',exact:true}).waitFor();
  const viewBefore=await page.locator('svg.map').getAttribute('viewBox');
  await page.getByRole('button',{name:'放大图谱',exact:true}).click();
  assert.notEqual(await page.locator('svg.map').getAttribute('viewBox'),viewBefore);
  await page.getByRole('button',{name:'复位',exact:true}).click();
  await page.screenshot({path:resolve(output,'memory-center-graph.png'),fullPage:true});
  await navigate('模型配置');
  await page.getByRole('heading',{name:'把记忆系统放在这台电脑',exact:true}).waitFor();
  assert.equal(await page.locator('.local-model-card').count(),3);
  await page.getByText('CPU 写入和原文召回已测；复杂编译与后台整理尚待验收。',{exact:true}).waitFor();
  await page.locator('#writer-base').waitFor();
  assert.equal(await page.locator('#organizer-base').isVisible(),false);
  await page.locator('#writer-base').fill('https://provider.example/v1');
  await page.locator('#writer-model').fill('test-writer');
  await page.locator('#writer-key').fill('synthetic-browser-only-secret');
  await page.locator('#provider-inherit').uncheck();
  await page.locator('#organizer-base').fill('https://provider.example/v1');
  await page.locator('#organizer-model').fill('test-organizer');
  await page.locator('#organizer-key').fill('synthetic-organizer-only-secret');
  await page.locator('#saveProviders').click();
  await page.getByText('模型配置已保存到本机',{exact:true}).waitFor();
  assert.equal(await page.locator('#writer-key').inputValue(),'');
  assert.equal(await page.locator('#organizer-key').inputValue(),'');
  assert(!(await page.locator('body').innerText()).includes('synthetic-browser-only-secret'));
  await page.screenshot({path:resolve(output,'memory-center-providers.png'),fullPage:true});
  for (const width of [390, 768, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    for (const name of ["总览", "任务接续", "记忆来源", "写入记录", "会话设置", "知识库", "知识图谱", "模型配置"]) {
      await navigate(name);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, name + " overflows at " + width);
    }
    if (width === 390) {
      await navigate("总览");
      await page.screenshot({ path: resolve(output, "memory-center-mobile.png"), fullPage: true });
    }
  }
  assert(requests.every(url => url.startsWith(fixture.baseUrl + "/")), "The UI must not send requests to any external host");

  // A failed initial load has a usable retry path; ordinary page refresh is not
  // used, because the local token is deliberately removed from browser history.
  const errorPage = await browser.newPage();
  await errorPage.route("**/api/action", route => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ ok: false, error: "模拟服务暂时不可用" }) }));
  await errorPage.goto(fixture.url);
  await errorPage.getByRole("heading", { name: "暂时无法连接工作台" }).waitFor();
  await errorPage.unroute("**/api/action");
  await errorPage.getByRole("button", { name: "重新连接", exact: true }).click();
  await errorPage.getByRole("heading", { name: "记忆工作台", exact: true }).waitFor();
  await errorPage.close();

  await page.setViewportSize({ width: 1440, height: 1060 });
  await navigate("会话设置");
  await page.locator("#close").click();
  await page.locator("#editor").getByRole("button", { name: "关闭服务", exact: true }).click();
  await page.getByRole("heading", { name: "本机工作台已关闭" }).waitFor();
  await page.close();
  await fixture.dispose();
  fixture = await memoryCenterFixture({ empty: true });
  const emptyPage = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "reduce" });
  await emptyPage.goto(fixture.url);
  await emptyPage.getByRole("heading", { name: "从一项任务开始" }).waitFor();
  await emptyPage.screenshot({ path: resolve(output, "memory-center-empty.png"), fullPage: true });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, headlessUI: true, eightPages: true, knowledgeEvidence:true, graphSelectionAndZoom:true, localProviderSave:true, taskCRUD: true, correctionRetryIdempotent: true, feedbackRequiresEffective: true, modes: true, budget: true, search: true, sourceXSSSafe: true, responsive: [390,768,1280], connectionRecovery: true, closeService: true, emptyState: true, externalRequests: 0, screenshots: output }));
} finally {
  await browser?.close();
  await fixture?.dispose();
}

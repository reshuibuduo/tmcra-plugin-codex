# TMCRA Memory for Codex

TMCRA Memory 通过本机 Memory API 或 TMCRA 托管服务为 Codex 提供自动长期记忆。Windows 本地安装独立于 TMCRA 服务器和账号。此仓库是 [TMCRA 主仓库](https://github.com/reshuibuduo/tmcra/tree/main/07-tmcra-codex-plugins/tmcra-memory)中插件源码的独立发行镜像。

[English](README.md)

## 1.0.0-rc.1 新增记忆控制

- 任务接续：短句“继续”使用当前绑定目标、最近结果和下一步；多个候选任务会要求明确选择。
- 可操作控制台：调用 `tmcra_open_memory_center`，传当前 `session_id` 和 `project_path`，可以查看任务、来源、写入状态，执行纠错、忽略与恢复。
- 统一工作台：集成记忆写入 / 后台整理 API 配置、知识库和知识图谱，支持来源追溯与范围筛选；密钥只在本机表单输入，保存后不回显。
- 聊天纠错：识别真实纠错意图后先暂停当前回合自动写入，核对来源与新内容，再请求宿主聊天确认。拒绝、取消、过期和缺少确认能力均不提交。
- 会话开关：`normal` 正常读写、`recall_only` 仅召回、`off` 关闭；恢复后不会补写关闭期间的回合。已经提交的远端写入不受此开关撤回。
- 召回预算：默认每轮 12000 字符，按完整证据块裁选；只对本轮重复和仍在宿主上下文中的相同证据去重。Token 用量为估计值。
- 有效纠错：需要同步部署服务端反馈接口更新。旧来源的召回会立即受控，新内容独立可搜索需要等待返回的索引任务完成。

任务与会话控制状态按服务地址、凭据指纹和项目隔离。切换登录凭据会创建独立的本机任务状态；跨应用已提交的长期记忆继续通过同一服务端项目 scope 共享。`TMCRA_MEMORY_STATE_DIR` 可显式指定多个本机适配器共用的状态目录。API Key 不会写入这个状态目录。

新增功能和部署验收说明见 [memory-controls.md](docs/memory-controls.md)。

## 核心能力

- `SessionStart` 初始化全局、项目和会话边界。
- `UserPromptSubmit` 根据当前问题召回相关证据，并保留用户与 Agent 的来源区别。
- `PostToolUse` 只在达到阈值时形成经过脱敏的长任务检查点。
- `PreCompact`、`PostCompact` 保存并恢复上下文压缩前后的任务状态。
- `Stop` 写入真实完成的用户/assistant 回合；`StopFailure` 只保存用户请求检查点。
- `SubagentStart`、`SubagentStop` 隔离并记录子 Agent 生命周期。
- 内置 MCP 工具可查看上一轮实际使用的召回、主动召回、写入记忆和检查异步任务。

全部九项 Hook 都需要用户在 Codex 中明确审核和信任。

## 安装

**脱离 TMCRA 服务器：**下载 Release ZIP 并解压，双击 `Install-Local.cmd`。安装器自动注册插件、准备独立 Python、下载并校验模型、生成本机身份并启动记忆服务，无需 TMCRA 账号、API Key 或预装 Python。首次下载需要联网。完成后重启 Codex，并由你审核九项 Hook。当前支持 Windows x64；轻量档建议 16GB 内存，启动时约需 6.3GiB 空闲内存。

托管服务账号模式使用下面的安装方式。

从插件市场安装的用户，可直接说“打开 TMCRA 本地安装”，调用 `tmcra_open_local_install` 免账号打开同一工作台。选好档位并在页面确认后才开始下载安装。

从 [GitHub Releases](https://github.com/reshuibuduo/tmcra-plugin-codex/releases) 下载带版本号的 ZIP 和对应 SHA-256 文件。校验后解压到稳定目录。

Windows：

```powershell
.\Install-TMCRA.ps1
```

macOS / Linux：

```sh
sh ./install.sh
```

安装器会注册并启用插件，然后打开 TMCRA 设备授权页面。登录账号、确认短码并重启 Codex；随后运行 `/hooks`，审核九项 TMCRA Hook。访问 Token、设备码、PKCE 校验值和交付回执不会打印到终端。

## 记忆边界

- 稳定用户事实进入全局层，每个项目使用独立 scope，每个 Codex 任务作为项目内 session。
- 当前用户指令的优先级高于历史用户事实，历史 Agent 进度保持 assistant 来源。
- 自动 Hook 短超时并 fail-open；显式 MCP 操作使用独立的较长超时。
- 凭据、私钥、验证码、开发者指令和 chain-of-thought 不会进入记忆。
- 旧 Codex 历史与仓库基线只能先本地预览，再由用户明确确认导入。

## 本地 Writer 与后台整理模型

### 完整本地部署预览

工作台新增三档 embedding / reranker：E5-small + 多语言 MiniLM、BGE-M3 + BGE-reranker-v2-m3、Qwen3-Embedding-4B + Qwen3-Reranker-0.6B，以及安装、推荐、状态和启动/停止入口。

Release 和市场插件包均内置真实后端及 SHA-256 清单。使用 `Install-Local.cmd`，或在插件目录运行 `node scripts/local_setup.mjs`，即可免登录打开安装页。Python 与模型自动下载，身份自动生成并登记；重启宿主后 Codex、DSH 和通用 TMCRA MCP 自动发现本地连接。选择本地后，旧云端连接和后台云模型请求会被拦截；安装失败保留本地选择。原云端凭据保留在原位置。高级用户先清除显式 `TMCRA_CONFIG_FILE` 覆盖再进行自动安装。其他 MCP 宿主可使用[独立运行包](https://github.com/reshuibuduo/tmcra/releases/tag/v1.0.0-rc.1)。

完整验收仍在进行：CPU 合成写入 112 秒、原文召回 0.52 秒；复杂编译 600 秒超时，后台整理和完整重启恢复待测。轻量启动需约 6.3GiB 空闲内存。详见[验收记录](https://github.com/reshuibuduo/tmcra/blob/v1.0.0-rc.1/docs/LOCAL_DEPLOYMENT_PREVIEW.zh-CN.md)。宿主 Agent 使用云端主模型时，召回证据仍可能由宿主发往云端。

### 配置独立模型 API

在 Codex 中输入“打开 TMCRA 本地模型设置”即可打开配置页。MCP 返回结果不含 API Key 和页面会话令牌。配置页只监听 `127.0.0.1`，每次启动生成随机令牌；Writer 与后台整理可以共用模型，也可以分别填写 Provider、Base URL、模型名称和 API Key。

Codex 与 DeepSeek Harness 共用 `~/.config/tmcra/local-providers.json`。API Key 只保存在当前系统用户的本地文件中，测试连接时只发往用户填写的模型服务。Provider 或 Base URL 改变后必须重新填写 Key，已保存的 Key 不会转发到新地址。macOS/Linux 使用 `0600` 权限；Windows 移除继承 ACL，只授权当前用户与 SYSTEM。同一系统用户运行的其他进程仍可能读取该文件，因此这项功能适合可信的本地账号。

保存后的配置会在 Codex MCP 进程运行期间生效。写入请求会把 Writer 工作（包括强制慢阶段）路由到已认证的本机执行器；`tmcra_consolidate` 会用同一路径处理显式后台整理任务。执行器只领取由当前设备凭据创建的任务，调用用户配置的 OpenAI 兼容端点，并回传解析后的 JSON、标准化 Token 用量和服务商请求 ID。API Key 与服务商原始响应包保留在当前用户的本机进程中。

## 开发验证

```sh
npm ci
npm run verify
npm run build:release
```

确定性测试使用隔离的 mock 服务，不读取日常 Codex 历史，也不访问公开 TMCRA API。测试覆盖 Codex/Claude Code 生命周期、设备授权、项目隔离、可靠写入、敏感信息脱敏、MCP 工具和跨平台发布包。

## 许可

Apache-2.0。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

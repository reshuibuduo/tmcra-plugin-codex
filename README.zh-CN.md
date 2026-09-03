# TMCRA Memory for Codex

TMCRA Memory 通过公开 TMCRA API 为 Codex 提供自动长期记忆。普通用户无需服务器权限，也无需复制 API Key。此仓库是 [TMCRA 主仓库](https://github.com/reshuibuduo/tmcra/tree/main/07-tmcra-codex-plugins/tmcra-memory)中插件源码的独立发行镜像。

[English](README.md)

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

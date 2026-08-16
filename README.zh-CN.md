# TMCRA Codex 本地记忆插件

![TMCRA 本地记忆：跨软件记忆，连续工作](assets/overview.png)

TMCRA Local Memory 让 Codex 在不同会话和已支持的 Agent 工具之间接着做同一个项目。插件把 Codex 生命周期 Hook 接入用户电脑上的 [TMCRA Agent Memory](https://github.com/reshuibuduo/TMCRA-Agent-Memory) 运行时；插件自身只访问本机回环 API，本地模型或 BYOK 模型供应商的网络请求由用户选择的运行时配置决定。

[English](README.md)

## 功能

- **跨会话继续工作。** Codex 回答前，同时召回用户全局记忆和当前项目记忆。
- **跨软件共享项目进度。** 各接入工具依次通过 `.tmcra/project.json`、Git origin、Git 根目录或规范化目录路径识别同一项目。
- **区分信息主体。** 用户消息与 Codex 回答分别写入，并保留角色、来源软件和原始会话标识。
- **会话归属于项目。** Session 负责来源追踪与分组，不会成为脱离项目的第三个召回范围。
- **本地写入可重试。** 运行时暂时不可用时，记录进入仅当前用户可访问的 outbox，下一次生命周期事件继续提交。
- **不阻断 Codex。** Hook 失败时 Codex 继续运行；召回内容带有“不可信数据”边界；常见凭据会在召回和写入前脱敏。

## 工作流程

```text
本轮提问
  -> 同时召回用户全局记忆与当前项目记忆
  -> 将有来源标记、长度受限的证据注入 Codex
  -> Codex 回答
  -> 分别写入 USER 与 CODEX 两条记录
```

插件使用四个 Codex Hook：

| Hook | 作用 |
| --- | --- |
| `SessionStart` | 检查本地 TMCRA 运行时，并重试尚未完成的写入。 |
| `UserPromptSubmit` | 召回相关记忆、注入证据、保存脱敏后的用户提问。 |
| `Stop` | 把本轮可见回答作为独立的 assistant 记录写入。 |
| `StopFailure` | 清理本轮待处理状态，不会虚构一条 assistant 回答。 |

## 使用条件

- Node.js 18 或更新版本。
- 支持插件市场与 Hooks 的 Codex 版本。
- 已安装 [TMCRA 本地运行时](https://github.com/reshuibuduo/TMCRA-Agent-Memory)，并仅在回环地址上提供 API。

本仓库只发行 Codex 接入层。记忆引擎、本地 API、模型选择、存储和模型供应商配置位于 TMCRA 主仓库。

## 从 Awesome Codex Plugins 安装

```bash
codex plugin marketplace add \
  'https://github.com/hashgraph-online/awesome-codex-plugins.git' \
  --ref 'main' \
  --sparse '.agents/plugins' \
  --sparse 'plugins'
codex plugin install tmcra-local-memory --source awesome-codex-plugins
```

社区 Registry 会把本仓库镜像成可安装的插件包。参与源码开发时可以直接运行下方的验证命令，不需要注册第二个长期 marketplace。

## 配置本地运行时接入

先安装并启动 TMCRA，再由 TMCRA 主仓库写入接入配置，同时跳过其中用于开发的内置插件副本。

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-local.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex-local.ps1 -SkipPluginInstall
```

macOS 或 Linux：

```bash
./scripts/install-local.sh
TMCRA_SKIP_CODEX_PLUGIN_INSTALL=1 ./scripts/install-codex-local.sh
```

重启 Codex，打开 `/hooks`，逐项检查并信任四个 **TMCRA Local Memory** Hook。Codex 会把这一步明确交给用户确认。

需要使用非默认运行时的开发者，可以在本仓库执行：

```bash
node scripts/configure.mjs \
  --runtime-config /absolute/path/to/local-runtime.json \
  --base-url http://127.0.0.1:2009
```

生成的接入配置只保存 token 文件的绝对路径，不会复制或打印 bearer token。

## 安全边界

- API 地址仅允许 `localhost`、`127.0.0.1` 或 `::1`。
- 每次请求都会从仅当前用户可访问的文件读取本地 bearer token。
- 提问和回答会限制长度，并在进入召回或写入链路前脱敏常见凭据。
- 召回内容以“不可信证据”形式注入，不能充当可执行指令。
- 诊断日志只记录受限长度的错误名称与错误消息，不记录提问或回答。
- 本仓库不包含 TMCRA 生产服务代码和生产凭据。

漏洞报告方式见 [SECURITY.md](SECURITY.md)。

## 开发验证

```bash
npm ci
npm run verify
```

测试会启动真实的回环 HTTP fixture，核验跨软件项目身份、会话隔离、角色来源、脱敏、召回注入以及用户/assistant 分轨写回。

## 许可

Apache-2.0。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

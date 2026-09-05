# TMCRA Memory for Codex

## 独立本地安装 / Server-independent installation

Windows x64：解压后双击 `Install-Local.cmd`。自动安装 Python、依赖、校验后的模型和本机 Memory API，自动生成私有本地身份。无需 TMCRA 账号、服务器或预装 Python；首次下载需联网。完成后重启 Codex，并由用户审核九项 Hook。轻量档建议 16GB 内存且启动时约 6.3GiB 空闲。完整模型验收仍有限制，详见插件 README 的本地部署章节。

On Windows x64, double-click `Install-Local.cmd` after extracting the entire archive. Setup automatically prepares private Python, verified models and the Memory API, with local identity discovery. No TMCRA server, account or preinstalled Python is required. Internet is needed for first-time downloads. Restart Codex and personally review all nine Hooks. See the bundled README for resource requirements and partial model acceptance results.

## 托管服务 / Hosted service

保留解压后的 `.agents`、`plugins` 和根目录安装脚本，然后在解压目录中运行：

```powershell
.\Install-TMCRA.ps1
```

安装器会注册 `TMCRA Memory` 插件，打开 TMCRA 设备授权页并显示短码。登录后确认当前 Codex 设备即可；全程无需 SSH，也无需复制 API Key。

安装完成后：

1. 重启 Codex Desktop，在插件管理页确认 `TMCRA Memory` 已启用。
2. 在 Codex 中输入 `/hooks`，审核并信任全部九项 TMCRA 生命周期 Hook。Codex 要求用户明确确认，安装器无法代替用户授权。
3. 新建任务并完成一轮对话，再在 TMCRA Memory 桌面应用中验证真实运行状态。Ready 需要服务端观察到当前插件版本的 SessionStart、召回和写入事件。
4. 迁移旧对话时，先在桌面应用中选择项目并本地预览，再单独确认导入。推理、工具日志、开发者指令和疑似凭据不会上传。
5. 如需配置自有模型，在 Codex 中输入“打开 TMCRA 本地模型设置”。Provider API Key 只写入当前用户的本地配置文件，页面不会将明文返回给 Codex。

macOS / Linux：

```sh
sh ./install.sh
```

macOS / Linux 需要在 `PATH` 中提供 Codex CLI 与 Node.js 18 或更高版本。

---

Keep `.agents`, `plugins`, and the root installers together after extraction. Run `./Install-TMCRA.ps1` on Windows or `sh ./install.sh` on macOS/Linux. The installer registers `TMCRA Memory`, then opens TMCRA device authorization. Sign in and approve the short code; SSH access and copied API keys are not required.

Restart Codex Desktop, confirm `TMCRA Memory` is enabled in Plugins, run `/hooks`, and explicitly review all nine lifecycle hooks. Complete one turn in a new task, then verify real Codex execution in the TMCRA Memory app. Ready requires observed SessionStart, recall, and capture events from the current plugin version. Ask Codex to open the TMCRA local model settings when configuring user-supplied Writer or background-organizer providers; the keys remain in the local user configuration.

# TMCRA Memory for Codex

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

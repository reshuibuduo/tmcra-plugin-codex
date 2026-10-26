[CmdletBinding()]
param(
    [string]$ApiKey,
    [string]$BaseUrl = "https://api.tmcra.com",
    [string]$AuthorizationUrl = "https://tmcra.com",
    [string]$ScopeNamespace = "tmcra",
    [string]$NodePath,
    [switch]$SkipConfigure,
    [switch]$SkipPluginInstall,
    [switch]$NoBrowser,
    [switch]$ProgressJson,
    [switch]$ApiOnlyCheck,
    [switch]$LocalMemory,
    [ValidateSet('lite-cpu','balanced-bge','quality-qwen')][string]$LocalProfile = 'lite-cpu'
)

$ErrorActionPreference = "Stop"
$installer = Join-Path $PSScriptRoot "plugins\tmcra-memory\scripts\install.ps1"
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "TMCRA package is incomplete: plugins/tmcra-memory/scripts/install.ps1 is missing."
}

& $installer @PSBoundParameters
$installerExitCode = $LASTEXITCODE
if ($installerExitCode -ne 0) {
    exit $installerExitCode
}

# Run scripts/train_baselines.sh while asking Windows not to idle-sleep, then release.
# Start it detached so training survives closing the terminal or Claude session:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\launch_baselines.ps1','<dataset dir>'
param([Parameter(Mandatory = $true)][string]$DataDir)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'

$repo = Split-Path -Parent $PSScriptRoot
$bash = 'C:\Program Files\Git\bin\bash.exe'
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    Write-Host "Training baselines (computer kept awake until done). Logs: $repo\outputs\baseline"
    & $bash (Join-Path $repo 'scripts/train_baselines.sh') $DataDir
} finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

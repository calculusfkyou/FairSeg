# Run scripts/train_transunet.sh now, keeping Windows from idle-sleeping until it ends.
# An existing output folder from an earlier run is renamed (not deleted) first, because
# train.py appends to log.txt there and would mix the runs. Start it detached:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\launch_transunet.ps1','<dataset dir>','<tag>'
param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter(Mandatory = $true)][string]$Tag   # names the log file and runs.txt lines
)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'

$repo = Split-Path -Parent $PSScriptRoot
$bash = 'C:\Program Files\Git\bin\bash.exe'
$out = Join-Path $repo 'model\TU_FairSeg224\TU_pretrain_ViT-B_16_skip0_epo300_bs12_224'

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    if (Test-Path $out) {
        $stamp = (Get-Item $out).LastWriteTime.ToString('yyyyMMdd_HHmm')
        Rename-Item $out ("$(Split-Path $out -Leaf)_before_$Tag" + "_$stamp") -ErrorAction Stop
    }
    Write-Host "Training TransUNet ($Tag), computer kept awake until done. Logs: $repo\outputs\baseline"
    & $bash (Join-Path $repo 'scripts/train_transunet.sh') $DataDir $Tag
} finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

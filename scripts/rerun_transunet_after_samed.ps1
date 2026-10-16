# Wait for the running scripts/train_baselines.sh to finish SAMed, then rerun TransUNet
# (its first run lost the final checkpoint to a write error). Keeps Windows from
# idle-sleeping while waiting and training. Start it detached:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\rerun_transunet_after_samed.ps1','<dataset dir>'
param([Parameter(Mandatory = $true)][string]$DataDir)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'

$repo = Split-Path -Parent $PSScriptRoot
$runs = Join-Path $repo 'outputs\baseline\runs.txt'
$bash = 'C:\Program Files\Git\bin\bash.exe'
$old = Join-Path $repo 'model\TU_FairSeg224\TU_pretrain_ViT-B_16_skip0_epo300_bs12_224'

function Test-TrainingAlive {
    [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'train\.py' -and $_.CommandLine -match 'FairSeg_final' })
}

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    Write-Host "Waiting for SAMed to finish, then rerunning TransUNet. Logs: $repo\outputs\baseline"
    while (-not ((Test-Path $runs) -and ((Get-Content $runs -Raw) -match 'samed exit \d+'))) { Start-Sleep -Seconds 60 }
    while (Test-TrainingAlive) { Start-Sleep -Seconds 30 }   # let GPU memory be released

    # Keep the first run (epoch_199/249 are intact, epoch_299 is truncated) out of the way:
    # train.py appends to log.txt in this folder, which would mix the two runs.
    if (Test-Path $old) {
        Rename-Item $old ("$(Split-Path $old -Leaf)_run1_failed_save") -ErrorAction Stop
    }
    & $bash (Join-Path $repo 'scripts/train_transunet.sh') $DataDir 'transunet_rerun'
} finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

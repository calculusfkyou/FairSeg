# Run the formal DeepLabV3+ training (scripts/deeplabv3plus.sh ... train) now, keeping Windows
# from idle-sleeping until it ends. An existing outputs\deeplabv3plus\train_seed42 from an earlier
# run is renamed (not deleted) first, because train.py appends to log.txt there and overwrites
# epochs.csv and the checkpoints. Start it detached:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\launch_deeplabv3plus.ps1','<dataset dir>','deeplabv3plus_seed42'
# One GPU: with nn.DataParallel on 2 GPUs the main process leaked ~0.43 GB of system memory per
# epoch here, and the 11/23 run ran out of memory at epoch 152 (spec section 8).
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
$out = Join-Path $repo 'outputs\deeplabv3plus\train_seed42'

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    if (Test-Path $out) {
        $stamp = (Get-Item $out).LastWriteTime.ToString('yyyyMMdd_HHmm')
        Rename-Item $out ("$(Split-Path $out -Leaf)_before_$Tag" + "_$stamp") -ErrorAction Stop
    }
    Write-Host "Training DeepLabV3+ ($Tag), computer kept awake until done. Logs: $repo\outputs\deeplabv3plus"
    & $bash (Join-Path $repo 'scripts/deeplabv3plus.sh') $DataDir train $Tag `
        --list_dir lists/FairSeg_split --output_dir ../outputs/deeplabv3plus/train_seed42 `
        --max_epochs 300 --batch_size 24 --n_gpu 1 --base_lr 0.01 --seed 42
} finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

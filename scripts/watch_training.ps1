# 監控 scripts/train_baselines.sh 與 TransUNet 重跑的訓練進度：每分鐘更新一次進度，
# 並在 TransUNet 完成、SAMed 完成，或訓練意外中止時跳出置頂提示視窗。
# 啟動（獨立視窗，不依賴 Claude）：
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\watch_training.ps1'
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo 'outputs\baseline'
$runs = Join-Path $logs 'runs.txt'
# 每輪步數與總步數：TransUNet 8000/24 = 334 步 x 300 輪；SAMed 8000/16 = 500 步，第 70 輪停止
$models = @(
    @{ Key = 'transunet';       Name = 'TransUNet (第1次)'; Log = Join-Path $logs 'transunet_train.log';       PerEpoch = 334; Epochs = 300 },
    @{ Key = 'samed';           Name = 'SAMed';             Log = Join-Path $logs 'samed_train.log';           PerEpoch = 500; Epochs = 70 },
    # scripts/rerun_transunet_after_samed.ps1：SAMed 結束後自動重跑 TransUNet
    @{ Key = 'transunet_rerun'; Name = 'TransUNet (重跑)';  Log = Join-Path $logs 'transunet_rerun_train.log'; PerEpoch = 334; Epochs = 300 }
)
$lastKey = 'transunet_rerun'   # 這個模型結束後，監控就停止

function Show-Popup([string]$text, [string]$title, [int]$icon) {
    # 另開程序顯示，避免提示視窗擋住監控迴圈；0x40000 = 置頂
    $cmd = "(New-Object -ComObject WScript.Shell).Popup('$($text -replace "'", "''")', 0, '$title', $($icon + 0x40000)) | Out-Null"
    Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $cmd
}

function Get-LastIteration([string]$log) {
    if (-not (Test-Path $log)) { return $null }
    $line = Get-Content $log -Tail 200 -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '^iteration (\d+) : loss : ([\d.]+)' } | Select-Object -Last 1
    if ($line -and $line -match '^iteration (\d+) : loss : ([\d.]+)') {
        return @{ Iter = [int]$Matches[1]; Loss = [double]$Matches[2] }
    }
    return $null
}

function Test-TrainingAlive {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'train\.py' -and $_.CommandLine -match 'FairSeg_final' }
    return [bool]$p
}

$notified = @{}
$rate = @{}          # 每個模型第一次看到的 (時間, 步數)，用來估算速度
$deadChecks = 0
# 啟動前就已經結束的模型不再跳提示（例如重新開啟監控視窗時）
$startText = if (Test-Path $runs) { Get-Content $runs -Raw } else { '' }
foreach ($m in $models) { if ($startText -match "(?m)^$($m.Key) exit \d+") { $notified[$m.Key] = $true } }
Show-Popup "訓練監控已啟動。`n完成或中途停止時會再跳出提示。" 'FairSeg 訓練監控' 64

while ($true) {
    $runText = if (Test-Path $runs) { Get-Content $runs -Raw } else { '' }
    $now = Get-Date
    Clear-Host
    Write-Host "FairSeg baseline 訓練監控    更新時間 $($now.ToString('MM/dd HH:mm:ss'))（每分鐘更新）"
    Write-Host ('-' * 70)

    foreach ($m in $models) {
        $done = [regex]::Match($runText, "(?m)^$($m.Key) exit (\d+)\s+finished (.+)")
        if ($done.Success) {
            $code = $done.Groups[1].Value
            $status = if ($code -eq '0') { '已完成' } else { "出錯結束 (exit $code)" }
            Write-Host ("{0,-18} {1}  {2}" -f $m.Name, $status, $done.Groups[2].Value.Trim())
            if (-not $notified[$m.Key]) {
                $notified[$m.Key] = $true
                if ($code -eq '0') {
                    Show-Popup "$($m.Name) 訓練完成！`n$($done.Groups[2].Value.Trim())" 'FairSeg 訓練監控' 64
                } else {
                    Show-Popup "$($m.Name) 訓練出錯結束 (exit $code)。`n請查看 outputs\baseline\$($m.Key)_train.log" 'FairSeg 訓練監控' 16
                }
            }
            continue
        }
        $it = Get-LastIteration $m.Log
        if (-not $it) { Write-Host ("{0,-18} 尚未開始" -f $m.Name); continue }
        $total = $m.PerEpoch * $m.Epochs
        $epoch = [math]::Ceiling($it.Iter / $m.PerEpoch)
        $pct = 100.0 * $it.Iter / $total
        if (-not $rate[$m.Key]) { $rate[$m.Key] = @{ T = $now; I = $it.Iter } }
        $eta = '計算中（約 2 分鐘後顯示）'
        $dt = ($now - $rate[$m.Key].T).TotalSeconds
        $di = $it.Iter - $rate[$m.Key].I
        if ($dt -gt 90 -and $di -gt 0) {
            $finish = $now.AddSeconds(($total - $it.Iter) * $dt / $di)
            $eta = "預計 $($finish.ToString('MM/dd HH:mm')) 完成"
        }
        Write-Host ("{0,-18} 第 {1}/{2} 輪  ({3:N1}%)  loss {4:N4}  {5}" -f $m.Name, $epoch, $m.Epochs, $pct, $it.Loss, $eta)
    }

    $allDone = $runText -match "(?m)^$lastKey exit \d+"
    if ($allDone) {
        Write-Host ('-' * 70)
        Write-Host '全部訓練已結束。可以關閉此視窗。'
        break
    }

    # 訓練程序消失、卻沒有寫入結束紀錄 → 意外中止（連續 3 分鐘確認，避開兩個模型交接的空檔）
    if (Test-TrainingAlive) { $deadChecks = 0 } else { $deadChecks++ }
    if ($deadChecks -ge 3 -and -not $notified['crash']) {
        $notified['crash'] = $true
        Show-Popup "訓練程序已停止，但沒有正常結束的紀錄。`n可能是當機或被關閉，請查看 outputs\baseline 的 log。" 'FairSeg 訓練監控' 48
    }
    if ($notified['crash']) {
        Write-Host ('-' * 70)
        Write-Host '警告：訓練程序已停止，但沒有正常結束的紀錄！'
    }
    Write-Host ('-' * 70)
    Write-Host '關閉此視窗只會停止監控，不會影響訓練。'
    Start-Sleep -Seconds 60
}

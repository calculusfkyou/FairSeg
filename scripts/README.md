# scripts/

重現 FairSeg 論文 SAMed 與 TransUNet baseline 所用的腳本：資料檢查、訓練、測試，以及結果整理。

- 所有指令都從 **repo 根目錄**執行。
- Python 腳本使用 `.venv\Scripts\python.exe`，套件版本見 `requirements-lock.txt`。
- `.sh` 用 Git Bash 執行，`.ps1` 用 Windows PowerShell 執行。
- 以下的 `<DATA>` 指 HF 資料集的 `Dataset` 資料夾，也就是裡面有 `Training/`、`Test/` 的那一層。本實驗為 `C:/Users/user/source/repo/master-thesis-fair-seg/data/fairseg_full/Dataset`。程式會往下搜尋所有 npz，再依官方 `lists/FairSeg_final` 決定哪些是訓練集、哪些是測試集。

整體流程與結果見 `docs/baseline_reproduction.md`。

## 總覽

| 階段 | 腳本 | 用途 |
|---|---|---|
| 資料檢查 | `visualize_samples.py` | mask 疊圖，人工確認前處理 |
| | `scan_crop_coverage.py` | 掃描全部 mask，確認中央裁切沒有切掉視盤 |
| 試訓練 | `benchmark_batch.py` | 測量不同 batch size 的顯存與速度 |
| 訓練 | `train_baselines.sh` | 依序訓練 TransUNet 與 SAMed |
| | `train_transunet.sh` | 只訓練 TransUNet |
| | `launch_baselines.ps1`、`launch_transunet.ps1` | 在背景啟動上面兩個腳本，並防止電腦睡眠 |
| | `rerun_transunet_after_samed.ps1` | 等 SAMed 結束後自動重跑 TransUNet（一次性，已用過） |
| | `watch_training.ps1` | 訓練進度視窗與完成提示 |
| 測試 | `test_baselines.sh` | 用 4 個屬性測試兩個模型 |
| 結果 | `summarize_tests.py` | 把測試 log 整理成表格，並與論文比較 |
| | `paper_baselines.csv` | 論文 Table 2–5 的 baseline 數值 |

## 資料檢查

### `visualize_samples.py`

把 mask 疊在影像上，依序顯示每個前處理階段：原圖、中央裁切 512、SAMed 輸入 512、SAMed loss 目標 128、TransUNet 輸入 224、TransUNet 訓練時的資料增強。使用的是兩個模型實際的 transform。綠色為 rim，紅色為 cup。

```bash
.venv/Scripts/python.exe scripts/visualize_samples.py --root_path <DATA>
.venv/Scripts/python.exe scripts/visualize_samples.py --root_path <DATA> --files data_00113.npz data_06252.npz
```

預設從訓練集與測試集各隨機挑 3 張。輸出到 `outputs/data_check/overlay.png`。

### `scan_crop_coverage.py`

逐一讀取全部 npz 的 mask，記錄每張影像的尺寸、視盤與 cup 的像素數、在 512×512 中央裁切內的像素數，以及視盤的範圍。

```bash
.venv/Scripts/python.exe scripts/scan_crop_coverage.py --root_path <DATA>
```

輸出到 `outputs/data_check/crop_coverage.csv`。讀取約 45 GB，需要數分鐘。本實驗的結果：10,000 張都沒有空 mask，視盤也都沒有被裁切到。

## 試訓練

### `benchmark_batch.py`

用隨機資料測量每個 batch size 的顯存峰值、每一步的時間，以及每秒可處理的影像數。加上 `--loader` 會再測量官方 DataLoader 讀取資料的速度。

```bash
.venv/Scripts/python.exe scripts/benchmark_batch.py --model samed --batch_sizes 4 8
.venv/Scripts/python.exe scripts/benchmark_batch.py --model samed --n_gpu 2 --batch_sizes 16
.venv/Scripts/python.exe scripts/benchmark_batch.py --model transunet --batch_sizes 24 42 --loader --root_path <DATA>
```

在 Windows 上，顯存不夠時驅動程式不會報錯，而是借用系統記憶體，速度會變得非常慢。顯存用量接近上限時，輸出會加上提示。

## 訓練

訓練設定與官方程式（`train.sh`）的差異都寫在各腳本的註解中，完整比較見 `docs/baseline_reproduction.md` 第 3 節。

### `train_baselines.sh`

依序訓練 TransUNet 與 SAMed，每個模型只訓練一次。baseline 訓練不使用敏感屬性，所以一個模型可以用來測試全部四個屬性。

```bash
bash scripts/train_baselines.sh <DATA>
```

| 模型 | 設定 |
|---|---|
| TransUNet | 2 張 GPU × 12 = batch 24，lr 0.01，SGD，300 epochs |
| SAMed | 2 張 GPU × 8 = batch 16，lr 0.005，warmup，AdamW，最多 100 epochs，第 70 epoch 停止 |

log 寫到 `outputs/baseline/<model>_train.log`，開始與結束時間寫到 `outputs/baseline/runs.txt`。TransUNet 的模型存到 `model/`，SAMed 的模型存到 `outputs/baseline/samed/`。

### `train_transunet.sh`

只訓練 TransUNet，設定與 `train_baselines.sh` 相同。第二個參數是標籤，用來命名 log 檔與 `runs.txt` 中的紀錄。

```bash
bash scripts/train_transunet.sh <DATA> transunet_run3
```

### `launch_baselines.ps1`、`launch_transunet.ps1`

在另一個程序中執行上面的 bash 腳本，訓練期間防止 Windows 閒置睡眠，結束後自動解除，不會修改電源設定。用 `Start-Process` 啟動，關閉終端機或 Claude 都不會中斷訓練：

```bash
Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\launch_transunet.ps1','<DATA>','<標籤>'
```

`launch_transunet.ps1` 開始前，會把上一次的 TransUNet 輸出資料夾改名保留（加上 `_before_<標籤>_<時間>`）。這是因為 `train.py` 會把 log 接在舊檔案後面，不改名的話新舊紀錄會混在一起。

防睡眠無法阻止闔上筆電螢幕、手動睡眠，或 Windows Update 自動重新開機。

### `rerun_transunet_after_samed.ps1`

等 `train_baselines.sh` 的 SAMed 結束後，自動以標籤 `transunet_rerun` 重跑 TransUNet。本實驗用過一次（第 2 次 TransUNet 訓練），之後不需要再用。

### `watch_training.ps1`

每分鐘更新一次的訓練進度視窗，顯示每個模型目前的 epoch、loss 和預計完成時間。訓練完成、出錯，或訓練程序意外消失時，會跳出置頂提示視窗。

```bash
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\watch_training.ps1'
```

要監控的訓練寫在腳本開頭的 `$models`，以 `runs.txt` 中的標籤區分。新增其他訓練時，要在這裡加上一筆。關掉這個視窗不會影響訓練。

## 測試

### `test_baselines.sh`

用官方 `test.txt`（2,000 張）測試兩個最終模型，依序測試 race、gender、language、ethnicity 四個屬性。SAMed 使用 GPU 0，TransUNet 同時使用 GPU 1，全部約需 40 分鐘。

```bash
bash scripts/test_baselines.sh <DATA>
```

| 模型 | 測試用的模型檔 |
|---|---|
| SAMed | `outputs/baseline/samed/FairSeg_512_pretrain_vit_b_epo100_bs8_lr0.005/epoch_69.pth` |
| TransUNet | `model/TU_FairSeg224/TU_pretrain_ViT-B_16_skip0_epo300_bs12_224/epoch_299.pth` |

兩個都是最後一個 epoch 的模型，沒有用測試集挑選。結果寫到 `outputs/baseline_test/`。

## 結果整理

### `summarize_tests.py`

讀取 `test_baselines.sh` 產生的 8 個測試 log，以及 `paper_baselines.csv`，輸出三個檔案到 `outputs/baseline_test/`：

| 檔案 | 內容 |
|---|---|
| `results.csv` | 每一個數值：本實驗、論文、差距 |
| `comparison.md` | 依屬性分開的本實驗 vs 論文對照表 |
| `paper_tables.md` | 論文 Table 2–5 的格式，每一列論文數值下方加上本實驗的數值 |

```bash
.venv/Scripts/python.exe scripts/summarize_tests.py
```

`docs/baseline_reproduction.md` 的四張表就是 `paper_tables.md` 的內容。

族群名稱依照 HF 資料集的編碼（例如 gender 0 = Female、1 = Male）。論文的 gender 與 ethnicity 欄位疑似對調，詳見 `docs/baseline_reproduction.md`。

### `paper_baselines.csv`

論文 Table 2–5 中 SAMed 與 TransUNet baseline 的所有數值，數字依論文原文，族群名稱也依論文原表的標示。欄位如下：

| 欄位 | 內容 |
|---|---|
| `attribute` | race、gender、language、ethnicity |
| `model` | SAMed、TransUNet |
| `region` | Cup、Rim |
| `metric` | ES-Dice、Dice、ES-IoU、IoU |
| `group` | 空白表示整體分數，否則為族群名稱 |
| `value` | 數值 |

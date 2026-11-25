# outputs/

訓練 log、SAMed 模型、測試結果、資料檢查結果，以及第 ⑤ 步試訓練的產物。

除了這份 README，資料夾內的檔案都不在 git 中（見 `.gitignore`），需要另外備份。**最需要保留的是 `baseline/`、`baseline_test/` 與 `deeplabv3plus/`。**

TransUNet 的模型檔不在這裡，而是在 `model/`（官方程式寫死的路徑），詳見 `model/README.md`。

## 結構

```
outputs/
├── baseline/            ← ★ 正式訓練：log 與 SAMed 模型
├── baseline_test/       ← ★ 正式測試：log 與結果表格
├── data_check/          資料檢查（第 ④ 步）
├── deeplabv3plus/       ← ★ DeepLabV3+ 實驗（ethnicity、I-score + BDA）：訓練、BDA、test
├── deeplabv3plus_smoke/ DeepLabV3+ 小資料試跑，可刪除
├── lists_tiny/          16 張影像的小型清單（第 ⑤ 步）
├── lists_tiny_dlv3/     DeepLabV3+ 試跑用的小型 train／val／test 清單
├── lists_tiny_trainset/ 同上，但把訓練影像當作測試集
└── smoke/               第 ⑤ 步試訓練與各種測試，可刪除
```

## `deeplabv3plus/`：DeepLabV3+ 實驗

由 `scripts/deeplabv3plus.sh`（訓練透過 `scripts/launch_deeplabv3plus.ps1`）產生，設定見 `DeepLabv3+/docs/spec.md`，結果整理在 `DeepLabv3+/docs/results_ethnicity.md`。全部是正式結果，約 1.1 GB（大部分是 7 個 checkpoint）。

| 檔案 | 內容 |
|---|---|
| `runs.txt` | 每次執行的 commit、指令、開始與結束時間、exit code。前兩次訓練（11/23 藍屏、11/24 04:11 記憶體用盡）的紀錄也在這裡，正式訓練是 11/24 10:41 開始、`exit 0` 的那一次 |
| `<標籤>.log` | 每次執行的完整輸出。`deeplabv3plus_seed42.log` 接著放了 11/23 22:24 中斷那次與正式那次的輸出，以 `started` 行分隔 |
| `train_seed42/` | **正式訓練**：`epochs.csv`（每個 epoch 的 loss 與 val 指標）、`best_dice.pth`（①）、`best_es_<屬性>.pth`（②，六個）、`args.json`、`log.txt` |
| `bda_dice/`、`bda_es_ethnicity/` | 對 ① 與 ②-ethnicity 的 I-score + BDA：`medians.npy`、`bda_runs.json`、`ranked.json`、`union_val.csv`、`selection.json`（選出的 K 與 X*，綁定 checkpoint 的 SHA256） |
| `test/` | 10 次 test 評估的 JSON：`best_dice.json`、`best_es_<屬性>.json`，以及遮罩後的 `bda_dice_dice.json`、`bda_dice_es.json`、`bda_es_ethnicity_dice.json` |

`test.py` 不會覆寫已存在的 JSON，`bda.py` 也不會寫進已有 `args.json` 的資料夾，要重跑請先改名或指定新的輸出位置。

## `deeplabv3plus_smoke/`、`lists_tiny_dlv3/`：DeepLabV3+ 試跑（可刪除）

`lists_tiny_dlv3/` 是 16 張影像的 train／val／test 清單；`deeplabv3plus_smoke/` 是用它跑完訓練、BDA、test 整條流程的產物（spec §13），約 1.1 GB。分數**不能用於任何比較或報告**。

## `baseline/`：正式訓練

由 `scripts/train_baselines.sh`、`scripts/train_transunet.sh` 產生。

| 檔案 | 內容 |
|---|---|
| `runs.txt` | 每次訓練的 commit、開始與結束時間、exit code |
| `samed_train.log` | SAMed 訓練的完整輸出（70 epochs） |
| `transunet_run3_train.log` | TransUNet **第 3 次（正式）**訓練的完整輸出 |
| `transunet_train.log` | TransUNet 第 1 次訓練：最後存檔失敗 |
| `transunet_rerun_train.log` | TransUNet 第 2 次訓練：第 250 epoch 存檔失敗 |
| `transunet_run3_code.diff` | 第 3 次訓練開始時尚未 commit 的程式差異（之後已 commit 為 `d697bbd`） |
| `samed/FairSeg_512_pretrain_vit_b_epo100_bs8_lr0.005/` | SAMed 模型與 log，見下表 |

`runs.txt` 中第 3 次 TransUNet 訓練沒有 exit code，原因是啟動腳本在 Python 結束後卡住，只能手動停止。訓練本身已完成，三個模型檔都驗證過可以完整載入，所以補寫了一行說明。

### SAMed 模型：`samed/FairSeg_512_pretrain_vit_b_epo100_bs8_lr0.005/`

資料夾名稱由 `SAMed/train.py` 依設定產生：`512` 為輸入大小、`epo100` 為最大 epochs、`bs8` 為每張 GPU 的 batch size、`lr0.005` 為學習率。實際在第 70 epoch 停止。

| 檔案 | 說明 |
|---|---|
| `epoch_69.pth` | **最終模型，第 ⑦ 步的測試與報告都使用這個檔案** |
| `epoch_9.pth` … `epoch_59.pth` | 每 10 個 epoch 的備份 |
| `config.txt` | 訓練參數 |
| `log.txt`、`log/` | loss 紀錄與 TensorBoard 紀錄 |

每個模型檔約 16 MB，只包含 LoRA、prompt encoder 與 mask decoder 的權重。載入時需要搭配 SAM 預訓練權重 `SAMed/checkpoints/sam_vit_b_01ec64.pth`。

## `baseline_test/`：正式測試

由 `scripts/test_baselines.sh` 與 `scripts/summarize_tests.py` 產生。測試集為官方 `test.txt`（2,000 張）。

| 檔案 | 內容 |
|---|---|
| `paper_tables.md` | **論文 Table 2–5 格式的對照表**，`docs/baseline_reproduction.md` 的表格就是這份 |
| `comparison.md` | 依屬性分開，逐項列出本實驗、論文與差距 |
| `results.csv` | 所有數值（本實驗、論文、差距），方便用其他工具分析 |
| `runs.txt` | 每個測試的 exit code 與結束時間 |
| `test_code.diff` | 測試開始時尚未 commit 的程式差異（之後已 commit 為 `77e0349`） |
| `samed_<屬性>.log`、`transunet_<屬性>.log` | 每個測試的完整輸出 |
| `samed/epoch_69_<屬性>/test_log/log.txt` | SAMed 測試結果（官方 `test.py` 的格式） |
| `transunet/TU_pretrain_ViT-B_16_skip0_epo300_bs12_224_epoch_299/<屬性>.txt` | TransUNet 測試結果 |

`<屬性>` 為 race、gender、language、ethnicity。每份結果包含 Cup 與 Rim 的整體 Dice、IoU、HD95、ES-Dice、ES-IoU，以及各族群的分數。

重新執行 `summarize_tests.py` 會讀取最新的測試 log，重新產生三個表格檔案。

## `data_check/`：資料檢查（第 ④ 步）

| 檔案 | 產生方式 | 內容 |
|---|---|---|
| `overlay.png` | `scripts/visualize_samples.py` | 6 張影像在各前處理階段的 mask 疊圖 |
| `crop_coverage.csv` | `scripts/scan_crop_coverage.py` | 10,000 張 mask 的尺寸、視盤與 cup 像素數、裁切後保留的像素數、視盤範圍 |

## `lists_tiny/`、`lists_tiny_trainset/`

格式與官方 `lists/FairSeg_final` 相同的小型清單，用於第 ⑤ 步試訓練。用 `--list_dir` 指定即可使用。

| 資料夾 | 內容 |
|---|---|
| `lists_tiny/` | `train.txt`、`test.txt`：取官方清單的前 16 筆 |
| `lists_tiny_trainset/` | `test.txt` 就是上面的 16 張訓練影像，用來測試模型在訓練資料上的分數 |

## `smoke/`：試訓練與各種測試（可刪除）

第 ⑤ 步試訓練，以及之後修改程式時的驗證，不屬於任何正式結果。全部刪除約可釋放 216 MB。

| 項目 | 用途 |
|---|---|
| `samed/`、`samed_tiny.log` | SAMed 16 張影像試訓練，沒有 warmup，loss 卡在 0.53 |
| `samed_warmup/`、`samed_tiny_warmup.log` | 同上，加上 warmup，loss 正常下降 |
| `samed_lr1e-3/`、`samed_tiny_lr1e-3.log` | 同上，不加 warmup，改用 lr 0.001，loss 正常下降 |
| `samed_test_lists_tiny*/` | 上面模型在 16 張影像上的測試 |
| `transunet_tiny.log` | TransUNet 16 張影像試訓練（模型在 `model/..._epo40_bs4_224/`） |
| `samed_full_epoch/`、`samed_full_epoch.log` | SAMed 用完整資料跑 1 epoch，測量時間 |
| `samed_full_epoch_test/`、`samed_full_epoch_test.log` | 上面模型在完整測試集上的測試，用來確認測試流程 |
| `transunet_full_epoch.log` | TransUNet 用完整資料跑 1 epoch |
| `benchmark.txt`、`benchmark2.txt` | `scripts/benchmark_batch.py` 的顯存與速度測量結果 |
| `transunet_savetest.log` | 驗證新的存檔方式 |
| `transunet_test_check/`、`transunet_test_nii/` | 驗證 `TransUNet/test.py` 改為依屬性分開存檔 |
| `transunet_run3_tiny/` | 驗證 `test.py` 能讀取雙 GPU 訓練的模型 |
| `transunet_test_old/` | `TransUNet/test.py` 修改前的測試結果（原本在 `TransUNet/results_iclr_rebuttal/`） |

這裡的分數都來自只用 16 張影像或只訓練 1 epoch 的模型，**不能用於任何比較或報告**。

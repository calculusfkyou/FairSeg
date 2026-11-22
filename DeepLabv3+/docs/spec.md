# DeepLabV3+ 公平性實驗規格：ethnicity、I-score + BDA

- 日期：2026-11-23
- 狀態：草稿，待使用者確認
- 依據：`碩論相關/research.md` 第 3 節「下一階段實驗設定」（11/20 meeting），以及 11/21–11/23 與使用者逐項確認的決定
- 實作：使用者在 `FairSeg/DeepLabv3+/` 重新撰寫程式，不沿用 `master-thesis-fair-seg` 的程式碼

本文件標記說明：
- 沒有標記的設定：使用者或 11/20 meeting 已決定。
- **【我的建議】**：使用者沒有明確決定、由我提出的預設值，使用者可以改。
- **【待確認】**：還需要使用者決定的事。
- **【交給實作】**：不影響實驗結果，由寫程式的人決定。

---

## 1. 目的

最終目的是改善公平性（11/20 教授強調）。這一輪要回答：

1. 在官方切分（加上驗證集）下，DeepLabV3+／ResNet-50 baseline 的分割表現和公平性如何，和 FairSeg 論文的 SAMed／TransUNet 表格相比如何。
2. 以 I-score + BDA 找出與 ethnicity 相關的 layer4 channels 並遮罩後，ethnicity 的公平性（ES-Dice）是否改善、分割表現（Dice）掉多少。

上一次 experiment_001 只遮罩 1 個 channel，指標只變動約 1e-5，看不出效果；而且它用的 test set 和官方不同，不能和 SAMed／TransUNet 比較。這一輪改用官方 test set、加入驗證集，並改用「前 K 名 subset 取聯集」來決定遮罩。

## 2. 目標、非目標、反目標

**目標**

- 用官方 8,000 張 train 切出 7,000／1,000 的 train／val，官方 2,000 張 test 不動。
- 訓練一個 DeepLabV3+ baseline，在驗證集上選出 checkpoint ①（Dice）和六個 checkpoint ②（每個屬性的 ES-Dice）。
- 在 checkpoint ① 與 ②-ethnicity 上各做一次 I-score + BDA，用驗證集分別依 Dice、ES-Dice 選 K，得到 4 組 X*，各在 test 上評估一次。
- 評估流程和 FairSeg 官方 TransUNet 相同，數字可以和論文表格及本 repo 的重現結果並列。並列時要註明訓練條件不同：DeepLabV3+ 用 7,000 張訓練、在 val 上選 checkpoint、loss 不同；SAMed／TransUNet 用 8,000 張、取最後一個 epoch。

**非目標（這一輪不做）**

- ethnicity 以外屬性的 BDA；多類別屬性的 one-vs-rest 細節。
- 偏差分類器、保留「重要且與敏感屬性無關的 channels」。
- 遮罩後 fine-tune。
- 調整 ASPP rates（雖然 224 輸入下 rate 18 會退化成 1×1，這一輪仍維持 6／12／18）。
- 多個 training seed。
- 移植 GroupDRO、ADV、FEBS。
- 處理論文表格疑似欄位對調的問題；age、marital status 和 SAMed／TransUNet 的比較（需要改 FairSeg 的 test 程式）。
- 視覺化與解釋性。

**反目標（就算程式能跑，出現這些就算失敗）**

- Test set 參與任何選擇：選 checkpoint、算 median、I-score、BDA、選 K，全部只能用驗證集。
- 每組 X* 在 test 上跑超過一次，或看了 test 結果再回頭調整。
- 評估方式和 FairSeg 官方不同，導致數字不能和論文表格並列。
- 正式執行時程式有未 commit 的修改，或沒有記錄 commit。

## 3. 限制與環境

- 硬體：雙卡 RTX 3060 12 GB，Windows。
- Python 環境：`FairSeg/.venv`（Python 3.12，torch 2.14.1+cu126，見 `requirements-lock.txt`）。
- 資料：HF `harvardairobotics/FairSeg`，本機在 `master-thesis-fair-seg/data/fairseg_full/Dataset`（含 `Training/`、`Test/`）；`fairseg_full/ReadMe/data_summary.csv`（和 `Dataset/` 同層）有每個檔案的屬性名稱。
- 程式位置：`FairSeg/DeepLabv3+/`，和 `SAMed/`、`TransUNet/` 一樣以該資料夾為 cwd 執行。資料夾名稱含 `+`，不能當 Python 套件匯入，只能在資料夾內直接執行腳本。
- Windows 已知問題（見 repo 的 `CLAUDE.md`）：
  - DataLoader 的 `worker_init_fn` 要是模組層級函式；開 `persistent_workers=True`。
  - 存 checkpoint 不要直接 `torch.save(obj, path)`，改用 `TransUNet/trainer.py` 的 `save_checkpoint` 寫法（先序列化到記憶體、寫檔、讀回驗證、再改名）。
  - 顯存不足時 Windows 會改用系統記憶體而不報錯，只會變得很慢。改 batch size 前先量測（`scripts/benchmark_batch.py` 目前只支援 SAMed／TransUNet，需要擴充或另寫）。
  - 正式執行前先 commit，並在 `runs.txt` 記錄 commit 與是否有未 commit 的修改。

## 4. 考慮過的做法

### 4.1 程式來源

| 做法 | 取捨 |
|---|---|
| **在 `FairSeg/DeepLabv3+/` 重寫（採用）** | 和 `SAMed/`、`TransUNet/` 並列，資料讀取與評估可以照 FairSeg 官方流程；成本是要重寫模型與 I-score／BDA |
| 沿用 `master-thesis-fair-seg` 的 experiment_001 程式 | 模型與 BDA 現成，但它的切分、評估方式都和 FairSeg 官方不同，該 repo 也還在重構中 |
| 用外部套件（如 segmentation_models_pytorch） | 模型現成，但要新增相依套件，且遮罩位置要自己掛 hook |

使用者選擇重寫。模型結構仍沿用 experiment_001 的設計（第 7 節），只是程式重新寫。

### 4.2 X* 怎麼來

| 做法 | 取捨 |
|---|---|
| 只取 I-score 最高的單一 subset（experiment_001） | 簡單，但結果只有 1 個 channel，遮罩後幾乎沒有變化 |
| 固定取前 K 名的聯集 | K 沒有依據 |
| **依 I-score 排序、去重後，K 從 1 試到最後一名，每個不同的聯集都在驗證集上評估，再用驗證集指標選 K（採用）** | 有依據且只用驗證集；成本是要評估很多次，用快取特徵降低成本（第 10.6 節） |

## 5. 資料

### 5.1 欄位與編碼（HF npz）

| 鍵 | 內容 |
|---|---|
| `slo_fundus` | 灰階 SLO 影像，664×798，uint8 |
| `disc_cup_mask` | 0 = 背景、-1 = rim、-2 = cup；轉成訓練標籤 0／1／2 |
| `gender` | 0 Female、1 Male |
| `race` | 0 Asian、1 Black、2 White |
| `ethnicity` | 0 Non-Hispanic、1 Hispanic、-1 Unknown |
| `language` | 0 English、1 Spanish、2 Others、-1 Unknown |
| `maritalstatus` | 0 Married or partnered、1 Single、2 Divorced、3 Widowed、4 Legally separated、-1 Unknown |
| `age` | 年齡（年，浮點數），沒有 Unknown |

npz 沒有 `pid`，用檔名（去掉副檔名）代替。官方 TransUNet 的 `read_attr` 會把 `hispanic` 視為 `ethnicity` 的別名，新程式直接讀 `ethnicity` 即可。

### 5.2 Age 與 marital status 分組

照 FairSeg 官方 `read_attr`（`TransUNet/datasets/dataset_fairseg.py`）的分組：

- **age**：< 60 歲、≥ 60 歲，兩組。
- **maritalstatus**：Married or partnered（編碼 0）、其他（編碼 1–4 合併為 Not married），兩組；Unknown（-1）維持 Unknown。

這兩個分組同時用於分層切分、驗證集的 ES-Dice 與 test 評估。

### 5.3 切分

- **輸入**：官方 `lists/FairSeg_final/train.txt`（8,000 張）。`test.txt`（2,000 張）直接當 test，不參與切分。
- **輸出**：`train.txt`（7,000）、`val.txt`（1,000）、`test.txt`（官方 2,000 的複本），格式和官方清單相同（每行一個檔名）。產生一次後 commit，之後的訓練與評估都讀這三份清單。
- **分層鍵**：六個屬性的組合 (gender, race, ethnicity, language, maritalstatus 二元組, age 二元組)，分組見第 5.2 節。**【我的建議】** Unknown 當成獨立的類別參與分層；比例檢查仍不含 Unknown，符合 11/20「分層不看 Unknown」的要求。
- **分配方式**：每一層的 val 名額 = 該層張數 × 1,000／8,000，先取整數部分，剩下的名額依小數部分由大到小分配（同分時隨機）；每一層內再隨機抽出 val 名額。隨機種子 42。
- **檢查**：
  - train 7,000、val 1,000，彼此不重疊，聯集等於官方 `train.txt`，val 和 `test.txt` 不重疊。
  - 每個屬性在 train 和 val 的族群比例（不含 Unknown）差距要列表報告。11/23 用這個做法模擬（seed 0–19），各屬性的最大差距都不超過 0.5 個百分點；val 的 Hispanic 在 33–35 張之間（seed 42 為 34 張）。
- 六個屬性組合起來約 193 層，其中約 53 層只有 1 張。這是預期的，不影響各屬性的比例。

### 5.4 前處理與資料增強

完全照 FairSeg 官方 TransUNet 的 `RandomGenerator`／`TestGenerator`（`TransUNet/datasets/dataset_fairseg.py`）：

1. 影像 clip 到 [0, 255]，除以 255。
2. 影像與標籤都從中央裁切 512×512。
3. 訓練時：50% 機率做 `random_rot_flip`（隨機旋轉 90° 的倍數，再沿隨機一軸翻轉）；否則再以 50% 機率做 `random_rotate`（角度 `np.random.randint(-20, 20)`，即 −20 到 19，影像與標籤都用最近鄰，`reshape=False`）。驗證與測試不做增強。
4. 縮放到 224×224：影像用 `scipy.ndimage.zoom(order=3)`，標籤用 `order=0`。
5. 灰階複製成 3 個 channel。
6. **【我的建議】** 再用 ImageNet 的 mean／std 正規化（torchvision 預訓練權重的慣例；experiment_001 也是這樣做）。官方 TransUNet 沒有這一步。

## 6. 評估方式（驗證與測試共用）

照 FairSeg 官方 TransUNet（`TransUNet/utils.py` 的 `test_single_image`、`calculate_metric_percase`，`TransUNet/inference.py` 的彙總）：

- 在 224×224 上評估：預測取 softmax 後 argmax，和縮放到 224 的標籤比較。batch size 1 或逐張計算皆可，結果要相同。
- 每張影像分別計算 rim（標籤 1）與 cup（標籤 2）的 Dice 與 IoU（medpy 的 `dc`、`jc`），邊界情況照官方：
  - 預測與 GT 都非空：正常計算。
  - 預測非空、GT 為空：Dice = 1、IoU = 1。
  - 預測為空：Dice = 0、IoU = 0（包含 GT 也為空的情況）。
- **Overall**：所有影像的平均（包含該屬性為 Unknown 的影像）。
- **族群分數**：該族群影像的平均；Unknown（-1）不屬於任何族群。
- **ES-Dice／ES-IoU**：overall ÷ (1 + Σ_g |族群分數_g − overall|)，α = 1（論文正文的 Δ 版）。沒有成員的族群不計入 Σ（官方程式會得到 NaN）；正式資料不會發生，發生時要印出 warning。
- **「平均 Dice」與「平均 ES-Dice」**：(cup + rim) / 2。這兩個數字用於選 checkpoint 與選 K。
- 各屬性的族群：

| 屬性 | 族群 |
|---|---|
| race | Asian、Black、White |
| gender | Female、Male |
| ethnicity | Hispanic、Non-Hispanic |
| language | English、Spanish、Others |
| age | < 60、≥ 60（第 5.2 節） |
| maritalstatus | Married or partnered、Not married（第 5.2 節） |

論文表格沒有報告 age 和 marital status，這兩個屬性只和自己的結果比較。

## 7. 模型

沿用 experiment_001 的 DeepLabV3+ 結構（程式重寫）：

| 部分 | 設定 |
|---|---|
| Backbone | torchvision ResNet-50，ImageNet 預訓練（`IMAGENET1K_V2`）；去掉 avgpool 與 fc |
| Output stride | 16：layer4 改用 dilation（`replace_stride_with_dilation=(False, False, True)`）；224 輸入時 layer4 輸出為 2,048×14×14 |
| ASPP | 1×1 分支 + 3 個 3×3 dilated 分支（rates 6／12／18）+ image pooling 分支，各 256 channels；concat 後 1×1 投影到 256、BN、ReLU、dropout 0.1 |
| Low-level 特徵 | layer1 輸出（256 channels，56×56），1×1 conv + BN + ReLU 降到 48 channels |
| Decoder | ASPP 輸出 bilinear 放大到 56×56，與 low-level 特徵 concat，接兩層 3×3 conv（256）+ BN + ReLU，再 1×1 conv 輸出 3 類 logits |
| 輸出 | logits bilinear 放大到輸入大小（224×224） |
| 遮罩位置 | layer4 輸出（Z）之後、ASPP 之前；被選中的 channels 在所有空間位置設為 0 |
| 其他 | 卷積用一般卷積（非 separable）；不用 multi-grid |

Head 的初始化方式【交給實作】（experiment_001 用 Kaiming normal）。

## 8. 訓練

| 項目 | 設定 |
|---|---|
| Loss | L = 0.5 · CE + 0.5 · Dice loss（11/23 使用者決定；和官方 TransUNet 不同，官方的 Dice 分母有平方且包含背景） |
| CE | 3 類、不加權，對所有像素取平均 |
| Dice loss | 1 − mean_{c ∈ {rim, cup}} (2 Σ p_c g_c + ε) / (Σ p_c + Σ g_c + ε)；p 為 softmax 機率，g 為 one-hot；分母不平方；不算背景 |
| Dice 的加總範圍 | **【我的建議】** 整個 batch 一起加總（batch-level），和 FairSeg 官方與 experiment_001 相同 |
| ε | **【我的建議】** 1e-5（FairSeg 官方的值） |
| Optimizer | SGD，momentum 0.9，weight decay 1e-4，套用到所有參數 |
| Learning rate | 0.01，每個 iteration 依 poly 衰減：lr = 0.01 × (1 − iter / max_iter)^0.9，max_iter = 300 × 每 epoch iteration 數 |
| Batch size | 總共 24；DataParallel 分到兩張卡，每張 12（BN 統計量各卡分開算） |
| Epochs | 300 |
| 資料增強 | 第 5.4 節 |
| Seed | 42（Python、NumPy、PyTorch）；cuDNN deterministic【交給實作】 |
| 驗證 | 每個 epoch 結束後，在 val 1,000 張上用第 6 節的方式評估 |
| Checkpoint ① | 驗證集平均 Dice 最高的 epoch |
| Checkpoint ②-屬性 | 六個屬性各一個：驗證集該屬性的平均 ES-Dice 最高的 epoch |
| 同分 | **【我的建議】** 取較早的 epoch |

另外【交給實作】：是否 `drop_last`、DataLoader worker 數、是否另存最後一個 epoch 供中斷後續跑。每個 epoch 的 train loss（CE、Dice 分開）與驗證指標要寫進 CSV，供畫訓練曲線與判斷 overfitting（8/13、7/30 meeting 要求）。

## 9. Baseline 的 test 評估

**【我的建議】** 以下只做評估、不需要額外訓練，用來和論文表格比較：

- Checkpoint ①：在 test 上對六個屬性分別評估（同一次推論，依屬性分組即可）。
- Checkpoint ②-屬性：在 test 上對各自的屬性評估。

## 10. I-score 與 BDA（ethnicity）

對 checkpoint ① 和 checkpoint ②-ethnicity **各做一次**以下流程，兩次完全獨立。

### 10.1 特徵

- 模型設為 `eval()`，不做資料增強，對 val 1,000 張推論。
- 取 layer4 輸出 Z（2,048×14×14），做 global average pooling，得到每張 2,048 維的特徵。

### 10.2 樣本與 Y

- 排除 ethnicity 為 Unknown 的影像（val 約 55 張），其餘約 945 張參與 I-score。
- Y：Hispanic = 1、Non-Hispanic = 0。

### 10.3 二值化

- 每個 channel 以參與 I-score 的樣本計算 median；大於 median 為 1，否則為 0。
- Median 只算一次，之後 BDA 不重算。

### 10.4 I-score

對一組 channels S，依每張影像在 S 上的 0／1 組合把樣本分成若干格 j：

I(S) = Σ_j n_j² (Ȳ_j − Ȳ)² ／ (n · s²)

- n：樣本數；n_j：第 j 格的樣本數；Ȳ：全部樣本的 Y 平均；Ȳ_j：第 j 格的 Y 平均；s² = (1/n) Σ (Y_i − Ȳ)²。
- S 為空集合時 I = 0。
- 這和學長的 `I_Score_function6.py` 與 experiment_001 是同一個公式。

### 10.5 BDA

重複 10,000 次（抽樣用的隨機種子 42，要記錄）：

1. 從 2,048 個 channels 隨機抽出 3 個不同的 channels，作為目前的 S，計算 I(S)。
2. 對 S 中每個 channel，計算拿掉它之後的 I；gain = 拿掉後的 I − 目前的 I。
3. 若最大 gain **> 0**，拿掉該 channel（多個 channel 同為最大 gain 時，拿掉 index 最小的），回到第 2 步；否則停止。
4. 記錄最後的 S 與 I(S)。

這個規則刻意和學長的 `I_Score_function6.py` 不同：學長的程式在 gain = 0 時也會移除，同分時移除 list 中最後一個。

### 10.6 排序、聯集與驗證集評估

1. 10,000 筆結果依 I(S) 由高到低排序；I 相同時依執行順序排列【我的建議】。
2. 去掉重複的 S（以排序後的 channel 組合判斷），得到 M 個不同的 subset。
3. 對 K = 1…M，U_K = 前 K 個 subset 的 channel 聯集。U_K 和 U_{K−1} 相同時跳過，所以要評估的聯集最多 2,048 種。
4. 每個要評估的 U_K：把 U_K 的 channels 在 Z 上設為 0，在 val 1,000 張（包含 Unknown，用於 overall）上依第 6 節評估，記錄 K、|U_K|、cup／rim Dice、cup／rim ES-Dice（ethnicity）、平均 Dice、平均 ES-Dice。
5. 另外記錄不遮罩（K = 0，U_0 為空集合）的結果，它也是選 K 的候選之一。
6. 為了加速，val 的 Z（約 1.6 GB）和 layer1 輸出（約 3.2 GB，float32）在 `eval()` 下先算好存起來，每個聯集只重跑 ASPP 與 decoder。存放在記憶體或硬碟、是否用 float16【交給實作】，但結果要和完整模型推論一致（第 13 節）。

### 10.7 選 K 與 test

候選為 K = 0 以及第 10.6 節評估過的所有 K。

- **K_Dice**：驗證集平均 Dice 最高的 K。
- **K_ES**：驗證集平均 ES-Dice（ethnicity）最高的 K。
- 同分時 **【我的建議】** 取 |U_K| 較小的（所以和不遮罩同分時選 K = 0）。
- 每個選出的 X* = U_K，用完整模型加遮罩在 test 上評估一次（第 6 節，依 ethnicity 分組；可同時記錄其他屬性的分數）。若 K_Dice = K_ES，該組只跑一次。
- 若選到 K = 0：表示在驗證集上沒有任何遮罩勝過不遮罩。這一組不另外跑 test，直接沿用不遮罩的 test 結果，並在報告中寫明「驗證集選到不遮罩」。遮罩不 fine-tune 時，K_Dice 很可能是這個結果。
- 風險：驗證集只有約 34 張 Hispanic，K_ES 是從最多 2,048 個候選遮罩中挑驗證集 ES-Dice 最高的，選出的遮罩可能只是貼合驗證集的雜訊，test 上的改善不一定成立。報告時要把驗證集與 test 的結果並列。

## 11. Test 評估總表（ethnicity 這一輪）

| 抽特徵的 checkpoint | 遮罩 | 跑 test 次數 |
|---|---|---|
| ① | 不遮罩 | 1（第 9 節，六個屬性一起） |
| ① | X* = U_{K_Dice} | 1 |
| ① | X* = U_{K_ES} | 1 |
| ②-ethnicity | 不遮罩 | 1 |
| ②-ethnicity | X* = U_{K_Dice} | 1 |
| ②-ethnicity | X* = U_{K_ES} | 1 |

加上第 9 節其他五個 ②-屬性的 baseline 評估。選到 K = 0 或 K_Dice = K_ES 的組合不另外跑，所以實際次數可能少於表上的 6 次。

## 12. 產出與紀錄

【交給實作】決定實際路徑，建議放在 repo 根目錄的 `outputs/deeplabv3plus/`（`outputs/` 只有 README 進 git）。至少要有：

- 切分清單（`train.txt`、`val.txt`、`test.txt`，commit 進 git）與切分檢查表。
- 訓練：設定檔或參數紀錄、每個 epoch 的 CSV、checkpoint ① 與六個 ②、log。
- BDA（每個 checkpoint 一份）：median 門檻、10,000 筆 BDA 結果（初始 S、每輪拿掉的 channel、最後 S 與 I）、去重後排序、每個 U_K 的驗證集結果 CSV、選出的 K 與 X*。
- Test：每次評估的完整指標（cup／rim 的 Dice、IoU、ES-Dice、ES-IoU、各族群分數），格式可以對照 `docs/baseline_reproduction.md` 的表格。
- `runs.txt`：每次正式執行的時間、指令、git commit、是否有未 commit 的修改。

## 13. 驗證（實作完成後、正式執行前）

- **小資料試跑**：用 16 張的 `outputs/lists_tiny`（另指定輸出資料夾）跑完訓練、選 checkpoint、BDA、選 K、test 整條流程，不要寫進正式輸出資料夾。
- **切分檢查**：第 5.3 節的檢查全部通過。
- **I-score 檢查**：在固定的小例子上，新寫的 I-score 和學長的 `I_Score_function6.py` 算出相同的值（plan 的 Task 7 使用 11/23 用學長程式算出的參考值）。
- **快取路徑檢查**：遮罩為空時，在相同的 batch size（1）下，用快取特徵重跑 ASPP + decoder 的逐張 rim／cup Dice，要和完整模型直接推論的結果完全相同。
- **評估檢查**：Dice／IoU 的邊界情況（第 6 節）各有一個小測試。

## 14. 之後再做

- 其他五個屬性的 BDA（race、language、marital status 需要 one-vs-rest）。
- 偏差分類器、保留「重要且與敏感屬性無關的 channels」。
- 處理論文表格疑似欄位對調；age、marital status 和自己重現的 SAMed／TransUNet 比較。
- 調整 ASPP rates、多個 seed、遮罩後 fine-tune、視覺化與解釋性。

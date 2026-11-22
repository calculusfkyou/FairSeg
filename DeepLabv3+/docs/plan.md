# DeepLabV3+ ethnicity 實驗實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 在 `FairSeg/DeepLabv3+/` 寫出一套 DeepLabV3+／ResNet-50 程式，完成 spec 的切分、訓練、I-score + BDA（ethnicity）與 test 評估。

**Architecture：** 和 `SAMed/`、`TransUNet/` 一樣是獨立資料夾，以它為 cwd 執行。資料讀取與評估照 FairSeg 官方 TransUNet 的流程重寫在本資料夾（repo 慣例是各模型各有一份，不互相 import）。共用的評估函式（`utils.py`）同時給訓練、BDA、test 使用；BDA 的數學部分（`iscore.py`）是純 NumPy，和模型分開。

**Tech Stack：** Python 3.12、torch 2.14.1+cu126、torchvision 0.29.1、NumPy、SciPy（`ndimage.zoom`、`rotate`）、MedPy 0.5.2（`dc`、`jc`）。全部已在 `requirements-lock.txt`，不新增套件。

**Spec：** `DeepLabv3+/docs/spec.md`（以下稱 spec，章節以 §N 表示）

## Global Constraints

- Python 一律用 repo 的 `.venv/Scripts/python.exe`；不新增任何套件（repo 沒有 pytest，測試是可直接執行的 Python 檔）。
- 所有程式以 `FairSeg/DeepLabv3+` 為 cwd 執行；資料夾名稱含 `+`，不能被當成套件 import。
- `<DATA>` = HF `Dataset` 資料夾（含 `Training/`、`Test/`），本機為 `C:/Users/user/source/repo/master-thesis-fair-seg/data/fairseg_full/Dataset`。
- Test set 不參與任何選擇：checkpoint、median、I-score、BDA、選 K 都只用 val。每組 X* 在 test 上只跑一次。
- 評估方式 = FairSeg 官方 TransUNet：224×224、逐張 medpy `dc`／`jc`、官方邊界規則、overall 含 Unknown、族群不含 Unknown、ES = overall ÷ (1 + Σ|group − overall|)。
- Age 分組 < 60／≥ 60；marital 分組 Married（0）／Not married（1–4）；Unknown 一律 -1。
- 所有隨機種子 42（切分、訓練、BDA 抽樣）。
- Windows：DataLoader 的 `worker_init_fn` 是模組層級函式（用 `functools.partial`）、`persistent_workers=True`；checkpoint 用 `save_checkpoint`（照 `TransUNet/trainer.py`），不直接 `torch.save(obj, path)`。
- 正式執行前先 commit；`runs.txt` 記錄 commit 與是否有未 commit 的修改。
- 文件用繁體中文；commit message 用英文。

## Review Focus

spec 沒寫、但最容易出事的五種情況，各自的測試已加到負責的 task：

1. **某個族群在資料中一個都沒有**（小資料試跑、或某屬性在 val 缺某族群）：該族群不計入 ES 的 Σ，ES 仍是有限值，而不是 NaN 讓「最佳 checkpoint」永遠不更新。→ Task 3 `test_summarize_skips_empty_group`
2. **驗證指標在不同 epoch 完全同分**：保留較早的 epoch，不會被後面同分的 epoch 覆蓋。→ Task 6 `test_update_best_keeps_earlier_on_tie`
3. **用 DataParallel 存的 checkpoint 在單卡載入**：存檔時去掉 `module.` 前綴，單卡 `strict=True` 載入成功。→ Task 6 `test_state_dict_has_no_module_prefix`
4. **同一組 X* 不小心在 test 上跑第二次**：輸出檔已存在時直接拒絕執行。→ Task 9 `test_refuses_existing_output`
5. **浮點誤差讓 BDA 的 gain = 0 或同分判斷失準**：gain 與同分都用 1e-12 容差判斷，gain 剛好為 0 時不移除。→ Task 7 `test_bda_zero_gain_does_not_drop`、`test_bda_tie_drops_smaller_index`

---

## 檔案結構

| 檔案 | 責任 |
|---|---|
| `DeepLabv3+/datasets/dataset_fairseg.py` | 讀 npz、屬性分組、前處理與資料增強、Dataset |
| `DeepLabv3+/make_split.py` | 產生 7000／1000／2000 清單與切分檢查表 |
| `DeepLabv3+/lists/FairSeg_split/{train,val,test}.txt`、`split_report.csv` | 切分結果（commit 進 git） |
| `DeepLabv3+/networks/deeplabv3plus.py` | 模型，含 `encode`／`decode` 拆分與 channel 遮罩 |
| `DeepLabv3+/utils.py` | 評估指標、ES、loss、逐張推論、`save_checkpoint`、`seed_worker` |
| `DeepLabv3+/train.py`、`trainer.py` | 參數與訓練迴圈、每個 epoch 驗證、存 7 個 checkpoint |
| `DeepLabv3+/iscore.py` | 二值化、I-score、BDA、排序去重、前綴聯集（純 NumPy） |
| `DeepLabv3+/bda.py` | 抽 val 特徵與快取、跑 BDA、在 val 評估每個聯集、選 K |
| `DeepLabv3+/test.py` | 在指定清單上評估 checkpoint（可加遮罩），依屬性彙總 |
| `DeepLabv3+/tests/test_*.py` | 各 task 的測試，`python tests/test_x.py` 執行，全部通過時印出 `OK` |
| `scripts/deeplabv3plus.sh` | 正式執行入口：train／bda／test 三個 stage，寫 `runs.txt` 與 log |
| `scripts/launch_deeplabv3plus.ps1` | 背景執行並讓 Windows 保持喚醒（照 `launch_transunet.ps1`） |
| `scripts/benchmark_batch.py`（修改） | 加入 `--model deeplabv3plus` |
| `scripts/watch_training.ps1`（修改） | 加入這次訓練的監控項目 |

輸出（不進 git）：`outputs/deeplabv3plus/`，底下 `train_seed42/`、`bda_dice/`、`bda_es_ethnicity/`、`test/`、`runs.txt`、各 stage 的 log。

測試檔的共同寫法：檔案開頭先把 `DeepLabv3+/` 插到 import 路徑最前面（用 `python tests/test_x.py` 執行時，`sys.path[0]` 是 `tests/`；而且 stdlib 也有名為 `test` 的套件，必須用 insert 而不是 append）：

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

每個 `test_*` 函式用 `assert`，檔尾

```python
if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
```

---

### Task 1：資料集與屬性分組

**Files：**
- Create：`DeepLabv3+/datasets/dataset_fairseg.py`
- Test：`DeepLabv3+/tests/test_dataset.py`

**Interfaces：**
- Produces：
  - `ATTRS = ('race', 'gender', 'ethnicity', 'language', 'maritalstatus', 'age')`
  - `GROUP_NAMES: dict[str, tuple[str, ...]]`，索引 = 分組後的代碼：race `('Asian', 'Black', 'White')`、gender `('Female', 'Male')`、ethnicity `('Non-Hispanic', 'Hispanic')`、language `('English', 'Spanish', 'Others')`、maritalstatus `('Married', 'Not married')`、age `('<60', '>=60')`
  - `read_attrs(data: Mapping[str, np.ndarray]) -> dict[str, int]`：六個屬性的分組代碼，Unknown 為 -1
  - `index_npz_files(base_dir: str) -> dict[str, str]`（照 `TransUNet/datasets/dataset_fairseg.py`）
  - `RandomGenerator(output_size: int = 224, crop: int = 512)`、`TestGenerator(output_size: int = 224, crop: int = 512)`：`__call__(sample: dict) -> dict`，輸出 `image` FloatTensor[3, 224, 224]（0–1）、`label` LongTensor[224, 224]（0／1／2）
  - `FairSegDataset(base_dir: str, list_file: str, transform)`：`__getitem__` 回傳 `{'image', 'label', 'attrs': LongTensor[6]（ATTRS 順序）, 'name': str}`

- [ ] **Step 1：寫測試**

```python
def test_read_attrs_grouping():
    d = {'race': np.array(1), 'gender': np.array(0), 'ethnicity': np.array(-1), 'language': np.array(2),
         'maritalstatus': np.array(3), 'age': np.array(59.99)}
    assert read_attrs(d) == {'race': 1, 'gender': 0, 'ethnicity': -1, 'language': 2, 'maritalstatus': 1, 'age': 0}
    d.update(maritalstatus=np.array(0), age=np.array(60.0))
    assert read_attrs(d)['maritalstatus'] == 0 and read_attrs(d)['age'] == 1
    d.update(maritalstatus=np.array(-1))
    assert read_attrs(d)['maritalstatus'] == -1

def test_test_generator():
    image = np.full((664, 798), 255, np.uint8)
    mask = np.zeros((664, 798)); mask[300:364, 360:440] = -1; mask[320:344, 390:410] = -2
    out = TestGenerator()({'image': image, 'label': mask})
    assert out['image'].shape == (3, 224, 224) and out['image'].dtype == torch.float32
    assert abs(out['image'].max().item() - 1.0) < 1e-6
    assert out['label'].shape == (224, 224) and out['label'].dtype == torch.int64
    assert set(out['label'].unique().tolist()) == {0, 1, 2}
    assert out['label'][112, 112] == 2   # 中央裁切 512 後，影像中心落在 cup 內

def test_random_generator_keeps_label_values():
    random.seed(0); np.random.seed(0)
    image = np.random.randint(0, 256, (664, 798)).astype(np.uint8)
    mask = np.zeros((664, 798)); mask[300:364, 360:440] = -1; mask[320:344, 390:410] = -2
    for _ in range(10):
        out = RandomGenerator()({'image': image.copy(), 'label': mask.copy()})
        assert out['image'].shape == (3, 224, 224)
        assert set(out['label'].unique().tolist()) <= {0, 1, 2}
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_dataset.py`
Expected：`ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3：實作 `datasets/dataset_fairseg.py`**

前處理照 spec §5.4 第 1–5 步，逐行對照 `TransUNet/datasets/dataset_fairseg.py` 的 `RandomGenerator`／`TestGenerator`（含 `random_rot_flip`、`random_rotate` 的機率與 `order=0`、縮放的 `order=3`／`order=0`）。ImageNet 正規化放在模型裡（Task 5），這裡不做。`read_attrs` 只讀 `ethnicity` 鍵（HF 版沒有 `hispanic`）。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_dataset.py`
Expected：`OK`

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/datasets DeepLabv3+/tests/test_dataset.py
git commit -m "DeepLabV3+: dataset, attribute grouping and FairSeg preprocessing"
```

---

### Task 2：分層切分

**Files：**
- Create：`DeepLabv3+/make_split.py`
- Create（產出）：`DeepLabv3+/lists/FairSeg_split/train.txt`、`val.txt`、`test.txt`、`split_report.csv`
- Test：`DeepLabv3+/tests/test_split.py`

**Interfaces：**
- Consumes：`ATTRS`、`GROUP_NAMES`、`read_attrs`、`index_npz_files`（Task 1）
- Produces：
  - `stratified_split(names: Sequence[str], keys: Sequence[Hashable], n_val: int, seed: int) -> tuple[list[str], list[str]]`：回傳 (train, val)，兩者都依 `names` 原本的順序
  - `split_report(attrs: Mapping[str, dict[str, int]], train: Sequence[str], val: Sequence[str], attr_names: Sequence[str] = ATTRS) -> list[dict]`：每列 `attr, group, n_train, n_val, pct_train, pct_val, diff`；`group` 用 `GROUP_NAMES` 的名稱，代碼 -1 的列名稱為 `'Unknown'`；百分比不含 Unknown，Unknown 列的百分比為 `None`
  - 清單檔：每行一個檔名，例如 `data_03998.npz`

- [ ] **Step 1：寫測試**

```python
def test_exact_quotas():
    names = [f'f{i}' for i in range(80)]
    keys = ['a'] * 40 + ['b'] * 24 + ['c'] * 16
    train, val = stratified_split(names, keys, n_val=10, seed=42)
    assert len(train) == 70 and len(val) == 10 and not set(train) & set(val)
    assert set(train) | set(val) == set(names)
    by_key = Counter(keys[names.index(n)] for n in val)
    assert by_key == {'a': 5, 'b': 3, 'c': 2}
    assert stratified_split(names, keys, 10, 42) == (train, val)

def test_largest_remainder():
    names = [f'f{i}' for i in range(20)]
    keys = ['a'] * 7 + ['b'] * 7 + ['c'] * 6        # 名額 1.75／1.75／1.5
    _, val = stratified_split(names, keys, n_val=5, seed=42)
    assert Counter(keys[names.index(n)] for n in val) == {'a': 2, 'b': 2, 'c': 1}

def test_report_excludes_unknown():
    attrs = {'n1': {'ethnicity': 1}, 'n2': {'ethnicity': 0}, 'n3': {'ethnicity': -1}, 'n4': {'ethnicity': 0}}
    rows = {r['group']: r for r in split_report(attrs, ['n1', 'n2', 'n3'], ['n4'], attr_names=('ethnicity',))}
    assert rows['Hispanic']['n_train'] == 1 and rows['Hispanic']['pct_train'] == 50.0
    assert rows['Hispanic']['pct_val'] == 0.0
    assert rows['Unknown']['pct_train'] is None
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_split.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `stratified_split`、`split_report` 與 CLI**

分配方式照 spec §5.3：每層名額 = 層大小 × n_val ／ 總數，先取整數部分，剩餘名額依小數部分由大到小給（同分用 `random.Random(seed)` 決定順序），每層內再用同一個 RNG 抽樣。分層鍵是 `tuple(read_attrs(npz)[a] for a in ATTRS)`。

CLI：`python make_split.py --root_path <DATA> --official_dir ../TransUNet/lists/FairSeg_final --out_dir lists/FairSeg_split --n_val 1000 --seed 42`。讀官方 `train.txt` 的 8,000 張、寫出三份清單（`test.txt` 直接複製官方的）與 `split_report.csv`，並印出每個屬性 train／val 比例的最大差距。寫檔前用 `assert` 檢查：train 7,000、val 1,000、兩者不重疊、聯集等於官方 train、val 與官方 test 不重疊。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_split.py`
Expected：`OK`

- [ ] **Step 5：Commit 程式**

```bash
git add DeepLabv3+/make_split.py DeepLabv3+/tests/test_split.py
git commit -m "DeepLabV3+: stratified train/val split script"
```

- [ ] **Step 6：用已 commit 的程式產生正式切分並檢查**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe make_split.py --root_path <DATA> --official_dir ../TransUNet/lists/FairSeg_final --out_dir lists/FairSeg_split --n_val 1000 --seed 42`

Expected：
- `train.txt` 7,000 行、`val.txt` 1,000 行、`test.txt` 2,000 行；train ∩ val、val ∩ test 都是空集合。
- 各屬性最大差距都不超過約 0.5 個百分點；val 的 Hispanic 約 33–35 張。（11/23 模擬值，實作的抽樣順序不同時數字會略有差異。）

- [ ] **Step 7：Commit 切分結果**

```bash
git add DeepLabv3+/lists/FairSeg_split
git commit -m "DeepLabV3+: official 7000/1000/2000 split lists (seed 42)"
```

---

### Task 3：評估指標與 ES

**Files：**
- Create：`DeepLabv3+/utils.py`（本 task 只放評估部分）
- Test：`DeepLabv3+/tests/test_metrics.py`

**Interfaces：**
- Consumes：`ATTRS`、`GROUP_NAMES`（Task 1）
- Produces：
  - `case_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]`：鍵 `rim_dice, rim_iou, cup_dice, cup_iou`；輸入是 224×224 的 0／1／2 標籤圖
  - `summarize(records: Sequence[dict], attr: str) -> dict`：`records` 每筆含上面四個指標與 `'attrs': dict[str, int]`；回傳 `cup_dice, rim_dice, cup_iou, rim_iou, cup_es_dice, rim_es_dice, cup_es_iou, rim_es_iou, mean_dice, mean_es_dice` 與 `groups: {族群名稱: {'n', 'cup_dice', 'rim_dice', 'cup_iou', 'rim_iou'}}`

- [ ] **Step 1：寫測試**

```python
def test_case_metrics_edges():
    gt = np.zeros((4, 4), int); gt[0, :4] = 1                     # rim 4 px，沒有 cup
    pred = np.zeros((4, 4), int); pred[0, :2] = 1; pred[1, :2] = 1  # rim 4 px，重疊 2 px
    pred[3, 3] = 2                                                  # 預測有 cup、GT 沒有
    m = case_metrics(pred, gt)
    assert abs(m['rim_dice'] - 0.5) < 1e-9 and abs(m['rim_iou'] - 1 / 3) < 1e-9
    assert m['cup_dice'] == 1 and m['cup_iou'] == 1                  # 官方規則：預測非空、GT 空 → 1
    empty = case_metrics(np.zeros((4, 4), int), np.zeros((4, 4), int))
    assert empty['cup_dice'] == 0 and empty['rim_iou'] == 0          # 預測為空 → 0（含 GT 也空）

def _rec(cup, eth):
    return {'cup_dice': cup, 'rim_dice': 0.5, 'cup_iou': cup, 'rim_iou': 0.5,
            'attrs': {'ethnicity': eth, 'gender': 0}}

def test_summarize_es():
    s = summarize([_rec(0.8, 0), _rec(0.6, 0), _rec(0.4, 1), _rec(1.0, -1)], 'ethnicity')
    assert abs(s['cup_dice'] - 0.7) < 1e-9                           # overall 含 Unknown
    assert s['groups']['Hispanic']['n'] == 1 and s['groups']['Non-Hispanic']['n'] == 2
    assert abs(s['cup_es_dice'] - 0.7 / 1.3) < 1e-9                  # |0.7−0.7| + |0.4−0.7| = 0.3
    assert abs(s['rim_es_dice'] - 0.5) < 1e-9
    assert abs(s['mean_dice'] - 0.6) < 1e-9
    assert abs(s['mean_es_dice'] - (0.7 / 1.3 + 0.5) / 2) < 1e-9

def test_summarize_skips_empty_group():
    s = summarize([_rec(0.8, 0), _rec(0.6, 0)], 'gender')            # 沒有 Male
    assert s['groups']['Male']['n'] == 0
    assert math.isfinite(s['cup_es_dice']) and abs(s['cup_es_dice'] - 0.7) < 1e-9
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_metrics.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `case_metrics` 與 `summarize`**

`case_metrics` 照 `TransUNet/utils.py` 的 `calculate_metric_percase`（只取 Dice 與 IoU，不算 HD95），rim 用 `label == 1`、cup 用 `label == 2`。`summarize` 照 `TransUNet/inference.py` 的彙總與 `equity_scaled_perf`（α = 1），只多一條規則：沒有成員的族群不計入 Σ。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_metrics.py`
Expected：`OK`

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/utils.py DeepLabv3+/tests/test_metrics.py
git commit -m "DeepLabV3+: per-image Dice/IoU and equity-scaled summary as in FairSeg"
```

---

### Task 4：Loss

**Files：**
- Modify：`DeepLabv3+/utils.py`
- Test：`DeepLabv3+/tests/test_loss.py`

**Interfaces：**
- Produces：`CEDiceLoss(eps: float = 1e-5)`，`forward(logits: Tensor[B, 3, H, W], target: LongTensor[B, H, W]) -> tuple[Tensor, Tensor, Tensor]`，回傳 `(loss, ce, dice)`，其中 `loss = 0.5 * ce + 0.5 * dice`

- [ ] **Step 1：寫測試**

```python
def test_dice_not_squared_and_no_background():
    logits = torch.zeros(1, 3, 2, 2)                      # softmax 後每類都是 1/3
    target = torch.ones(1, 2, 2, dtype=torch.long)        # 全部是 rim
    loss, ce, dice = CEDiceLoss()(logits, target)
    # rim：(2·4/3 + ε)/(4/3 + 4 + ε) ≈ 0.5；cup：(0 + ε)/(4/3 + ε) ≈ 0 → dice loss ≈ 1 − 0.25
    assert abs(dice.item() - 0.75) < 1e-4
    assert abs(loss.item() - (0.5 * ce.item() + 0.5 * dice.item())) < 1e-6

def test_perfect_prediction():
    target = torch.tensor([[[0, 1], [2, 1]]])
    logits = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    loss, ce, dice = CEDiceLoss()(logits, target)
    assert ce.item() < 1e-6 and dice.item() < 1e-4

def test_no_foreground_is_finite():
    loss, _, _ = CEDiceLoss()(torch.randn(2, 3, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))
    assert torch.isfinite(loss)
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_loss.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `CEDiceLoss`**

照 spec §8：CE 是 3 類、不加權；Dice 對 rim、cup 兩類分別計算 (2 Σ p g + ε)／(Σ p + Σ g + ε)，加總範圍是整個 batch，取兩類平均後以 1 減去。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_loss.py`
Expected：`OK`

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/utils.py DeepLabv3+/tests/test_loss.py
git commit -m "DeepLabV3+: 0.5 CE + 0.5 foreground soft Dice loss"
```

---

### Task 5：模型

**Files：**
- Create：`DeepLabv3+/networks/deeplabv3plus.py`
- Test：`DeepLabv3+/tests/test_model.py`

**Interfaces：**
- Produces：
  - `DeepLabV3Plus(num_classes: int = 3, pretrained: bool = True)`
    - `encode(x: Tensor[B, 3, H, W]) -> tuple[Tensor, Tensor]`：輸入 0–1，模型內先做 ImageNet mean／std 正規化；回傳 `(z, low)`，224 輸入時 `z` 為 [B, 2048, 14, 14]（layer4）、`low` 為 [B, 256, 56, 56]（layer1）
    - `decode(z: Tensor, low: Tensor, size: tuple[int, int]) -> Tensor`：ASPP → 放大 → concat → decoder → logits 放大到 `size`
    - `forward(x: Tensor, mask_channels: Sequence[int] | None = None) -> Tensor`：`decode(apply_channel_mask(z, mask_channels), low, x.shape[-2:])`
  - `apply_channel_mask(z: Tensor, channels: Sequence[int] | None) -> Tensor`：`None` 或空序列時原樣回傳

- [ ] **Step 1：寫測試**

```python
def test_shapes_and_mask():
    torch.manual_seed(0)
    net = DeepLabV3Plus(pretrained=False).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        z, low = net.encode(x)
        assert z.shape == (2, 2048, 14, 14) and low.shape == (2, 256, 56, 56)
        out = net(x)
        assert out.shape == (2, 3, 224, 224)
        assert torch.equal(out, net(x, mask_channels=[]))
        assert torch.equal(out, net.decode(z, low, (224, 224)))
        zm = apply_channel_mask(z, [0, 5])
        assert zm[:, [0, 5]].abs().sum() == 0
        keep = [c for c in range(2048) if c not in (0, 5)]
        assert torch.equal(zm[:, keep], z[:, keep])
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_model.py`
Expected：`ImportError`

- [ ] **Step 3：實作模型**

結構照 spec §7 的表格（torchvision `resnet50`、`IMAGENET1K_V2`、`replace_stride_with_dilation=(False, False, True)`、ASPP rates 6／12／18 + image pooling、各 256 channels、投影後 dropout 0.1、low-level 48 channels、兩層 3×3／256、bilinear 放大）。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_model.py`
Expected：`OK`

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/networks DeepLabv3+/tests/test_model.py
git commit -m "DeepLabV3+: ResNet-50 DeepLabV3+ with encode/decode split and layer4 channel mask"
```

---

### Task 6：訓練

**Files：**
- Modify：`DeepLabv3+/utils.py`（加 `predict_records`、`save_checkpoint`、`seed_worker`、`state_dict_of`）
- Create：`DeepLabv3+/train.py`、`DeepLabv3+/trainer.py`
- Create：`scripts/deeplabv3plus.sh`、`scripts/launch_deeplabv3plus.ps1`
- Modify：`scripts/benchmark_batch.py`、`scripts/watch_training.ps1`
- Test：`DeepLabv3+/tests/test_trainer.py`

**Interfaces：**
- Consumes：Task 1 的 Dataset 與 transforms、Task 3 的 `case_metrics`／`summarize`、Task 4 的 `CEDiceLoss`、Task 5 的模型
- Produces：
  - `predict_records(model, loader, mask_channels: Sequence[int] | None = None) -> list[dict]`：`eval()`、`no_grad`，逐張 argmax 後 `case_metrics`，每筆加上 `'attrs'`（ATTRS → int）與 `'name'`
  - `save_checkpoint(obj, path: str) -> None`：照 `TransUNet/trainer.py`
  - `seed_worker(seed: int, worker_id: int) -> None`
  - `state_dict_of(model: nn.Module) -> dict`：有 `DataParallel` 時取 `.module.state_dict()`
  - `update_best(best: dict[str, float], epoch: int, summaries: dict[str, dict]) -> list[str]`（`trainer.py`）：`summaries` 是 ATTRS → `summarize` 的結果；檢查 `'dice'`（用任一屬性的 `mean_dice`，六個都相同）與 `'es_<attr>'`（該屬性的 `mean_es_dice`），只有嚴格大於目前最佳值才更新，回傳更新了哪些名稱
  - Checkpoint 檔：`<output_dir>/best_dice.pth`、`best_es_<attr>.pth`（6 個），內容 `{'state_dict', 'epoch', 'val': summaries}`
  - `<output_dir>/epochs.csv` 欄位：`epoch, lr, loss, ce, dice_loss, val_mean_dice, val_cup_dice, val_rim_dice`，再加六欄 `val_mean_es_dice_<attr>`
  - log 每 20 個 iteration 一行 `iteration <n> : loss : <x>`（`watch_training.ps1` 解析這個格式）
  - CLI：`train.py --root_path <DATA> --list_dir lists/FairSeg_split --output_dir ../outputs/deeplabv3plus/train_seed42 --max_epochs 300 --batch_size 24 --n_gpu 2 --base_lr 0.01 --seed 42 --num_workers 8`。**`--batch_size` 是兩張卡加總的 batch**，和 TransUNet 的每卡 batch 慣例不同，也不做任何 lr 換算。

- [ ] **Step 1：寫測試**

```python
def _summ(mean_dice, es):
    return {a: {'mean_dice': mean_dice, 'mean_es_dice': es} for a in ATTRS}

def test_update_best_keeps_earlier_on_tie():
    best = {}
    assert set(update_best(best, 1, _summ(0.80, 0.70))) == {'dice'} | {f'es_{a}' for a in ATTRS}
    assert update_best(best, 2, _summ(0.80, 0.70)) == []
    assert update_best(best, 3, _summ(0.81, 0.70)) == ['dice']

def test_state_dict_has_no_module_prefix():
    net = DeepLabV3Plus(pretrained=False)
    sd = state_dict_of(nn.DataParallel(net))
    assert not any(k.startswith('module.') for k in sd)
    DeepLabV3Plus(pretrained=False).load_state_dict(sd, strict=True)
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_trainer.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `utils.py` 新增函式、`trainer.py`、`train.py`**

照 spec §8：SGD（momentum 0.9、weight decay 1e-4、所有參數）；每個 iteration 更新 lr = 0.01 × (1 − iter／max_iter)^0.9，max_iter = epochs × len(trainloader)；DataParallel；train DataLoader 用 `shuffle=True`、`pin_memory=True`、`persistent_workers=True`、`worker_init_fn=functools.partial(seed_worker, seed)`；val DataLoader batch 1、不打亂，也要 `persistent_workers=True`（否則 Windows 每個 epoch 重啟 workers 約 50 秒）。開始訓練前把全部參數寫進 log，並存成 `<output_dir>/args.json`。每個 epoch 結束跑一次 `predict_records(val)`，對六個屬性各 `summarize` 一次，寫 CSV、呼叫 `update_best`，對回傳的名稱用 `save_checkpoint` 存 `state_dict_of(model)`。`train.py` 設定 Python／NumPy／PyTorch 種子為 42、`cudnn.deterministic = True`、`cudnn.benchmark = False`，所有會啟動 worker 的程式碼放在 `if __name__ == '__main__':` 之後。不另存最後一個 epoch（spec 交給實作，這裡選擇不做）。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_trainer.py`
Expected：`OK`

- [ ] **Step 5：小資料試跑**

建立試跑清單（`outputs/` 不進 git）：

```bash
mkdir -p outputs/lists_tiny_dlv3
cp outputs/lists_tiny/train.txt outputs/lists_tiny_dlv3/train.txt
cp outputs/lists_tiny/test.txt outputs/lists_tiny_dlv3/val.txt
cp outputs/lists_tiny/test.txt outputs/lists_tiny_dlv3/test.txt
```

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe train.py --root_path <DATA> --list_dir ../outputs/lists_tiny_dlv3 --output_dir ../outputs/deeplabv3plus_smoke/train --max_epochs 2 --batch_size 4 --n_gpu 1 --num_workers 2`

Expected：`epochs.csv` 有 2 列；`best_dice.pth` 與 6 個 `best_es_*.pth` 都存在；log 中有 `iteration` 行。

- [ ] **Step 6：`scripts/benchmark_batch.py` 加入 `--model deeplabv3plus` 並量測**

`--model` 的選項加上 `deeplabv3plus`，模型資料夾對應 `DeepLabv3+`；`build()` 回傳 `DeepLabV3Plus()`、上述 SGD 與輸入 224；`step()` 在這個分支用 `CEDiceLoss`。檔案開頭的 `from utils import DiceLoss` 要改成依 `--model` 分支 import（`DeepLabv3+/utils.py` 沒有 `DiceLoss`）。原本兩個模型的行為不變。

Run：`.venv/Scripts/python.exe scripts/benchmark_batch.py --model deeplabv3plus --n_gpu 2 --batch_sizes 24`

Expected：有一列 batch 24 的結果，且沒有 `near VRAM limit` 警告。出現警告就停下來，回報結果再決定 batch size。

- [ ] **Step 7：正式執行用的腳本**

- `scripts/deeplabv3plus.sh <DATA> <stage> <tag> [其他參數]`：`stage` 為 `train`、`bda` 或 `test`，在 `DeepLabv3+/` 執行 `train.py`、`bda.py`、`test.py`，自動加上 `--root_path "$DATA"`，其他參數原樣傳下去。開始與結束各在 `outputs/deeplabv3plus/runs.txt` 寫一行（格式照 `scripts/train_transunet.sh`：tag、commit、`(uncommitted changes)` 標記、時間、exit code），開始那一行另外寫出完整的執行指令。輸出用 append（`>>`）寫到 `outputs/deeplabv3plus/<tag>.log`，避免用相同 tag 重跑時清空原本的 log。
- `scripts/launch_deeplabv3plus.ps1 <DataDir> <Tag>`：照 `launch_transunet.ps1`，讓 Windows 保持喚醒並執行 `deeplabv3plus.sh <DataDir> train <Tag> --list_dir lists/FairSeg_split --output_dir ../outputs/deeplabv3plus/train_seed42 --max_epochs 300 --batch_size 24 --n_gpu 2 --base_lr 0.01 --seed 42`；`outputs/deeplabv3plus/train_seed42` 已存在時先改名，不刪除。
- `scripts/watch_training.ps1`：`$models` 加入 `@{ Key = 'deeplabv3plus_seed42'; Name = 'DeepLabV3+ (seed 42)'; Log = Join-Path $repo 'outputs\deeplabv3plus\deeplabv3plus_seed42.log'; PerEpoch = 292; Epochs = 300 }`（7,000／24 取上整數），`$lastKey` 改成 `'deeplabv3plus_seed42'`；`Test-TrainingAlive` 的條件改成同時接受 `FairSeg_final` 和 `FairSeg_split`；找 `exit` 紀錄時，除了 `outputs\baseline\runs.txt` 也要讀 `outputs\deeplabv3plus\runs.txt`。
- `scripts/README.md`：加入 `deeplabv3plus.sh`、`launch_deeplabv3plus.ps1` 的說明，以及 `benchmark_batch.py` 新增的 `--model deeplabv3plus`。

- [ ] **Step 8：Commit**

```bash
git add DeepLabv3+/utils.py DeepLabv3+/train.py DeepLabv3+/trainer.py DeepLabv3+/tests/test_trainer.py scripts/deeplabv3plus.sh scripts/launch_deeplabv3plus.ps1 scripts/benchmark_batch.py scripts/watch_training.ps1 scripts/README.md
git commit -m "DeepLabV3+: training loop with per-epoch validation and seven selection checkpoints"
```

---

### Task 7：I-score 與 BDA

**Files：**
- Create：`DeepLabv3+/iscore.py`
- Test：`DeepLabv3+/tests/test_iscore.py`

**Interfaces：**
- Produces（輸入都是 NumPy，`X` 為 0／1 的 [n, d]，`y` 為 0／1 的 [n]）：
  - `binarize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]`：回傳 `(X, medians)`，`X = features > medians`（每個 channel 的 median）
  - `iscore(X, y, subset: Sequence[int]) -> float`：spec §10.4；空集合或 `y` 變異數為 0 時回傳 0
  - `bda_once(X, y, start: Sequence[int]) -> tuple[tuple[int, ...], float, list[int]]`：回傳 `(final, score, dropped)`，`final` 為排序後的 tuple，`dropped` 依拿掉的順序
  - `run_bda(X, y, n_runs: int = 10000, start_size: int = 3, seed: int = 42) -> list[dict]`：每筆 `{'run', 'start', 'dropped', 'final', 'score'}`；起始組合用 `np.random.default_rng(seed).choice(d, start_size, replace=False)` 依序抽出
  - `rank_unique(runs: Sequence[dict]) -> list[tuple[tuple[int, ...], float]]`：依 score 由高到低、同分時依 `run` 順序，去掉重複的 `final`
  - `prefix_unions(ranked) -> list[tuple[int, tuple[int, ...]]]`：回傳 `(K, U_K)`，K 從 1 起算，只列出聯集有變大的 K

測試用資料（I-score 數值已在 2026-11-23 用學長的 `I_Score_function6.py` 驗證相同）：

```python
X = np.array([[0,0],[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1],[1,1]])
y = np.array([0,0,1,1,1,1,1,0,0,0])
NOISE = np.array([0,1,0,1,0,1,0,1,0,1])
A = np.array([0,1,1,0,0,1,1,0,1,1]); B = np.array([1,0,0,1,0,1,0,0,0,0])
```

- [ ] **Step 1：寫測試**

```python
def test_iscore_matches_senior_values():
    assert abs(iscore(X, y, [0, 1]) - 1.8) < 1e-12
    assert abs(iscore(X, y, [0]) - 0.2) < 1e-12 and abs(iscore(X, y, [1]) - 0.2) < 1e-12
    assert iscore(X, y, []) == 0 and iscore(X, np.zeros(10, int), [0]) == 0

def test_bda_drops_noise():
    X3 = np.column_stack([X, NOISE])                       # I({0,1,2}) = 1.0
    final, score, dropped = bda_once(X3, y, [0, 1, 2])
    assert final == (0, 1) and dropped == [2] and abs(score - 1.8) < 1e-12

def test_bda_zero_gain_does_not_drop():
    X4 = np.column_stack([X, NOISE, NOISE])                # 拿掉 2 或 3 的 gain 都剛好是 0
    final, score, dropped = bda_once(X4, y, [0, 1, 2, 3])
    assert final == (0, 1, 2, 3) and dropped == [] and abs(score - 1.0) < 1e-12

def test_bda_tie_drops_smaller_index():
    X5 = np.column_stack([X, A, B])                        # 拿掉 2 或 3 都得到 1.4
    final, score, dropped = bda_once(X5, y, [0, 1, 2, 3])
    assert dropped == [2, 3] and final == (0, 1) and abs(score - 1.8) < 1e-12

def test_run_bda_deterministic():
    rng = np.random.default_rng(0)
    Xr, yr = rng.integers(0, 2, (50, 20)), rng.integers(0, 2, 50)
    a, b = run_bda(Xr, yr, n_runs=30, seed=42), run_bda(Xr, yr, n_runs=30, seed=42)
    assert a == b and len(a) == 30
    assert all(len(set(r['start'])) == 3 and all(0 <= c < 20 for c in r['start']) for r in a)

def test_rank_and_unions():
    runs = [{'run': 0, 'final': (1, 2), 'score': 0.5}, {'run': 1, 'final': (3,), 'score': 0.9},
            {'run': 2, 'final': (1, 2), 'score': 0.5}, {'run': 3, 'final': (4,), 'score': 0.5}]
    assert rank_unique(runs) == [((3,), 0.9), ((1, 2), 0.5), ((4,), 0.5)]
    ranked = [((3,), .9), ((1, 3), .8), ((1,), .7), ((5,), .6)]
    assert prefix_unions(ranked) == [(1, (3,)), (2, (1, 3)), (4, (1, 3, 5))]
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_iscore.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `iscore.py`**

BDA 照 spec §10.5。gain 判斷用 `gain > 1e-12`；同分判斷用 1e-12 容差，同分時拿掉 index 最小的 channel。I-score 的分格可以把選中的 channels 當成二進位數字編碼（同學長的做法），用 `np.bincount` 計算每格的 n_j 與 Y 總和。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_iscore.py`
Expected：`OK`

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/iscore.py DeepLabv3+/tests/test_iscore.py
git commit -m "DeepLabV3+: I-score, backward dropping, ranking and prefix unions"
```

---

### Task 8：BDA 流程與選 K

**Files：**
- Create：`DeepLabv3+/bda.py`
- Test：`DeepLabv3+/tests/test_bda.py`

**Interfaces：**
- Consumes：Task 5 的 `encode`／`decode`、Task 3 的 `case_metrics`／`summarize`、Task 6 的 `predict_records`、Task 7 的全部函式
- Produces：
  - `iscore_inputs(features: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`：回傳 `(X, y, medians, keep_idx)`；只保留 `codes != -1` 的樣本，median 也只用這些樣本算；`codes` 中出現 -1、0、1 以外的值時丟出 `ValueError`（這一輪只支援二元屬性）
  - `choose_k(rows: Sequence[dict], metric: str) -> dict`：`rows` 每筆含 `K, n_channels` 與指標；取 `metric` 最大者，同分時取 `n_channels` 較小者
  - `extract_val(model, loader) -> dict`：`{'z': FloatTensor[N, 2048, 14, 14] (CPU), 'low': FloatTensor[N, 256, 56, 56] (CPU), 'labels': uint8 [N, 224, 224], 'attrs': list[dict], 'names': list[str]}`
  - `records_from_cache(model, cache: dict, channels: Sequence[int], batch_size: int = 50) -> list[dict]`：格式同 `predict_records`
  - CLI：`bda.py --root_path <DATA> --list_dir lists/FairSeg_split --checkpoint <ckpt> --attribute ethnicity --output_dir <dir> --n_runs 10000 --seed 42`（`--list_dir`、`--n_runs`、`--seed` 的預設值就是這三個值）
  - 輸出：`args.json`（全部參數，含 seed 與 n_runs）、`medians.npy`、`bda_runs.json`（全部 10,000 筆）、`ranked.json`、`union_val.csv`（欄位 `K, n_channels, cup_dice, rim_dice, cup_es_dice, rim_es_dice, mean_dice, mean_es_dice`，第一列是 K = 0）、`selection.json`（`{'dice': {'K', 'channels'}, 'es': {'K', 'channels'}}`）

- [ ] **Step 1：寫測試**

```python
def test_iscore_inputs_excludes_unknown():
    feats = np.array([[1., 5.], [2., 6.], [100., 100.], [3., 7.], [4., 8.]])
    X, y, med, keep = iscore_inputs(feats, np.array([0, 1, -1, 0, 1]))
    assert keep.tolist() == [0, 1, 3, 4] and y.tolist() == [0, 1, 0, 1]
    assert np.allclose(med, [2.5, 6.5])                    # median 不含 Unknown 那一列
    assert X.shape == (4, 2)

def test_iscore_inputs_rejects_multiclass():
    try:
        iscore_inputs(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError('expected ValueError')

def test_choose_k_ties_prefer_fewer_channels():
    rows = [{'K': 0, 'n_channels': 0, 'mean_dice': 0.80, 'mean_es_dice': 0.70},
            {'K': 1, 'n_channels': 3, 'mean_dice': 0.80, 'mean_es_dice': 0.72},
            {'K': 4, 'n_channels': 9, 'mean_dice': 0.79, 'mean_es_dice': 0.72}]
    assert choose_k(rows, 'mean_dice')['K'] == 0
    assert choose_k(rows, 'mean_es_dice')['K'] == 1
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_bda.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `bda.py`**

流程照 spec §10.1–10.7：載入 checkpoint（單卡、`eval()`）→ `extract_val` → `z` 做 GAP 得到特徵 → `iscore_inputs` → `run_bda` → `rank_unique` → `prefix_unions` → K = 0 與每個 U_K 用 `records_from_cache` 評估、`summarize(attribute)` → 寫 `union_val.csv` → `choose_k` 兩次（`mean_dice`、`mean_es_dice`）→ 寫 `selection.json`。

為了讓結果可以逐位元重現，`bda.py` 開頭設定 `cudnn.deterministic = True`、`cudnn.benchmark = False`、`torch.backends.cudnn.allow_tf32 = False`、`torch.backends.cuda.matmul.allow_tf32 = False`；`extract_val` 用 batch 1 的 val loader。

**執行時的一致性檢查（不可省略）**：寫任何輸出前，比較 `records_from_cache(model, cache, [], batch_size=1)` 和 `predict_records(model, val_loader)`（同樣是 batch 1）的每張 `cup_dice`、`rim_dice`，必須完全相同；不同就丟出錯誤並停止（spec §13 的快取路徑檢查）。兩邊用相同的 batch size，是為了避免 cuDNN 因 batch size 不同而選用不同演算法、造成邊界像素誤判。之後評估各個聯集時可以用 batch 50 加速。

- [ ] **Step 4：執行測試，確認通過**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_bda.py`
Expected：`OK`

- [ ] **Step 5：小資料試跑**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe bda.py --root_path <DATA> --list_dir ../outputs/lists_tiny_dlv3 --checkpoint ../outputs/deeplabv3plus_smoke/train/best_dice.pth --attribute ethnicity --output_dir ../outputs/deeplabv3plus_smoke/bda --n_runs 50`

Expected：一致性檢查通過；`union_val.csv` 第一列 K = 0；`selection.json` 有 `dice` 與 `es` 兩組。（16 張裡 Hispanic 可能很少或沒有；若 `y` 只有一類，I-score 全為 0，流程仍要跑完。）

- [ ] **Step 6：Commit**

```bash
git add DeepLabv3+/bda.py DeepLabv3+/tests/test_bda.py
git commit -m "DeepLabV3+: BDA pipeline with cached validation features and K selection"
```

---

### Task 9：Test 評估

**Files：**
- Create：`DeepLabv3+/test.py`
- Test：`DeepLabv3+/tests/test_eval.py`

**Interfaces：**
- Consumes：Task 1 的 Dataset、Task 3 的 `summarize`、Task 5 的模型、Task 6 的 `predict_records`、Task 8 的 `selection.json`
- Produces：
  - `ensure_new(path: str) -> None`：檔案已存在時丟出 `FileExistsError`
  - CLI：`test.py --root_path <DATA> --list_dir lists/FairSeg_split --split test --checkpoint <ckpt> [--selection <selection.json> --key dice|es] [--attributes race gender ...] --output <json>`；`--list_dir` 預設 `lists/FairSeg_split`，`--attributes` 預設 ATTRS 全部；開頭設定和 `bda.py` 相同的 deterministic／TF32 選項
  - 輸出 JSON：`{'args', 'checkpoint', 'split', 'mask_channels', 'attributes': {attr: summarize 結果}}`，並在終端機印出每個屬性 cup／rim 的 Dice、IoU、ES-Dice、ES-IoU 與各族群 Dice

- [ ] **Step 1：寫測試**

```python
def test_refuses_existing_output():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.json')
        ensure_new(path)                                  # 不存在：通過
        open(path, 'w').close()
        try:
            ensure_new(path)
        except FileExistsError:
            return
        raise AssertionError('expected FileExistsError')
```

- [ ] **Step 2：執行測試，確認失敗**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_eval.py`
Expected：`ImportError`

- [ ] **Step 3：實作 `test.py`**

一開始就呼叫 `ensure_new(output)`；有 `--selection` 時從中讀 `[key]['channels']` 當遮罩；`predict_records` 跑一次，對每個屬性 `summarize`。

- [ ] **Step 4：執行測試與小資料試跑**

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe tests/test_eval.py`
Expected：`OK`

Run：`cd DeepLabv3+ && ../.venv/Scripts/python.exe test.py --root_path <DATA> --list_dir ../outputs/lists_tiny_dlv3 --split test --checkpoint ../outputs/deeplabv3plus_smoke/train/best_dice.pth --selection ../outputs/deeplabv3plus_smoke/bda/selection.json --key es --output ../outputs/deeplabv3plus_smoke/test_es.json`
Expected：產生 JSON，`attributes` 有六個屬性；同一指令再跑一次時立即因 `FileExistsError` 停止。

- [ ] **Step 5：Commit**

```bash
git add DeepLabv3+/test.py DeepLabv3+/tests/test_eval.py
git commit -m "DeepLabV3+: test-set evaluation with optional channel mask"
```

---

### Task 10：正式執行

沒有新程式，只有依序執行與檢查。每一步都透過 `scripts/deeplabv3plus.sh`，讓 `runs.txt` 有紀錄。

- [ ] **Step 1：確認工作目錄乾淨**

Run：`git status --porcelain --untracked-files=no`
Expected：沒有輸出。

- [ ] **Step 2：訓練（背景執行）**

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_deeplabv3plus.ps1 <DATA> deeplabv3plus_seed42
```

Expected：300 個 epoch 跑完，`runs.txt` 的 exit 為 0，`train_seed42/` 有 7 個 checkpoint 與 300 列的 `epochs.csv`。

- [ ] **Step 3：BDA（checkpoint ① 與 ②-ethnicity 各一次）**

先做完所有只用 val 的步驟，再碰 test。

```bash
bash scripts/deeplabv3plus.sh <DATA> bda bda_dice --list_dir lists/FairSeg_split --checkpoint ../outputs/deeplabv3plus/train_seed42/best_dice.pth --attribute ethnicity --n_runs 10000 --seed 42 --output_dir ../outputs/deeplabv3plus/bda_dice
bash scripts/deeplabv3plus.sh <DATA> bda bda_es_ethnicity --list_dir lists/FairSeg_split --checkpoint ../outputs/deeplabv3plus/train_seed42/best_es_ethnicity.pth --attribute ethnicity --n_runs 10000 --seed 42 --output_dir ../outputs/deeplabv3plus/bda_es_ethnicity
```

Expected：兩個資料夾都有 `selection.json`；一致性檢查通過。

- [ ] **Step 4：Baseline 的 test 評估（spec §9）**

```bash
bash scripts/deeplabv3plus.sh <DATA> test test_best_dice --list_dir lists/FairSeg_split --split test --checkpoint ../outputs/deeplabv3plus/train_seed42/best_dice.pth --output ../outputs/deeplabv3plus/test/best_dice.json
```

六個 `best_es_<attr>.pth` 各跑一次，加上 `--attributes <attr>`，輸出 `test/best_es_<attr>.json`。

- [ ] **Step 5：遮罩後的 test 評估（spec §11，最多 4 次）**

對兩個 BDA 資料夾的 `dice`、`es` 兩組各跑一次 `test.py --selection <...>/selection.json --key <dice|es> --attributes ethnicity`，輸出 `test/<bda 資料夾名>_<key>.json`。以下情況不跑，在結果文件中註明：
- 選到的 K = 0：直接沿用對應的不遮罩結果。
- `dice` 與 `es` 選到相同的 K：只跑一次。

- [ ] **Step 6：整理結果文件**

在 `DeepLabv3+/docs/` 新增結果文件（繁體中文），表格格式對照 `docs/baseline_reproduction.md`，至少包含：
- 切分檢查表（`split_report.csv`）。
- Baseline 六個屬性的 test 結果，前四個屬性和論文 SAMed／TransUNet 並列，並註明訓練條件的差異：DeepLabV3+ 用 7,000 張訓練、在 val 上選 checkpoint、loss 不同；SAMed／TransUNet 用 8,000 張、取最後一個 epoch。
- 兩次 BDA 的 K 對驗證集平均 Dice／平均 ES-Dice 曲線（`union_val.csv`）與選出的 K、|X*|。
- 遮罩前後的 ethnicity test 結果，驗證集與 test 並列（spec §10.7 的風險說明）。

同時更新 `outputs/README.md`：列出 `outputs/deeplabv3plus/` 底下哪些是正式結果，並標明 `outputs/deeplabv3plus_smoke/`、`outputs/lists_tiny_dlv3/` 是試跑用。

```bash
git add DeepLabv3+/docs outputs/README.md
git commit -m "DeepLabV3+: ethnicity experiment results"
```

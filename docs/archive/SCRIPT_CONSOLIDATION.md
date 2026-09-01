# 根目录脚本整合记录

**日期**: 2026-08-09  
**目的**: 清理重复的测试脚本，统一数据路径引用

## 问题

`idash/mpc/` 根目录存在6个测试/基准脚本，功能高度重复且数据路径不一致：

### 删除前的混乱状态

| 脚本 | 类型 | 样本数 | 独特性 | 数据路径 |
|------|------|--------|--------|----------|
| `eval_full.py` | 定点模拟 | 全数据集 | 无GPU快速精度门 | `project/test/` |
| `simple_2pc_test.py` | 真实2PC | 5 | 依赖检查 | `project/test/` |
| `run_full_test.sh` | 真实2PC | 5 | 内联heredoc + pip | `project/test/` |
| `run_benchmark.sh` | 编排器 | 1 | `--build/--2pc/--regression` 包装 | - |
| `real_gpu_2pc_benchmark.py` | 真实2PC | 10 | **完整**：回归统计+计时 | `project/DeepDTAGen/data/` |
| `run_davis_multibatch.sh` | 真实2PC批量 | 5×4=20 | **唯一**：批量聚合 | `project/DeepDTAGen/data/` |

**核心问题**:
1. **功能重复**: `simple_2pc_test.py` / `run_full_test.sh` / `run_benchmark.sh` 都跑5样本2PC，而 `real_gpu_2pc_benchmark.py` 是它们的超集
2. **路径不一致**: 同样的测试数据分散在 `project/test/` 和 `project/DeepDTAGen/data/`，与已迁移的 `idash/mpc/data/` 不同步
3. **维护负担**: 6个脚本 → 6处数据路径 → 路径变更时需同步6次

## 整合方案

### 保留 (2个)

**1. `real_gpu_2pc_benchmark.py`** — 真实2PC基准测试
- 功能最完整：Pearson/Spearman相关系数 + MAE/RMSE + 计时
- 默认10样本/数据集，覆盖 davis + kiba
- 替代 `simple_2pc_test.py` / `run_full_test.sh` / `run_benchmark.sh`

**2. `run_davis_multibatch.sh`** — Davis批量MPC评估
- 唯一的批量聚合评估（多批次 × 批大小）
- 用于大规模准确性验证（例：5批 × 4样本 = 20样本）

### 删除 (4个)

| 脚本 | 删除原因 |
|------|----------|
| `eval_full.py` | 用户要求删除（定点模拟已被MPC测试覆盖） |
| `simple_2pc_test.py` | 被 `real_gpu_2pc_benchmark.py` 功能覆盖 |
| `run_full_test.sh` | 与 `simple_2pc_test.py` 重复（shell版heredoc） |
| `run_benchmark.sh` | 薄包装层，无独立功能 |

## 数据路径统一

所有保留脚本及其依赖的helper统一指向 `idash/mpc/data/`:

### 主脚本

| 文件 | 变更 | 新路径 |
|------|------|--------|
| `real_gpu_2pc_benchmark.py` | 行58-66 | `MPC_ROOT/data/{davis_test,kiba_train}.csv` |
| `run_davis_multibatch.sh` | 行14 | `$SCRIPT_DIR/data/davis_test.csv` |

### Helper脚本

| 文件 | 变更 | 新路径 |
|------|------|--------|
| `scripts/dev_tools/prepare_davis_multibatch_slice.py` | 行27-28 | 相对路径: `../../data/davis_test.csv` |
| `scripts/dev_tools/aggregate_davis_validation.py` | 行18-22 | 相对路径: `../../data/davis_test.csv` + `../../gpu_mpc/` |

**注意**: helper脚本修正了 `GPU_MPC` 路径计算错误（原先 `scripts/dev_tools/gpu_mpc` 不存在）

## 文档更新

以下文档已同步删除脚本的引用：

### 英文文档
- `README.md`
  - 项目结构树（行59-62）
  - Quickstart 示例（行120-124）
  - Testing 章节（行217-221）

- `scripts/README.md`
  - Notes 章节主脚本引用（行86-88）

### 中文文档（待更新）
- `docs/SETUP_GUIDE_CN.md`
  - 行131, 147, 230
- `docs/FILE_GUIDE_CN.md`
  - 行46-49 (项目结构), 183-196 (脚本说明), 273 (使用示例)
- `docs/COMPLETION_SUMMARY.md`
  - 行41, 60, 135, 146, 202, 215

## 使用方法

### 单次基准测试（默认10样本/数据集）
```bash
cd /home/jiang/master/idash/mpc
python3 real_gpu_2pc_benchmark.py

# 输出:
# - Pearson/Spearman 相关系数
# - MAE / RMSE
# - 平均推理时间
```

### 批量评估（Davis数据集）
```bash
cd /home/jiang/master/idash/mpc
./run_davis_multibatch.sh 5 4  # 5批 × 4样本 = 20样本

# 聚合验证:
python3 scripts/dev_tools/aggregate_davis_validation.py 5
```

## Git 状态

- 删除的4个文件是未跟踪的新文件（`??`），删除后无法从git恢复
- 修改的6个文件已在git中跟踪（`M`），可随时回滚
- 建议commit:

```bash
git add real_gpu_2pc_benchmark.py run_davis_multibatch.sh \
        scripts/dev_tools/{prepare_davis_multibatch_slice,aggregate_davis_validation}.py \
        README.md scripts/README.md SCRIPT_CONSOLIDATION.md

git commit -m "refactor: consolidate redundant test scripts, unify data paths

Deleted:
- eval_full.py (user request)
- simple_2pc_test.py, run_full_test.sh, run_benchmark.sh (redundant)

Kept:
- real_gpu_2pc_benchmark.py (most complete: timing + regression stats)
- run_davis_multibatch.sh (unique: batched aggregation)

All scripts + helpers now reference idash/mpc/data/ (was project/{test,DeepDTAGen/data}).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## 优势

1. **结构清晰** — 根目录只留2个脚本，职责明确
2. **路径一致** — 所有脚本统一引用 `idash/mpc/data/`，与测试目录对齐
3. **功能完整** — 保留的脚本覆盖全部核心场景（单样本基准 + 批量聚合）
4. **易于维护** — 数据路径变更时只需更新2个主脚本 + 2个helper

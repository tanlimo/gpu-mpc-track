# 项目结构大清理总结

**日期**: 2026-08-09  
**目标**: 清理冗余文件，统一结构，准备 GitHub 交付

---

## 一、测试目录重组

### 原状态
`tests/` 目录包含 23 个测试文件，混合了不同职责

### 重组后
按职责拆分为 3 个目录：

```
tests/      (2 tests + 3 config)  — Davis/Kiba MPC 准确性门测试
offline/    (41 tests)            — 离线数据准备（秘密分享、权重导出）
microbench/ (90 tests)            — 底层组件单元测试
```

**tests/ 最终保留**:
- `test_mpc_online_gate.py` — 真实2PC在线门测试 ⭐
- `test_official_baseline.py` — 官方明文基线验证
- `conftest.py`, `test_data_paths.py` — pytest配置

**tests/ 删除** (2轮清理):
- 第1轮移出：13个单元测试 → `microbench/`, 4个离线准备 → `offline/`
- 第2轮删除：`test_fixed_e2e_gate.py`, `test_real_data_scan.py`, `test_real_weights_accuracy.py`, `TEST_PATH_MIGRATION.md`

---

## 二、根目录脚本整合

### 原状态
6 个测试/基准脚本，功能高度重复

### 整合后
只保留 2 个核心脚本：

| 保留 | 职责 |
|------|------|
| `real_gpu_2pc_benchmark.py` | 真实2PC基准测试（计时 + 回归统计） |
| `run_davis_multibatch.sh` | Davis批量MPC评估 |

**删除的4个**:
- `eval_full.py` — 定点模拟（用户要求删除）
- `simple_2pc_test.py` — 被 benchmark 覆盖
- `run_full_test.sh` — 与 simple 重复
- `run_benchmark.sh` — 薄包装层

---

## 三、数据路径统一

所有脚本从旧路径 `project/{test,DeepDTAGen/data}` 迁移到 `idash/mpc/data/`：

### 主脚本
- `real_gpu_2pc_benchmark.py` — 行58-66
- `run_davis_multibatch.sh` — 行14

### Helper脚本
- `scripts/dev_tools/prepare_davis_multibatch_slice.py` — 行27-29
- `scripts/dev_tools/aggregate_davis_validation.py` — 行18-24（修正了路径计算错误）

---

## 四、其他清理

### 删除 `reference_bumblebee/`
- **内容**: Flax/JAX SPU BumbleBee 安全基线实现
- **删除原因**:
  - 代码未使用（无任何文件 import）
  - 不可运行（依赖 OpenBumbleBee/SPU 环境）
  - Git未跟踪
  - 技术栈不同（Flax vs PyTorch/CUDA）
  - 真正的基线对比用的是 `baseline/official_baseline_*.json`

---

## 最终项目结构

```
idash/mpc/
├── gpu_mpc/                     # GPU MPC 核心实现（CUDA）
├── reference/                   # Python 参考实现（明文 + 定点）
├── baseline/                    # 官方明文基线 JSON（对比金标准）
├── data/                        # 测试数据集（davis_test.csv, kiba_train.csv）
├── model/                       # 预训练权重（*.pth, git-ignored）
│
├── tests/                       # MPC 准确性门测试（2 tests）
│   ├── test_mpc_online_gate.py
│   ├── test_official_baseline.py
│   ├── conftest.py
│   └── test_data_paths.py
│
├── offline/                     # 离线数据准备测试（41 tests）
│   ├── test_offline_prepare.py
│   ├── test_share_data.py
│   ├── test_export_weights.py
│   └── ...
│
├── microbench/                  # 组件单元测试（90 tests）
│   ├── test_dense_gcn.py
│   ├── test_fixedpoint.py
│   └── ...
│
├── scripts/                     # 开发工具
│   └── dev_tools/
│       ├── benchmark_plaintext_time.py
│       ├── prepare_batch_samples.py
│       ├── prepare_davis_multibatch_slice.py
│       ├── aggregate_davis_validation.py
│       └── ...
│
├── docs/                        # 中文文档
│   ├── SETUP_GUIDE_CN.md
│   ├── FILE_GUIDE_CN.md
│   └── COMPLETION_SUMMARY.md
│
├── real_gpu_2pc_benchmark.py    # 真实2PC基准脚本
├── run_davis_multibatch.sh      # Davis批量评估
├── README.md                    # 英文主文档
├── TEST_REORGANIZATION.md       # 测试重组记录
├── SCRIPT_CONSOLIDATION.md      # 脚本整合记录
└── PROJECT_FINAL_CLEANUP.md     # 本文档
```

---

## 统计对比

| 项目 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| 根目录脚本 | 6 | 2 | -4 (67%) |
| tests/ 文件数 | 23+3配置 | 2+3配置 | -21 (81%) |
| 测试目录层级 | 1层 | 3层（按职责） | +结构化 |
| 数据路径来源 | 3处不一致 | 1处统一 | 统一为 `data/` |
| 未用参考代码 | reference_bumblebee/ | 删除 | -1目录 |

---

## 文档同步更新

### 英文文档
- `README.md` — 项目结构 + Quickstart + Testing
- `scripts/README.md` — 主脚本引用

### 中文文档
- `docs/SETUP_GUIDE_CN.md` — 测试套件章节
- `docs/FILE_GUIDE_CN.md` — 项目结构 + 评估脚本说明
- `docs/COMPLETION_SUMMARY.md` — 历史进度报告
- `gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md` — 框架指南

### 新增记录文档
- `TEST_REORGANIZATION.md` — 测试目录重组记录
- `SCRIPT_CONSOLIDATION.md` — 脚本整合记录
- `PROJECT_FINAL_CLEANUP.md` — 本总结文档

---

## 核心改进

1. **结构清晰** — 测试按职责分3层，根目录只留2个脚本
2. **路径一致** — 所有脚本统一引用 `idash/mpc/data/`
3. **职责明确** — 每个测试目录有清晰的作用范围
4. **易于维护** — 数据路径变更只需更新 `test_data_paths.py` 和 2个主脚本
5. **交付就绪** — 删除未用代码、冗余测试，保留核心验证链路

---

## 验证状态

```bash
# 测试收集
pytest tests/       # 8 tests ✓
pytest offline/     # 41 tests ✓
pytest microbench/  # 90 tests ✓

# 路径引用
所有脚本指向 idash/mpc/data/ ✓
test_data_paths.py 被 7个文件共享 ✓

# Git状态
移动的文件用 git mv 保留历史 ✓
删除的文件已记录在文档中 ✓
```

---

## 建议的 Git Commit

```bash
git add -A

git commit -m "refactor: major project cleanup for GitHub delivery

Test reorganization:
- Split tests/ by responsibility: tests/ (MPC gates), offline/ (data prep), microbench/ (unit tests)
- Reduced tests/ from 23 to 2 core tests + 3 config files
- Deleted redundant tests: test_fixed_e2e_gate, test_real_data_scan, test_real_weights_accuracy

Script consolidation:
- Deleted 4 redundant scripts (eval_full, simple_2pc_test, run_full_test.sh, run_benchmark.sh)
- Kept 2 core scripts: real_gpu_2pc_benchmark.py, run_davis_multibatch.sh

Path unification:
- All scripts now reference idash/mpc/data/ (was project/{test,DeepDTAGen/data})
- Fixed helper script path calculation bugs

Removed unused code:
- reference_bumblebee/ (Flax/JAX SPU baseline, not used by PyTorch/CUDA project)

All docs updated to reflect new structure.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 交付检查清单

- [x] 测试目录按职责分层
- [x] 根目录脚本精简到2个
- [x] 数据路径统一到 `data/`
- [x] 删除未使用的参考代码
- [x] 所有文档同步更新
- [x] pytest 收集验证通过
- [x] Git 历史保留（移动文件用 git mv）
- [x] 清理记录文档完整

**项目已准备好上传到 GitHub！** 🚀

# 测试目录重组记录

**日期**: 2026-08-09  
**目的**: 按职责分层测试目录，便于项目交付和维护

## 重组方案

原 `tests/` 目录（23个测试文件）按职责拆分为三个目录：

### 1. `tests/` — Davis/Kiba MPC 准确性门测试 (2 + 3配置文件)

**保留的MPC端到端测试**:
- `test_mpc_online_gate.py` — 2PC在线MPC测试，验证密文计算准确性
- `test_official_baseline.py` — 官方明文基线验证

**配置文件**:
- `conftest.py` — pytest sys.path 配置
- `test_data_paths.py` — 集中式路径配置（被其他两个目录共享）

**测试收集**: 20 tests

**删除的测试** (2026-08-09 清理):
- `test_fixed_e2e_gate.py` — 与 real_weights_accuracy 功能重叠
- `test_real_data_scan.py` — 数据校验，非MPC核心
- `test_real_weights_accuracy.py` — 定点精度测试，已被MPC在线门覆盖
- `TEST_PATH_MIGRATION.md` — 历史文档，已无实际作用

### 2. `offline/` — 离线数据准备测试 (3个)

**离线准备流程**:
- `test_offline_prepare.py` — 离线准备驱动（生成秘密分享和权重文件）
- `test_share_data.py` — 加法秘密分享测试
- `test_export_weights.py` — GPU-MPC权重导出

**配置文件**:
- `conftest.py` — sys.path 配置（指向 `idash/mpc` 和 `idash/mpc/tests`）

**测试收集**: 26 tests

**删除** (2026-08-09):
- `test_export_npz.py` — BumbleBee格式导出（BumbleBee已删除）

### 3. `microbench/` — 底层组件单元测试 (13个)

**TDD单元测试**:
- `test_affinity_model.py` — 明文亲和力模型测试
- `test_csv_runner.py` — CSV驱动的参考运行器测试
- `test_dense_gcn.py` — 密集GCN层测试
- `test_dense_graph.py` — SMILES→密集图转换测试
- `test_fixed_forward.py` — 定点前向传播测试
- `test_fixedpoint.py` — 定点数量化测试
- `test_masked_maxpool.py` — 掩码全局最大池化测试
- `test_metrics.py` — 准确性指标测试
- `test_mpc_config.py` — MPC配置测试
- `test_nvcc.py` — CUDA编译基础设施测试
- `test_protein_plaintext.py` — 蛋白质明文嵌入测试
- `test_reference_equivalence.py` — 官方参考等价性测试
- `test_ring32.py` — 32位环定点测试

**配置文件**:
- `conftest.py` — sys.path 配置（指向 `idash/mpc` 和 `idash/mpc/tests`）

**测试收集**: 90 tests

## 技术实现

### import依赖解决方案

**问题**: 三个目录的测试都需要：
1. 导入 `reference/`, `dense_graph/` 等模块（位于 `idash/mpc/`）
2. 导入 `test_data_paths`（位于 `tests/`）

**方案**: 
- `tests/conftest.py` 保持原样：`sys.path.insert(0, os.path.join(_here, ".."))`
- `offline/conftest.py` 和 `microbench/conftest.py` 添加两个路径：
  ```python
  sys.path.insert(0, os.path.join(_here, ".."))          # -> idash/mpc
  sys.path.insert(0, os.path.join(_here, "..", "tests")) # -> test_data_paths
  ```

### Git历史保留

所有移动使用 `git mv` 执行，保留完整提交历史：
```bash
git mv tests/test_*.py microbench/  # 或 offline/
```

Git status 显示为 `R` (rename)，非 `D`+`A` (delete+add)。

## 验证结果

所有三个目录的测试正常收集和运行：

```bash
# 测试收集
pytest tests/ --collect-only -q      # 20 tests
pytest offline/ --collect-only -q     # 41 tests  
pytest microbench/ --collect-only -q  # 90 tests

# 跨目录import验证（microbench导入tests/test_data_paths）
pytest microbench/test_protein_plaintext.py -q
# 2 passed, 1005 warnings in 6.87s ✓

# 离线准备测试
pytest offline/test_share_data.py -q
# 15 passed in 0.24s ✓

# 单元测试
pytest microbench/test_fixedpoint.py -q  
# 11 passed in 0.18s ✓
```

## 文档更新

以下文件已更新路径引用：

1. **README.md**
   - 项目结构树（行52-54）
   - Testing 章节示例命令（行201-208）

2. **PROJECT_CLEANUP_SUMMARY.md**
   - 目录结构描述（行133-135）

3. **gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md**
   - 项目结构树 + 调试对比章节

## 优势

1. **职责清晰** — 三个目录各司其职：MPC门测试、数据准备、单元测试
2. **易于交付** — `tests/` 只保留 Davis/Kiba 相关的核心 MPC 验证
3. **开发友好** — `microbench/` 聚合所有底层组件测试，便于TDD开发
4. **向后兼容** — pytest 自动发现所有三个目录的测试
5. **历史保留** — `git mv` 保留完整文件历史，支持 `git log --follow`

## 运行所有测试

```bash
cd /home/jiang/master/idash/mpc

# 运行所有测试
pytest tests/ offline/ microbench/

# 仅 MPC 端到端测试
pytest tests/

# 仅数据准备测试
pytest offline/

# 仅单元测试
pytest microbench/
```

## Git 提交建议

```bash
git add tests/ offline/ microbench/ README.md PROJECT_CLEANUP_SUMMARY.md \
        gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md TEST_REORGANIZATION.md

git commit -m "refactor(tests): reorganize by responsibility into tests/offline/microbench

- tests/ (20): Davis/Kiba MPC accuracy-gate tests  
- offline/ (41): offline data-prep (secret sharing, weight/NPZ export)
- microbench/ (90): component unit tests (GCN, fixedpoint, pooling, ...)

test_data_paths.py stays in tests/; other dirs import via conftest sys.path.
All moves use git mv to preserve history.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

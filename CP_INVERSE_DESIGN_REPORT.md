# SPP2489 有限应变晶体塑性逆向设计报告

## 结论

真实 MOOSE–BoTorch 闭环已完成。32×32 隐藏目标为 **P11(1.01)=362.707025484 MPa**、**P11(1.02)=389.622645325 MPa**。最优观测设计为 **η*=0.000000000**；独立 FEM 重算的两个相对误差分别为 **0.482828%** 和 **0.823780%**。

该结果匹配的是规定的有限应变、率相关 FCC 晶体塑性加载路径，不是材料参数反演，也不意味着唯一恢复隐藏晶体取向。20 晶粒 RVE 是数值 demonstration，不代表统计收敛的 SS316L 性能。

## 计算定义

- 固定 20 晶粒周期性 Voronoi 几何、材料参数、滑移系和 A4 本构积分器；本任务未修改晶体塑性离散方程。
- 每个晶粒从随机端点到共同取向采用单位四元数最短弧 SLERP；同一旋转属性同时用于立方弹性张量、滑移方向和滑移面法向。
- Plane strain Total-Lagrangian 均匀化，F11=1+0.02t、T=1 s、P22=0、F12=F21=0。
- 目标函数为两个相对应力误差平方的等权平均；GP 对 −J 建模。失败 FEM 不进入训练数据。
- 端点 Euler 最大误差：η=0 为 0.000e+00°，η=1 为 0.000e+00°。

## 时间积分协议

η=0、0.5、1 的 current adaptive、strict adaptive 与 fixed-200 对照均实际运行（4 MPI ranks）。选择 `adaptive_strict`；其相对 fixed-200 的最大双指标差异为 0.048118%。输出同步机制确保 t=0.5 和 1.0 是真实求解时刻，而非事后线性插值。正式 inverse-design workflow 使用 8 MPI ranks；η=0.5 strict 的实测由 30.03 s 降至 22.14 s（1.36×）。

## Bayesian Optimization 与随机对照

- BO 共执行 6 次候选 FEM（其中顺序采集 0 次、失败 0 次），首次同时达到 1% 阈值的调用序号为 1。
- 候选 FEM 累计 wall time=134.495 s；GP 拟合与采集累计=0.000 s，二者分开统计。
- Random search executed 6 FEM calls; its best η=0.705151329, errors 2.760713% and 2.784314%. First tolerance call: None.
- BO 最优观测的 P11(1%)=364.458274966 MPa、P11(2%)=392.832277934 MPa、Hsec=2837.400296854 MPa。

本次随机搜索只有一个固定种子；结果仅用作闭环基准，不能据此宣称 BO 在统计意义上总优于随机搜索。对于一维变量，均匀网格同样是合理基准。

## 独立重算与多保真验证

- 32×32 独立最优 FEM：P11(1%)=364.458274966 MPa，P11(2%)=392.832277934 MPa，Hsec=2837.400296854 MPa；误差 0.482828% / 0.823780%；达标=True。
- 64×64 target: P11(1%)=359.282602893 MPa, P11(2%)=385.420851136 MPa.
- 64×64 optimized: P11(1%)=361.897113847 MPa, P11(2%)=389.216887728 MPa; same-mesh errors 0.727703% and 0.984907%.
- Target 32→64 changes: 0.953128% / 1.090183%; optimized 32→64 changes: 0.707704% / 0.928888%.
- 低、高保真比较都在相同网格内比较 target 与 candidate；没有把 32×32 candidate 直接与 64×64 target 混作优化误差。

## 参数敏感性、非唯一性与局部机制

- 已观测 η 区间内 P11(1%) 跨度为 101.260938 MPa，P11(2%) 跨度为 114.015985 MPa；按 η 排序后两条离散响应分别非单调、非单调。
- 候选观测点中有 1 个同时满足双 1% 指标；其 η 为 0.000000。优化完成后披露的隐藏参考为 η_ref=0.650000，其误差按构造为零；上述达标候选与参考 η 不同，因此在当前 1% 性能容差下不能唯一恢复原始织构。
- 最优与隐藏目标的平均累积滑移差为 -4.387833e-04，最大累积滑移差为 +8.223837e-03。Fig. 6 使用共同色标展示局部滑移，因此宏观性能接近时仍可识别局部机制差异。

## 数值与物理检查

独立最优解 all_checks_passed=True，accepted increments=53，failed attempts=6，total Newton iterations=348。最大 |P22|=8.247e-07 MPa，最大周期配对误差=1.554e-15，最大 Hill–Mandel 相对残差=2.750e-09，det(Fp) 范围=[1.000000004, 1.000037138]，塑性耗散=7.590812。检查还覆盖准确终止时间、同步 F11、宏观剪切、非零滑移、有限状态、正滑移阻力及 Exodus 字段。

## 可扩展性

单变量 demo 的主要成本是 CP FEM，不是 GP。扩展到二维或三维织构变量在软件结构上可行，但初始 DOE 与样本预算会增加；在启动大规模 64×64 BO 前，应采用多保真代理、并行候选或可信的低保真校正。

## 工程回归

- CP inverse-design Python tests: 10 passed。
- 原 MOOSE TestHarness: 8 passed, 0 failed, 0 skipped。
- 机器可读 workflow checks: 15/15 passed；缓存复用、目标隔离、独立重算、同网格高保真、图片/GIF/Exodus 均已检查。

## 产物与复现

- 8 张图：`results/cp_inverse_design/figures/fig01_*.png` 至 `fig08_*.png`。
- Exodus：`results/cp_inverse_design/exodus/`，包含隐藏目标与最优候选的低/高保真场。
- 机器可读历史：`results/cp_inverse_design/optimization_history.csv`、`random_search_history.csv`、`best_design.json`、`cp_inverse_design_summary.json`。
- GIF：`results/cp_inverse_design/animations/cp_inverse_design_history.gif`；每帧是一个真实 FEM 候选，不是连续形貌演化。

```bash
conda run --no-capture-output -n spp2489-inverse \
  python scripts/cp_inverse_design/optimize_cp_texture.py --mesh 32 \
  --max-bo-evals 12 --initial-samples 6 --target-tolerance 0.01 \
  --seed 2026 --reuse-cache --validate-high-fidelity
```

工作流实际总 wall time（含验证、绘图与缓存命中开销）为 531.934 s。

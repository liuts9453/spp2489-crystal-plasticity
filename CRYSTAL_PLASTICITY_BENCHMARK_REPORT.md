# SPP2489 20 晶粒有限应变 FCC 晶体塑性 RVE 与性能 Benchmark

## 1. 结论

本任务已完成，结果状态为**实测完成**，不是短程外推：

- 12 个 FCC `{111}<110>` 滑移系、乘法分解、幂律滑移和 Kalidindi 型硬化已直接接入现有 `new_system = true`、`formulation = TOTAL`、`strain = FINITE` 均匀化系统。
- 本构主路径以参考构形 PK2 为接口，由 MOOSE 的 `ComputeLagrangianStressPK2` 基类转换为 PK1；Cauchy 应力只作为 von Mises 输出的派生量，不参与本构积分或全局残差。
- 局部更新使用手工推导的解析局部 Jacobian和算法一致切线。有限差分只在显式开启验证选项时使用，从未用于生产求解。
- 20 晶粒 32×32、64×64 的 2% 拉伸全部完成；64×64/20 配置实测 365.80 s（6.10 min，4 MPI），64×64/40 配置实测 373.73 s（6.23 min）。
- 64×64/20 晶体塑性比对应超弹性慢 37.21 倍；局部晶体塑性积分占 wall-equivalent 时间 68.33%。
- 64×64 结果的平均累积滑移为 0.054856、最大积分点累积滑移为 0.125371，已经明显进入塑性阶段。
- 全部 TestHarness：12 passed、0 skipped、0 failed。

这里的材料参数用于算法 benchmark，不是 SS316L 标定结果。

## 2. MOOSE 接口审查

使用的 MOOSE commit 为 `11853d1623555a9e444ffb0725a4dc16f64fb255`
（`snapshot-20-10-27-58082-g11853d1623`）。源码审查得到：

1. 原生 `ComputeMultipleCrystalPlasticityStress` 最终提供 Cauchy 应力；其源码对 `Jacobian_mult` 明确注释为 `This is NOT the exact jacobian`。
2. 现有均匀化 Kernel 需要一致的 PK1 及其对总变形梯度的导数。
3. 通用旧材料包装器不能仅凭 property 名称把大变形各向异性 CP 变成可靠的 Total-Lagrangian CP。
4. 原生 Kalidindi 单晶模型本身可以运行，20 步参考结果为：取向 1 的最终 `P11=238.444684 MPa`，取向 2 的最终 `P11=354.852480 MPa`。但它不能直接作为当前 TL 均匀化的精确切线材料。

因此最终采用专用的 `ComputeLagrangianFCCCrystalPlasticityStress`，直接继承
`ComputeLagrangianStressPK2`。这不是 Cauchy 适配器；旧的 Cauchy 包装尝试已从工程中删除。

## 3. 本构离散与解析一致切线

### 3.1 离散局部问题

每个积分点、每个内部子步同时求解 18 个未知量：中间构形对称 PK2 应力的 6 个独立分量和 12 个滑移阻力，

\[
\boldsymbol x=(\boldsymbol S_e,g^1,\ldots,g^{12}).
\]

令 \(\boldsymbol M^\alpha=\boldsymbol s^\alpha\otimes\boldsymbol m^\alpha\)，后向 Euler 更新为

\[
\boldsymbol A=\boldsymbol I-\Delta t\sum_\alpha
\dot\gamma^\alpha\boldsymbol M^\alpha,
\qquad
\boldsymbol F_{p,n+1}^{-1}=\boldsymbol F_{p,n}^{-1}\boldsymbol A,
\]

\[
\boldsymbol F_e=\boldsymbol F\boldsymbol F_p^{-1},\qquad
\boldsymbol E_e=\tfrac12(\boldsymbol F_e^T\boldsymbol F_e-\boldsymbol I),
\qquad
\boldsymbol S_e=\mathbb C:\boldsymbol E_e.
\]

滑移率和硬化为

\[
\dot\gamma^\alpha=\dot\gamma_0
\left|\frac{\tau^\alpha}{g^\alpha}\right|^{1/m}
\operatorname{sign}(\tau^\alpha),
\]

\[
g_{n+1}^\alpha-g_n^\alpha-
\Delta t\sum_\beta q_{\alpha\beta}h^\beta
|\dot\gamma^\beta|=0.
\]

同一 `{111}` 面内取 \(q=1\)，其余取 latent ratio \(q=1.4\)。参考构形 PK2 为

\[
\boldsymbol S=J_p\boldsymbol F_p^{-1}\boldsymbol S_e\boldsymbol F_p^{-T},
\qquad J_p=\det\boldsymbol F_p,
\]

然后由基类计算 \(\boldsymbol P=\boldsymbol F\boldsymbol S\) 及全局 Jacobian。

### 3.2 解析线性化

代码显式使用

\[
\frac{\partial\dot\gamma^\alpha}{\partial\tau^\alpha}
=\frac1m\frac{|\dot\gamma^\alpha|}{|\tau^\alpha|},
\qquad
\frac{\partial\dot\gamma^\alpha}{\partial g^\alpha}
=-\frac1m\frac{\dot\gamma^\alpha}{g^\alpha},
\]

并包括硬化函数对阻力的导数，组装完整的 \(18\times18\) 局部 Jacobian
\(\boldsymbol J_x=\partial\boldsymbol R/\partial\boldsymbol x\)。对每个 Green–Lagrange 应变基方向求解

\[
\boldsymbol J_x\frac{d\boldsymbol x}{d\boldsymbol E}
=-\frac{\partial\boldsymbol R}{\partial\boldsymbol E}.
\]

若一次全局增量被内部子步拆分，\(d\boldsymbol F_p^{-1}/d\boldsymbol E\) 和
\(dg^\alpha/d\boldsymbol E\) 会沿所有子步逐步传递；最后对
\(J_p\boldsymbol F_p^{-1}\boldsymbol S_e\boldsymbol F_p^{-T}\) 完整求导，包括
\(dJ_p\)。因此交给 MOOSE 的是离散局部算法对应的 \(d\boldsymbol S/d\boldsymbol E\)，不是弹性近似切线，也不是数值扰动切线。

## 4. 参数与数值配置

| 参数 | 值 |
|---|---:|
| 单晶弹性 `C11, C12, C44` | 200000, 120000, 100000 MPa |
| 滑移系 | 12 FCC `{111}<110>` |
| 参考滑移率 | 0.001 s⁻¹ |
| 单晶验证 `m` | 0.05 |
| RVE 性能测试 `m` | 0.10 |
| 初始滑移阻力 | 80 MPa |
| 硬化模量 | 400 MPa |
| 饱和阻力 | 160 MPa |
| 硬化指数 | 1.0 |
| latent hardening ratio | 1.4 |
| 总物理加载时间 | 1 s |
| 宏观加载 | `F11: 1 -> 1.02`, `P22=0`, `F12=F21=0` |
| 平面运动学 | `F33=1`，内部变量仍为完整 3D |
| 并行/预条件 | 4 MPI，ASM overlap 2，子域 LU |
| Newton 容差 | `nl_rel_tol=1e-6`, `nl_abs_tol=1e-10` |
| 局部容差 | absolute `1e-10`, relative `1e-9` |

`m=0.05` 保留在单晶验证中；在多晶屈服转折处它造成持续的全局 continuation。
RVE 使用 `m=0.10` 才能在不放宽局部或全局收敛容差的情况下完成正式矩阵。两者均为率相关 benchmark 参数。

所有 CP 算例起始尝试任务规定的 20 或 40 个等间距最大增量。实际求解使用
`IterationAdaptiveDT`；失败尝试和切回后的接受步均计入 wall time。表中的“目标步数”不能解读为实际接受步数。

## 5. 正确性验证

### 5.1 单晶与切线

- 取向 `(0°,0°,0°)`、20 步：`P11=238.391326 MPa`，累积滑移 `0.0431065`，`det(Fp)=1.00001273`。
- 取向 `(37°,52°,11°)`、20 目标步：`P11=354.853131 MPa`，明显不同于取向 1；原生参考为 `354.852480 MPa`。
- 第一个 0.1% 增量取向 1 得到 `P11=110.164879 MPa`，与 `[100]` 方向无塑性小应变解析模量约 110000 MPa 一致。
- 塑性末步解析算法切线对中心有限差分的相对误差为 `4.7670e-9`。有限差分仅在 `verify_analytic_tangent=true` 时执行。
- PETSc 全局 Jacobian TestHarness 通过；开发期完整 RVE 耦合检查的两个相对 Frobenius 误差为 `2.6595e-8` 和 `5.2844e-8`。

### 5.2 64×64 随机取向 RVE（20 目标步）

| 检查量 | 实测值 |
|---|---:|
| 最终 `F11`, `F22` | 1.02, 0.9817277464 |
| 最终 `P11`, `P22` | 389.111097 MPa, 1.3973e-8 MPa |
| 最终宏观 `|F12|`, `|F21|` | < 2e-18 |
| 平均/最大积分点累积滑移 | 0.0548560 / 0.1253712 |
| 晶粒平均滑移范围 | 0.0122141 – 0.0873891 |
| `det(Fp)` 范围 | 1.0000000003 – 1.0000781364 |
| 周期位移波动最大配对误差 | 4.2188e-15 |
| Hill–Mandel 最大相对功率残差 | 3.1932e-9 |
| 最终塑性耗散率 | 7.56454（模型单位） |

Exodus 已由独立 SciPy NetCDF reader 打开，并逐字段验证。随机取向 RVE 的最终局部
`P11` 极差为 572.36 MPa，说明晶粒间应力和滑移重分布确实存在。

### 5.3 同取向退化

16×16 同取向 20 晶粒算例完整结束。最终波动位移 L2 范数小于 `1.25e-12`；20 个晶粒累积滑移的范围仅 `2e-15`，单元 `P11` 极差为 `1.8849e-10 MPa`，单元 `F11` 极差为 `2.22e-15`。这验证了晶粒分块本身不会制造非均匀响应。
图 1 中三维单晶采用自由横向收缩，而二维 RVE 是 `F33=1` 且宏观 `P22=0`；两条曲线只作定性展示，同取向退化的定量判断使用上述同一 RVE 边界条件结果。

### 5.4 网格与时间增量敏感性

- 20 目标步：16/32/64 的最终 `P11` 分别为 401.2057、392.6909、389.1111 MPa；32→64 差异为 0.91%，16→64 为 3.01%。
- 64×64 的 20 与 40 目标步最终 `P11` 分别为 389.1111 和 389.2156 MPa，相对差异 0.0269%。

## 6. 实测性能

测试机为 AMD Ryzen 7 5800X（8 核/16 线程），Linux 7.0.0-38。以下均使用
`/usr/bin/time -v`，4 MPI；积分点数由实际 QUAD4、2×2 积分确认。

| Case | 网格/积分点 | 目标/接受步 | 失败重试 | 总 Newton | Wall | User | RSS | 最终 P11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CP-0 | 1 HEX8 | 20/20 | 0 | 41 | 0.89 s | 0.52 s | 175 MB | 238.391 MPa |
| CP-1 | 16×16 / 1024 | 20/31 | 4 | 217 | 14.63 s | 55.57 s | 226 MB | 401.206 MPa |
| CP-2 | 32×32 / 4096 | 20/36 | 6 | 286 | 63.95 s | 251.18 s | 232 MB | 392.691 MPa |
| CP-3 | 64×64 / 16384 | 20/45 | 11 | 442 | 365.80 s | 1447.04 s | 272 MB | 389.111 MPa |
| CP-4 | 64×64 / 16384 | 40/61 | 9 | 443 | 373.73 s | 1476.50 s | 273 MB | 389.216 MPa |

“总 Newton”同时包括接受步和失败后切回的尝试；例如 CP-3 为 167 次接受步 Newton
加 275 次失败尝试 Newton，共 442 次。CP-3 没有超时，完整达到 2%。16→32 wall time 增加 4.37 倍，32→64 增加 5.72 倍。
CP-4 虽接受步更多，但 CP-3 的大步尝试产生更多失败重试，因此两者总时间接近。
接受步的线性迭代总数 CP-1/2/3/4 分别为 118/138/169/221；失败尝试不会执行
`TIMESTEP_END` 输出，因此没有把其 KSP 次数伪装成精确统计。失败尝试的 Newton 次数则由日志逐次解析并已计入上表。

### 6.1 超弹性对照

| 网格/配置 | CP wall | 超弹性 wall | `r_CP` |
|---|---:|---:|---:|
| 16×16 / 20 | 14.63 s | 1.58 s | 9.26 |
| 32×32 / 20 | 63.95 s | 3.03 s | 21.11 |
| 64×64 / 20 | 365.80 s | 9.83 s | 37.21 |
| 64×64 / 40 | 373.73 s | 18.74 s | 19.94 |

超弹性最终 `P11≈4181 MPa`，CP 最终 `P11≈389 MPa`，也从宏观响应证明本次不是近弹性性能测试。

### 6.2 时间分解

自定义低开销计时器在每个 MPI rank 累积 CP 更新时间，求和后除以 4 得到 wall-equivalent：

| 网格/20 配置 | 局部 CP wall-equivalent | 总 wall | 占比 |
|---|---:|---:|---:|
| 16×16 | 8.804 s | 14.63 s | 60.18% |
| 32×32 | 43.613 s | 63.95 s | 68.20% |
| 64×64 | 249.951 s | 365.80 s | 68.33% |

MOOSE PerfGraph 的 residual/Jacobian assembly 区域包含 material 调用，属于嵌套计时，不能与局部 CP 时间相加。64×64 的 rank-0 inclusive 统计为 Jacobian assembly 170.04 s、residual assembly 136.84 s、`FEProblem::solve` 339.09 s；这些数字用于定位调用路径，不构成互斥分解。可靠的互斥结论是：约 68% 在局部 CP，余下约 32% 在装配、线性/非线性求解、后处理和输出。

## 7. Bayesian Optimization 粗预算

以完整的 CP-3 实测 365.80 s 为一次 64×64 FEM，串行且不计 GP 开销：

| FEM 次数 | 粗略 wall 预算 |
|---:|---:|
| 30 | 3.05 h |
| 50 | 5.08 h |
| 100 | 10.16 h |

因此 30–50 次在当前工作站上可以接受；100 次仍可运行，但更适合多任务并行、缓存和较低保真度筛选。由于局部积分占约 68%，下一步优化局部算法、向量化或减少重复 material evaluation 有明确收益；全局稀疏求解和 78–104 MB 的 Exodus 输出也是次要目标。

## 8. 交付、复现与状态

统一运行命令：

```bash
cd /home/liu/projects/spp2489
conda activate moose
python scripts/crystal_plasticity/generate_cp_materials.py --mode random
python scripts/crystal_plasticity/run_cp_benchmark.py --kind all --meshes 16 32 64 --steps 20
python scripts/crystal_plasticity/run_cp_benchmark.py --kind all --meshes 64 --steps 40
python scripts/crystal_plasticity/extract_cp_performance.py
python scripts/crystal_plasticity/verify_cp_results.py
python scripts/crystal_plasticity/plot_cp_benchmark.py
./run_tests -j4
```

主要机器可读结果：

- `results/crystal_plasticity/cp_runtime_summary.csv`
- `results/crystal_plasticity/cp_step_statistics.csv`
- `results/crystal_plasticity/cp_performance_breakdown.json`
- `results/crystal_plasticity/cp_verification.json`
- `results/crystal_plasticity/cp_material_parameters.json`
- `results/crystal_plasticity/analytic_cp_64_s20.e`

图：`fig01_cp_stress_strain.png` 到 `fig06_slip_activity.png`，位于
`results/crystal_plasticity/figures/`。正式结果前缀为 `analytic_cp_`；开发期 `cp_*` 和
`debug_*` 文件不纳入性能表。

状态分类：单晶、16/32/64、20/40 配置、超弹性对照、同取向退化、Exodus 字段检查和
TestHarness 均为**实测完成**；没有 timeout、部分完成或以估算冒充实测的条目。只有第 7 节的 BO 总预算明确为线性估算。

## 9. 对十个问题的直接回答

1. **原生 FCC CP 是否成功运行？** 是，单晶原生 Kalidindi 参考运行成功；正式 TL RVE 使用等价 FCC 物理的专用直接 PK2 实现。
2. **是否成功耦合现有 TL homogenization？** 是；直接 PK2→PK1 路径、解析一致切线、周期条件和 Hill–Mandel 均已验证。
3. **32×32 多久？** CP-2 实测 63.95 s，4 MPI。
4. **64×64 多久？** CP-3 实测 365.80 s；40 配置为 373.73 s。
5. **比超弹性慢多少？** 32×32 为 21.11 倍，64×64/20 为 37.21 倍。
6. **是否明显塑性？** 是；64×64 平均/最大累积滑移为 0.054856/0.125371，最终应力仅为弹性对照约 9.3%。
7. **局部积分占比？** 64×64/20 为 68.33%。
8. **速度瓶颈？** 第一瓶颈是局部 18 变量 CP 更新；第二是随网格超线性增长的全局装配和稀疏求解。
9. **30–100 次预算？** 串行约 3.05–10.16 h，未计并行吞吐收益。
10. **值得继续优化局部积分吗？** 值得；解析一致切线已经解决全局 Newton 质量，下一步应针对局部更新的重复调用、数据布局和批处理优化。

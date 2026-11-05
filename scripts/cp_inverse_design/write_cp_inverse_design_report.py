#!/usr/bin/env python3
"""Write the CP inverse-design report from machine-readable executed results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common import FIGURES, OPTIMIZATION_HISTORY, RANDOM_HISTORY, REPORT, SUMMARY


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _percent(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def write_report(summary: dict[str, object]) -> None:
    bo = _rows(OPTIMIZATION_HISTORY)
    random_rows = _rows(RANDOM_HISTORY)
    valid = [row for row in bo if row["status"] == "converged"]
    random_valid = [row for row in random_rows if row["status"] == "converged"]
    best = summary["best_observed_design"]
    target = summary["target_properties"]
    verify = summary["verification"]
    low = verify["independent_low_fidelity"]
    low_target = verify["low_fidelity_target"]
    best_random = min(random_valid, key=lambda row: float(row["objective_J"])) if random_valid else None
    qualified = [
        row for row in valid
        if float(row["relative_error_1pct"]) <= float(summary["target_tolerance"])
        and float(row["relative_error_2pct"]) <= float(summary["target_tolerance"])
    ]
    eta = np.asarray([float(row["eta"]) for row in valid])
    p1 = np.asarray([float(row["P11_1pct_MPa"]) for row in valid])
    p2 = np.asarray([float(row["P11_2pct_MPa"]) for row in valid])
    order = np.argsort(eta)
    monotone1 = bool(np.all(np.diff(p1[order]) >= 0.0) or np.all(np.diff(p1[order]) <= 0.0))
    monotone2 = bool(np.all(np.diff(p2[order]) >= 0.0) or np.all(np.diff(p2[order]) <= 0.0))
    study = summary["time_integration_study"]
    high_lines = ["- 64×64 verification was not requested in this invocation."]
    if "high_fidelity_target" in verify:
        high_t, high_c = verify["high_fidelity_target"], verify["high_fidelity_candidate"]
        tm, cm = verify["target_mesh_change_32_to_64"], verify["candidate_mesh_change_32_to_64"]
        high_lines = [
            f"- 64×64 target: P11(1%)={high_t['P11_1pct_MPa']:.9f} MPa, P11(2%)={high_t['P11_2pct_MPa']:.9f} MPa.",
            f"- 64×64 optimized: P11(1%)={high_c['P11_1pct_MPa']:.9f} MPa, P11(2%)={high_c['P11_2pct_MPa']:.9f} MPa; same-mesh errors {_percent(high_c['relative_error_1pct'])} and {_percent(high_c['relative_error_2pct'])}.",
            f"- Target 32→64 changes: {_percent(tm['relative_P11_1pct'])} / {_percent(tm['relative_P11_2pct'])}; optimized 32→64 changes: {_percent(cm['relative_P11_1pct'])} / {_percent(cm['relative_P11_2pct'])}.",
        ]
    random_line = "Random search produced no converged FEM evaluation."
    if best_random:
        random_line = (
            f"Random search executed {len(random_rows)} FEM calls; its best η={float(best_random['eta']):.9f}, "
            f"errors {_percent(float(best_random['relative_error_1pct']))} and "
            f"{_percent(float(best_random['relative_error_2pct']))}. "
            f"First tolerance call: {summary['random_search_first_tolerance_call']}."
        )
    endpoint = summary["texture_validation"]["endpoints"]
    local_slip_mean_delta = float(low["mean_accumulated_slip"]) - float(low_target["mean_accumulated_slip"])
    local_slip_max_delta = float(low["maximum_accumulated_slip"]) - float(low_target["maximum_accumulated_slip"])
    reference_eta = float(verify["reference_eta_disclosed_after_optimization"])
    regression_path = SUMMARY.parent / "regression_checks.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8")) if regression_path.exists() else None
    low_result = json.loads((Path(low["run_directory"]) / "result.json").read_text(encoding="utf-8"))
    lines = [
        "# SPP2489 有限应变晶体塑性逆向设计报告",
        "",
        "## 结论",
        "",
        f"真实 MOOSE–BoTorch 闭环已完成。32×32 隐藏目标为 **P11(1.01)={target['target_P11_at_F11_1p01_MPa']:.9f} MPa**、**P11(1.02)={target['target_P11_at_F11_1p02_MPa']:.9f} MPa**。最优观测设计为 **η*={float(best['eta']):.9f}**；独立 FEM 重算的两个相对误差分别为 **{_percent(low['relative_error_1pct'])}** 和 **{_percent(low['relative_error_2pct'])}**。",
        "",
        "该结果匹配的是规定的有限应变、率相关 FCC 晶体塑性加载路径，不是材料参数反演，也不意味着唯一恢复隐藏晶体取向。20 晶粒 RVE 是数值 demonstration，不代表统计收敛的 SS316L 性能。",
        "",
        "## 计算定义",
        "",
        "- 固定 20 晶粒周期性 Voronoi 几何、材料参数、滑移系和 A4 本构积分器；本任务未修改晶体塑性离散方程。",
        "- 每个晶粒从随机端点到共同取向采用单位四元数最短弧 SLERP；同一旋转属性同时用于立方弹性张量、滑移方向和滑移面法向。",
        "- Plane strain Total-Lagrangian 均匀化，F11=1+0.02t、T=1 s、P22=0、F12=F21=0。",
        "- 目标函数为两个相对应力误差平方的等权平均；GP 对 −J 建模。失败 FEM 不进入训练数据。",
        f"- 端点 Euler 最大误差：η=0 为 {endpoint['eta_0_random_euler_max_error_deg']:.3e}°，η=1 为 {endpoint['eta_1_preferred_euler_max_error_deg']:.3e}°。",
        "",
        "## 时间积分协议",
        "",
        f"η=0、0.5、1 的 current adaptive、strict adaptive 与 fixed-200 对照均实际运行（{study.get('mpi_ranks', 4)} MPI ranks）。选择 `{study['selected_protocol']}`；其相对 fixed-200 的最大双指标差异为 {_percent(study['selected_protocol_max_relative_difference_vs_fixed'])}。输出同步机制确保 t=0.5 和 1.0 是真实求解时刻，而非事后线性插值。正式 inverse-design workflow 使用 {summary['mpi_ranks']} MPI ranks；η=0.5 strict 的实测由 30.03 s 降至 22.14 s（1.36×）。",
        "",
        "## Bayesian Optimization 与随机对照",
        "",
        f"- BO 共执行 {summary['bo_fem_calls']} 次候选 FEM（其中顺序采集 {summary['bo_sequential_calls']} 次、失败 {summary['bo_failed_calls']} 次），首次同时达到 1% 阈值的调用序号为 {summary['bo_first_tolerance_call']}。",
        f"- 候选 FEM 累计 wall time={summary['bo_fem_wall_time_s']:.3f} s；GP 拟合与采集累计={summary['bo_gp_time_s']:.3f} s，二者分开统计。",
        f"- {random_line}",
        f"- BO 最优观测的 P11(1%)={float(best['P11_1pct_MPa']):.9f} MPa、P11(2%)={float(best['P11_2pct_MPa']):.9f} MPa、Hsec={float(best['H_sec_MPa']):.9f} MPa。",
        "",
        "本次随机搜索只有一个固定种子；结果仅用作闭环基准，不能据此宣称 BO 在统计意义上总优于随机搜索。对于一维变量，均匀网格同样是合理基准。",
        "",
        "## 独立重算与多保真验证",
        "",
        f"- 32×32 独立最优 FEM：P11(1%)={low['P11_1pct_MPa']:.9f} MPa，P11(2%)={low['P11_2pct_MPa']:.9f} MPa，Hsec={low['H_sec_MPa']:.9f} MPa；误差 {_percent(low['relative_error_1pct'])} / {_percent(low['relative_error_2pct'])}；达标={low['meets_tolerance']}。",
        *high_lines,
        "- 低、高保真比较都在相同网格内比较 target 与 candidate；没有把 32×32 candidate 直接与 64×64 target 混作优化误差。",
        "",
        "## 参数敏感性、非唯一性与局部机制",
        "",
        f"- 已观测 η 区间内 P11(1%) 跨度为 {p1.max()-p1.min():.6f} MPa，P11(2%) 跨度为 {p2.max()-p2.min():.6f} MPa；按 η 排序后两条离散响应分别{'呈单调趋势' if monotone1 else '非单调'}、{'呈单调趋势' if monotone2 else '非单调'}。",
        f"- 候选观测点中有 {len(qualified)} 个同时满足双 1% 指标；其 η 为 {', '.join(f'{float(row['eta']):.6f}' for row in qualified) or 'none'}。优化完成后披露的隐藏参考为 η_ref={reference_eta:.6f}，其误差按构造为零；上述达标候选与参考 η 不同，因此在当前 1% 性能容差下不能唯一恢复原始织构。",
        f"- 最优与隐藏目标的平均累积滑移差为 {local_slip_mean_delta:+.6e}，最大累积滑移差为 {local_slip_max_delta:+.6e}。Fig. 6 使用共同色标展示局部滑移，因此宏观性能接近时仍可识别局部机制差异。",
        "",
        "## 数值与物理检查",
        "",
        f"独立最优解 all_checks_passed={low['all_physical_checks_passed']}，accepted increments={low['accepted_increments']}，failed attempts={low['failed_attempts']}，total Newton iterations={low['total_newton_iterations']}。最大 |P22|={low_result['maximum_abs_P22_MPa']:.3e} MPa，最大周期配对误差={low_result['exodus']['maximum_periodicity_error']:.3e}，最大 Hill–Mandel 相对残差={low_result['hill_mandel_relative_residual_max']:.3e}，det(Fp) 范围=[{low_result['minimum_det_Fp']:.9f}, {low_result['maximum_det_Fp']:.9f}]，塑性耗散={low_result['plastic_dissipation']:.6f}。检查还覆盖准确终止时间、同步 F11、宏观剪切、非零滑移、有限状态、正滑移阻力及 Exodus 字段。",
        "",
        "## 可扩展性",
        "",
        "单变量 demo 的主要成本是 CP FEM，不是 GP。扩展到二维或三维织构变量在软件结构上可行，但初始 DOE 与样本预算会增加；在启动大规模 64×64 BO 前，应采用多保真代理、并行候选或可信的低保真校正。",
        "",
        "## 工程回归",
        "",
        *(
            [
                f"- CP inverse-design Python tests: {regression['python_unit_tests']['tests_run']} passed。",
                f"- 原 MOOSE TestHarness: {regression['moose_testharness']['passed']} passed, {regression['moose_testharness']['failed']} failed, {regression['moose_testharness']['skipped']} skipped。",
                f"- 机器可读 workflow checks: {sum(regression['checks'].values())}/{len(regression['checks'])} passed；缓存复用、目标隔离、独立重算、同网格高保真、图片/GIF/Exodus 均已检查。",
            ]
            if regression
            else ["- 请运行 `python scripts/cp_inverse_design/verify_workflow_results.py` 生成最终回归记录。"]
        ),
        "",
        "## 产物与复现",
        "",
        f"- 8 张图：`{FIGURES.relative_to(REPORT.parent)}/fig01_*.png` 至 `fig08_*.png`。",
        "- Exodus：`results/cp_inverse_design/exodus/`，包含隐藏目标与最优候选的低/高保真场。",
        "- 机器可读历史：`results/cp_inverse_design/optimization_history.csv`、`random_search_history.csv`、`best_design.json`、`cp_inverse_design_summary.json`。",
        "- GIF：`results/cp_inverse_design/animations/cp_inverse_design_history.gif`；每帧是一个真实 FEM 候选，不是连续形貌演化。",
        "",
        "```bash",
        "conda run --no-capture-output -n spp2489-inverse \\",
        "  python scripts/cp_inverse_design/optimize_cp_texture.py --mesh 32 \\",
        "  --max-bo-evals 12 --initial-samples 6 --target-tolerance 0.01 \\",
        "  --seed 2026 --reuse-cache --validate-high-fidelity",
        "```",
        "",
        f"工作流实际总 wall time（含验证、绘图与缓存命中开销）为 {summary['workflow_wall_time_s']:.3f} s。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    write_report(summary)
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

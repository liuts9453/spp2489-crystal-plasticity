#!/usr/bin/env python3
"""Fit positive grain-size and constrained aspect-ratio distributions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp


def _information_criteria(
    log_likelihood: float, count: int, parameters: int
) -> tuple[float, float]:
    return (
        2.0 * parameters - 2.0 * log_likelihood,
        parameters * math.log(count) - 2.0 * log_likelihood,
    )


def _single_fit(values: np.ndarray, name: str) -> dict[str, object]:
    distributions = {
        "lognormal": stats.lognorm,
        "weibull": stats.weibull_min,
        "gamma": stats.gamma,
    }
    distribution = distributions[name]
    parameters = distribution.fit(values, floc=0.0)
    log_pdf = distribution.logpdf(values, *parameters)
    log_likelihood = float(np.sum(log_pdf))
    aic, bic = _information_criteria(log_likelihood, len(values), len(parameters) - 1)
    ks = stats.kstest(values, distribution.cdf, args=parameters)
    empirical_quantiles = np.quantile(values, (0.95, 0.99))
    fitted_quantiles = distribution.ppf((0.95, 0.99), *parameters)
    return {
        "model": name,
        "scipy_parameters": [float(value) for value in parameters],
        "parameter_order": "shape(s), loc, scale",
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "q95_relative_error": float(
            (fitted_quantiles[0] - empirical_quantiles[0]) / empirical_quantiles[0]
        ),
        "q99_relative_error": float(
            (fitted_quantiles[1] - empirical_quantiles[1]) / empirical_quantiles[1]
        ),
    }


def _lognormal_mixture(values: np.ndarray) -> dict[str, object]:
    transformed = np.log(values)
    means = np.quantile(transformed, (0.30, 0.70))
    sigmas = np.full(2, max(float(np.std(transformed)), 1.0e-4))
    weights = np.full(2, 0.5)
    previous = -np.inf
    for _ in range(500):
        component_log_probability = np.column_stack(
            [
                np.log(weights[index])
                + stats.norm.logpdf(transformed, means[index], sigmas[index])
                for index in range(2)
            ]
        )
        normalization = logsumexp(component_log_probability, axis=1)
        responsibilities = np.exp(component_log_probability - normalization[:, None])
        effective = np.sum(responsibilities, axis=0)
        weights = effective / len(transformed)
        means = np.sum(responsibilities * transformed[:, None], axis=0) / effective
        variances = (
            np.sum(responsibilities * (transformed[:, None] - means) ** 2, axis=0)
            / effective
        )
        sigmas = np.sqrt(np.maximum(variances, 1.0e-8))
        current = float(np.sum(normalization))
        if abs(current - previous) <= 1.0e-10 * (1.0 + abs(current)):
            break
        previous = current
    order = np.argsort(means)
    weights = weights[order]
    sigmas = sigmas[order]
    means = means[order]
    component_log_probability = np.column_stack(
        [
            np.log(weights[index])
            + stats.norm.logpdf(transformed, means[index], sigmas[index])
            for index in range(2)
        ]
    )
    log_likelihood = float(
        np.sum(logsumexp(component_log_probability, axis=1)) - np.sum(np.log(values))
    )
    # Two weights sum to one: 2 means + 2 variances + 1 free weight.
    aic, bic = _information_criteria(log_likelihood, len(values), 5)
    grid = np.sort(values)
    cdf = sum(
        weight * stats.lognorm.cdf(grid, sigma, loc=0.0, scale=np.exp(mean))
        for weight, mean, sigma in zip(weights, means, sigmas)
    )
    empirical = np.arange(1, len(grid) + 1, dtype=float) / len(grid)
    return {
        "model": "two_component_lognormal_mixture",
        "weights": [float(value) for value in weights],
        "log_means": [float(value) for value in means],
        "log_standard_deviations": [float(value) for value in sigmas],
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "cdf_supremum_discrepancy": float(np.max(np.abs(cdf - empirical))),
    }


def fit_positive_distributions(values: np.ndarray) -> dict[str, object]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 10 or not np.all(np.isfinite(data)):
        raise ValueError(
            "Distribution fitting requires at least ten finite scalar values"
        )
    if np.any(data <= 0.0):
        raise ValueError("Positive distributions cannot be fit to nonpositive values")
    candidates = [_single_fit(data, name) for name in ("lognormal", "weibull", "gamma")]
    candidates.append(_lognormal_mixture(data))
    ranking = sorted(candidates, key=lambda item: float(item["bic"]))
    return {
        "sample_count": int(len(data)),
        "candidates": candidates,
        "ranking_by_bic": [str(item["model"]) for item in ranking],
        "selected_by_bic": ranking[0],
        "selection_warning": (
            "BIC is one diagnostic; CDF and q95/q99 tail errors must also be reviewed "
            "before selecting a Neper target distribution."
        ),
    }


def fit_grain_distributions(frame: pd.DataFrame) -> dict[str, object]:
    ecd = frame["equivalent_diameter_um"].to_numpy(dtype=float)
    aspect = frame["aspect_ratio_recomputed"].to_numpy(dtype=float)
    # AR >= 1. Fit the strictly positive excess and keep the physical shift explicit.
    excess = np.maximum(aspect - 1.0, np.finfo(float).eps)
    return {
        "equivalent_diameter_um": fit_positive_distributions(ecd),
        "aspect_ratio": {
            "support": "AR >= 1",
            "transformation": "z = AR - 1; positive distributions are fit to z",
            "excess_over_one": fit_positive_distributions(excess),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.cleaned_csv)
    result = fit_grain_distributions(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
VALIDATION_DIR = REPO_ROOT / "validation"
OUTPUT_DIR = REPO_ROOT / "public_reproduction_outputs"

DEG2_ORDER = ["C20", "C21", "S21", "C22", "S22"]
MODEL_ORDER = [
    "S40RTS",
    "TX2011",
    "SEISGLOB2",
    "SPani",
    "SEMUCB_WM1",
    "CSEM2_Globe_RHO",
    "CSEM2_Globe_VSV",
    "UNICA25_absolute_Vp",
]
PRIMARY_BRANCH = "R_compatible_central"
SENSITIVITY_BRANCH = "R_anomaly_sensitivity"
STAT_BRANCHES = [PRIMARY_BRANCH, SENSITIVITY_BRANCH]
PRIMARY_SUBSET = "A_full_mantle"
RANDOM_SEED = 42
N_MONTE_CARLO = 10_000
BETA_Z_INDETERMINATE = 2.0

COVARIANCE_TREATMENT = (
    "reference_figure_residual formal diagonal covariance from public degree2_gravity_product_summary.csv; "
    "compatible correction vectors held fixed because no correction covariance is documented"
)
CORRECTION_UNCERTAINTY_TREATMENT = (
    "compatible correction vectors held fixed; no documented correction covariance available in public inputs"
)

PUBLIC_INPUTS = {
    "degree2_gravity_product_summary": DATA_DIR / "degree2_gravity_product_summary.csv",
    "mantle_model_degree2_predictions": DATA_DIR / "mantle_model_degree2_predictions.csv",
    "compatible_residual_coefficients": DATA_DIR / "compatible_residual_coefficients.csv",
    "compatible_residual_definitions": DATA_DIR / "compatible_residual_definitions.csv",
    "compatible_reference_state_registry": DATA_DIR / "compatible_reference_state_registry.csv",
    "signed_beta_significance": VALIDATION_DIR / "signed_beta_significance.csv",
    "statistical_robustness_summary": VALIDATION_DIR / "statistical_robustness_summary.csv",
    "monte_carlo_sign_confidence": VALIDATION_DIR / "monte_carlo_sign_confidence.csv",
    "constrained_sign_hypothesis_comparison": VALIDATION_DIR / "constrained_sign_hypothesis_comparison.csv",
    "delta_chi_square_by_model": VALIDATION_DIR / "delta_chi_square_by_model.csv",
}

OUTPUT_FILES = [
    "sign_test_model_summary.csv",
    "sign_test_monte_carlo_summary.csv",
    "sign_test_sensitivity_summary.csv",
    "sign_test_validation.csv",
    "public_input_hashes.csv",
    "run_manifest.json",
]

MODEL_METADATA = {
    "S40RTS": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "TX2011": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "SEISGLOB2": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "SPani": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "SEMUCB_WM1": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "CSEM2_Globe_RHO": {
        "input_category": "native density-related",
        "plain_language_sign_interpretation": (
            "negative beta supports a global reversal of the native density-related degree-2 field; positive beta "
            "supports retaining the native field sign"
        ),
    },
    "CSEM2_Globe_VSV": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
    "UNICA25_absolute_Vp": {
        "input_category": "velocity-based",
        "plain_language_sign_interpretation": (
            "negative beta supports a globally reversed velocity-to-density proxy, interpreted as slow-as-light "
            "at degree 2; positive beta supports the adopted slow-as-dense proxy"
        ),
    },
}


@dataclass(frozen=True)
class PublicConfig:
    output_dir: Path = OUTPUT_DIR
    primary_branch: str = PRIMARY_BRANCH
    primary_subset: str = PRIMARY_SUBSET
    random_seed: int = RANDOM_SEED
    monte_carlo_count: int = N_MONTE_CARLO


def ensure_output_dir(config: PublicConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g")


def normal_cdf(x: float, mean: float, standard_deviation: float) -> float:
    z = (x - mean) / standard_deviation
    return float(0.5 * math.erfc(-z / math.sqrt(2.0)))


def coeff_vector(values: Any, name: str = "coefficient vector") -> np.ndarray:
    if isinstance(values, pd.Series):
        if all(k in values.index for k in DEG2_ORDER):
            arr = values[DEG2_ORDER].to_numpy(dtype=float)
        else:
            arr = values.to_numpy(dtype=float).reshape(-1)
    elif isinstance(values, dict):
        arr = np.array([values[k] for k in DEG2_ORDER], dtype=float)
    else:
        arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (5,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain five finite values in {DEG2_ORDER} order.")
    return arr


def validate_public_inputs_present() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in PUBLIC_INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing public input(s): " + ", ".join(missing))


def load_public_tables() -> dict[str, pd.DataFrame]:
    validate_public_inputs_present()
    return {input_id: pd.read_csv(path) for input_id, path in PUBLIC_INPUTS.items()}


def write_public_input_hashes(config: PublicConfig) -> pd.DataFrame:
    rows = [
        {
            "input_id": input_id,
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(path),
        }
        for input_id, path in PUBLIC_INPUTS.items()
    ]
    hashes = pd.DataFrame(rows)
    write_csv(hashes, config.output_dir / "public_input_hashes.csv")
    return hashes


def load_covariance(gravity_product_summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row = gravity_product_summary.loc[
        gravity_product_summary["product_id"] == "reference_figure_residual"
    ]
    if len(row) != 1:
        raise RuntimeError("Could not uniquely identify reference_figure_residual in the public gravity summary.")
    row = row.iloc[0]
    sigma = np.array([float(row[f"sigma_{k}"]) for k in DEG2_ORDER], dtype=float)
    if sigma.shape != (5,) or not np.all(np.isfinite(sigma)) or not np.all(sigma > 0):
        raise RuntimeError("Documented diagonal covariance is unavailable or invalid.")
    return np.diag(sigma**2), sigma


def vector_by_branch(residual_definitions: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(row["branch_id"]): coeff_vector(row[DEG2_ORDER], str(row["branch_id"]))
        for _, row in residual_definitions.iterrows()
    }


def adopted_prediction_rows(predictions: pd.DataFrame, subset: str) -> pd.DataFrame:
    available = predictions["available"].astype(str).str.lower().isin(["true", "1", "yes"])
    return predictions[
        available
        & (predictions["case_id"] == subset)
        & predictions["sign_hypothesis"].fillna("").str.contains("adopted", case=False)
    ].copy()


def model_prediction_vector(predictions: pd.DataFrame, model_key: str, subset: str) -> np.ndarray:
    rows = adopted_prediction_rows(predictions, subset)
    rows = rows[rows["model_key"] == model_key]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one adopted prediction for {model_key}/{subset}, found {len(rows)}.")
    return rows.iloc[0][[f"g_model_{k}" for k in DEG2_ORDER]].to_numpy(dtype=float)


def fit_signed_beta(d: np.ndarray, g: np.ndarray, weight: np.ndarray) -> dict[str, float | str]:
    d = coeff_vector(d, "observed gravity vector")
    g = coeff_vector(g, "model prediction vector")
    denominator = float(g @ weight @ g)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Non-positive weighted denominator in signed beta fit.")
    numerator = float(g @ weight @ d)
    beta = float(numerator / denominator)
    standard_error = float(math.sqrt(1.0 / denominator))
    fitted = beta * g
    residual = d - fitted
    chi_fit = float(residual @ weight @ residual)
    chi_null = float(d @ weight @ d)
    ci_low = beta - 1.96 * standard_error
    ci_high = beta + 1.96 * standard_error
    if abs(beta) <= BETA_Z_INDETERMINATE * standard_error or ci_low <= 0 <= ci_high:
        preferred_sign = "indeterminate"
    elif beta > 0:
        preferred_sign = "adopted"
    else:
        preferred_sign = "reversed"
    return {
        "numerator": numerator,
        "denominator": denominator,
        "beta": beta,
        "beta_sigma": standard_error,
        "beta_ci_low": ci_low,
        "beta_ci_high": ci_high,
        "beta_z_score": beta / standard_error,
        "p_beta_less_zero_analytic": normal_cdf(0.0, beta, standard_error),
        "weighted_misfit_chi2": chi_fit,
        "weighted_null_chi2": chi_null,
        "weighted_misfit_norm": math.sqrt(max(chi_fit, 0.0)),
        "improvement_over_null": float(1.0 - chi_fit / chi_null) if chi_null > 0 else np.nan,
        "preferred_sign": preferred_sign,
    }


def constrained_sign_residuals(d: np.ndarray, g: np.ndarray, weight: np.ndarray) -> dict[str, float | str]:
    fit = fit_signed_beta(d, g, weight)
    beta = float(fit["beta"])
    beta_adopted = max(0.0, beta)
    beta_reversed = max(0.0, -beta)
    chi_adopted = float((d - beta_adopted * g) @ weight @ (d - beta_adopted * g))
    chi_reversed = float((d + beta_reversed * g) @ weight @ (d + beta_reversed * g))
    delta = chi_adopted - chi_reversed
    return {
        "adopted_beta_nonnegative": beta_adopted,
        "reversed_beta_nonnegative": beta_reversed,
        "adopted_sign_weighted_residual_chi2": chi_adopted,
        "reversed_sign_weighted_residual_chi2": chi_reversed,
        "adopted_sign_weighted_residual_norm": math.sqrt(max(chi_adopted, 0.0)),
        "reversed_sign_weighted_residual_norm": math.sqrt(max(chi_reversed, 0.0)),
        "delta_chi2_adopted_minus_reversed": delta,
        "favored_constrained_sign": "reversed" if delta > 0 else ("adopted" if delta < 0 else "tie"),
    }


def plain_language_result(model_key: str, preferred_sign: str) -> str:
    category = MODEL_METADATA[model_key]["input_category"]
    if preferred_sign == "indeterminate":
        return "the public coefficient test is indeterminate for this model"
    if category == "native density-related":
        if preferred_sign == "reversed":
            return "observed gravity favors globally reversing the native density-related degree-2 field"
        return "observed gravity favors retaining the native density-related degree-2 field sign"
    if preferred_sign == "reversed":
        return "observed gravity favors the reversed velocity-based proxy, interpreted as slow-as-light at degree 2"
    return "observed gravity favors the adopted velocity-based proxy, interpreted as slow-as-dense at degree 2"


def degree2_to_trace_free_tensor(coeffs: Iterable[float]) -> np.ndarray:
    c20, c21, s21, c22, s22 = coeff_vector(coeffs, "degree-2 coefficients")
    sqrt5 = math.sqrt(5.0)
    sqrt15 = math.sqrt(15.0)
    qzz = -(2.0 * sqrt5 / 3.0) * c20
    qxz = -(sqrt15 / 3.0) * c21
    qyz = -(sqrt15 / 3.0) * s21
    qxy = -(sqrt15 / 3.0) * s22
    qxx_minus_qyy = -(2.0 * sqrt15 / 3.0) * c22
    qxx = (-qzz + qxx_minus_qyy) / 2.0
    qyy = (-qzz - qxx_minus_qyy) / 2.0
    return np.array([[qxx, qxy, qxz], [qxy, qyy, qyz], [qxz, qyz, qzz]], dtype=float)


def axial_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def role_swap_check(g: np.ndarray) -> dict[str, float | bool]:
    vals_a, vecs_a = np.linalg.eigh(degree2_to_trace_free_tensor(g))
    vals_b, vecs_b = np.linalg.eigh(degree2_to_trace_free_tensor(-coeff_vector(g)))
    min_to_reversed_max = axial_angle_deg(vecs_a[:, 0], vecs_b[:, 2])
    mid_to_reversed_mid = axial_angle_deg(vecs_a[:, 1], vecs_b[:, 1])
    max_to_reversed_min = axial_angle_deg(vecs_a[:, 2], vecs_b[:, 0])
    return {
        "min_to_reversed_max_angle_deg": min_to_reversed_max,
        "mid_to_reversed_mid_angle_deg": mid_to_reversed_mid,
        "max_to_reversed_min_angle_deg": max_to_reversed_min,
        "eigenvectors_preserved": bool(max(min_to_reversed_max, mid_to_reversed_mid, max_to_reversed_min) < 1e-5),
        "min_max_roles_swapped": bool(np.allclose(vals_b, -vals_a[::-1])),
    }


def deterministic_summary(
    config: PublicConfig,
    tables: dict[str, pd.DataFrame],
    weight: np.ndarray,
) -> pd.DataFrame:
    d = vector_by_branch(tables["compatible_residual_definitions"])[config.primary_branch]
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_ORDER:
        metadata = MODEL_METADATA[model_key]
        g = model_prediction_vector(tables["mantle_model_degree2_predictions"], model_key, config.primary_subset)
        fit = fit_signed_beta(d, g, weight)
        signs = constrained_sign_residuals(d, g, weight)
        preferred_sign = str(fit["preferred_sign"])
        rows.append(
            {
                "branch_id": config.primary_branch,
                "subset": config.primary_subset,
                "model_key": model_key,
                "input_category": metadata["input_category"],
                "plain_language_sign_interpretation": metadata["plain_language_sign_interpretation"],
                "plain_language_result": plain_language_result(model_key, preferred_sign),
                "beta": fit["beta"],
                "beta_sigma": fit["beta_sigma"],
                "beta_ci_low": fit["beta_ci_low"],
                "beta_ci_high": fit["beta_ci_high"],
                "beta_z_score": fit["beta_z_score"],
                "p_beta_less_zero_analytic": fit["p_beta_less_zero_analytic"],
                "preferred_sign": preferred_sign,
                "weighted_misfit_chi2": fit["weighted_misfit_chi2"],
                "weighted_null_chi2": fit["weighted_null_chi2"],
                "improvement_over_null": fit["improvement_over_null"],
                "adopted_beta_nonnegative": signs["adopted_beta_nonnegative"],
                "reversed_beta_nonnegative": signs["reversed_beta_nonnegative"],
                "adopted_sign_weighted_residual_chi2": signs["adopted_sign_weighted_residual_chi2"],
                "reversed_sign_weighted_residual_chi2": signs["reversed_sign_weighted_residual_chi2"],
                "adopted_sign_weighted_residual_norm": signs["adopted_sign_weighted_residual_norm"],
                "reversed_sign_weighted_residual_norm": signs["reversed_sign_weighted_residual_norm"],
                "delta_chi2_adopted_minus_reversed": signs["delta_chi2_adopted_minus_reversed"],
                "favored_constrained_sign": signs["favored_constrained_sign"],
                "coefficient_order": ",".join(DEG2_ORDER),
                "covariance_treatment": COVARIANCE_TREATMENT,
                "target_proximity_used_for_sign": False,
            }
        )
    return pd.DataFrame(rows)


def draw_branch_samples(
    config: PublicConfig,
    tables: dict[str, pd.DataFrame],
    cov: np.ndarray,
) -> dict[str, np.ndarray]:
    branches = vector_by_branch(tables["compatible_residual_definitions"])
    rng = np.random.default_rng(config.random_seed)
    return {
        branch: rng.multivariate_normal(branches[branch], cov, size=config.monte_carlo_count)
        for branch in STAT_BRANCHES
    }


def monte_carlo_summary(
    config: PublicConfig,
    tables: dict[str, pd.DataFrame],
    weight: np.ndarray,
    branch_samples: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    sample_by_model: dict[str, np.ndarray] = {}
    for model_key in MODEL_ORDER:
        g = model_prediction_vector(tables["mantle_model_degree2_predictions"], model_key, config.primary_subset)
        denominator = float(g @ weight @ g)
        samples = (branch_samples[config.primary_branch] @ weight @ g) / denominator
        sample_by_model[model_key] = samples
        fraction_negative = float(np.mean(samples < 0))
        fraction_positive = float(np.mean(samples > 0))
        q2_5, q5, q16, q50, q84, q95, q97_5 = np.quantile(
            samples, [0.025, 0.05, 0.16, 0.50, 0.84, 0.95, 0.975]
        )
        if fraction_negative >= 0.95:
            robustness = "robust_reversed_beta_negative_ge_95pct"
        elif fraction_positive >= 0.95:
            robustness = "robust_adopted_beta_positive_ge_95pct"
        else:
            robustness = "mixed_or_indeterminate"
        rows.append(
            {
                "branch_id": config.primary_branch,
                "subset": config.primary_subset,
                "model_key": model_key,
                "n_realizations": config.monte_carlo_count,
                "fraction_beta_lt_0": fraction_negative,
                "fraction_beta_gt_0": fraction_positive,
                "median_beta": q50,
                "p05_beta": q5,
                "p95_beta": q95,
                "q2_5_beta": q2_5,
                "q16_beta": q16,
                "q84_beta": q84,
                "q97_5_beta": q97_5,
                "robustness_classification": robustness,
                "uncertainty_model": COVARIANCE_TREATMENT,
                "correction_uncertainty_treatment": CORRECTION_UNCERTAINTY_TREATMENT,
            }
        )
    return pd.DataFrame(rows), sample_by_model


def sign_from_beta(beta: float, sigma: float) -> str:
    ci_low = beta - 1.96 * sigma
    ci_high = beta + 1.96 * sigma
    if abs(beta) <= BETA_Z_INDETERMINATE * sigma or ci_low <= 0 <= ci_high:
        return "indeterminate"
    return "adopted" if beta > 0 else "reversed"


def sensitivity_summary(
    config: PublicConfig,
    tables: dict[str, pd.DataFrame],
    cov: np.ndarray,
    weight: np.ndarray,
) -> pd.DataFrame:
    branches = vector_by_branch(tables["compatible_residual_definitions"])
    predictions = tables["mantle_model_degree2_predictions"]
    rows: list[dict[str, Any]] = []

    for branch_id, d in branches.items():
        signs = [
            str(fit_signed_beta(d, model_prediction_vector(predictions, model, config.primary_subset), weight)["preferred_sign"])
            for model in MODEL_ORDER
        ]
        rows.append(
            {
                "sensitivity_id": "gravity_residual_case",
                "case": branch_id,
                "status": "computed",
                "n_reversed": signs.count("reversed"),
                "n_adopted": signs.count("adopted"),
                "n_indeterminate": signs.count("indeterminate"),
                "details": ";".join(f"{model}:{sign}" for model, sign in zip(MODEL_ORDER, signs)),
            }
        )

    d_primary = branches[config.primary_branch]
    for omitted in DEG2_ORDER:
        keep = [i for i, coeff in enumerate(DEG2_ORDER) if coeff != omitted]
        w_sub = np.linalg.pinv(cov[np.ix_(keep, keep)])
        signs = []
        for model in MODEL_ORDER:
            g = model_prediction_vector(predictions, model, config.primary_subset)
            denominator = float(g[keep] @ w_sub @ g[keep])
            beta = float((g[keep] @ w_sub @ d_primary[keep]) / denominator)
            sigma = float(math.sqrt(1.0 / denominator))
            signs.append(sign_from_beta(beta, sigma))
        rows.append(
            {
                "sensitivity_id": "leave_one_coefficient_out",
                "case": f"omit_{omitted}",
                "status": "computed",
                "n_reversed": signs.count("reversed"),
                "n_adopted": signs.count("adopted"),
                "n_indeterminate": signs.count("indeterminate"),
                "details": ";".join(f"{model}:{sign}" for model, sign in zip(MODEL_ORDER, signs)),
            }
        )

    normalization_passed = True
    sign_identity_passed = True
    for model in MODEL_ORDER:
        g = model_prediction_vector(predictions, model, config.primary_subset)
        base = fit_signed_beta(d_primary, g, weight)
        for scale in [0.1, 10.0]:
            scaled = fit_signed_beta(d_primary, scale * g, weight)
            if str(base["preferred_sign"]) != str(scaled["preferred_sign"]):
                normalization_passed = False
            if not np.allclose(
                float(base["beta"]) * g,
                float(scaled["beta"]) * scale * g,
                rtol=1e-12,
                atol=1e-18,
            ):
                normalization_passed = False
        reversed_fit = fit_signed_beta(d_primary, -g, weight)
        if not np.isclose(float(reversed_fit["beta"]), -float(base["beta"]), rtol=1e-12, atol=1e-18):
            sign_identity_passed = False

    rows.extend(
        [
            {
                "sensitivity_id": "positive_normalization_invariance",
                "case": "scale_model_prediction_by_0p1_and_10",
                "status": "passed" if normalization_passed else "failed",
                "n_reversed": np.nan,
                "n_adopted": np.nan,
                "n_indeterminate": np.nan,
                "details": "positive scaling changes beta magnitude inversely but not fitted vector or sign",
            },
            {
                "sensitivity_id": "global_sign_reversal_identity",
                "case": "g_to_minus_g",
                "status": "passed" if sign_identity_passed else "failed",
                "n_reversed": np.nan,
                "n_adopted": np.nan,
                "n_indeterminate": np.nan,
                "details": "unconstrained beta changes sign under global model sign reversal",
            },
        ]
    )
    return pd.DataFrame(rows)


def max_abs_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> float:
    out = 0.0
    for left_col, right_col in pairs:
        diff = np.abs(left[left_col].to_numpy(dtype=float) - right[right_col].to_numpy(dtype=float))
        out = max(out, float(np.nanmax(diff)))
    return out


def pair_max_abs_diffs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for left_col, right_col in pairs:
        diff = np.abs(left[left_col].to_numpy(dtype=float) - right[right_col].to_numpy(dtype=float))
        diffs[left_col] = float(np.nanmax(diff))
    return diffs


def columns_allclose(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rtol: float = 1e-12,
    atol: float = 4.0,
) -> bool:
    return all(
        bool(
            np.allclose(
                left[left_col].to_numpy(dtype=float),
                right[right_col].to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            )
        )
        for left_col, right_col in pairs
    )


def primary_archive_rows(df: pd.DataFrame, branch_col: str = "branch_id", subset_col: str = "subset") -> pd.DataFrame:
    return df[(df[branch_col] == PRIMARY_BRANCH) & (df[subset_col] == PRIMARY_SUBSET)].copy()


def build_statistical_summary(
    model_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
) -> dict[str, Any]:
    negative_models = model_summary.loc[model_summary["beta"] < 0, "model_key"].tolist()
    positive_models = model_summary.loc[model_summary["beta"] > 0, "model_key"].tolist()
    mc_95_models = mc_summary.loc[mc_summary["fraction_beta_lt_0"] >= 0.95, "model_key"].tolist()
    not_mc_95_models = [model for model in MODEL_ORDER if model not in mc_95_models]
    return {
        "n_models": len(model_summary),
        "n_beta_negative": len(negative_models),
        "n_beta_positive": len(positive_models),
        "n_mc_beta_negative_ge_95pct": len(mc_95_models),
        "n_mc_beta_negative_ge_99pct": int((mc_summary["fraction_beta_lt_0"] >= 0.99).sum()),
        "median_mc_frequency_beta_negative": float(mc_summary["fraction_beta_lt_0"].median()),
        "median_delta_chi2_positive_favors_reversed": float(model_summary["delta_chi2_adopted_minus_reversed"].median()),
        "median_improvement_over_null": float(model_summary["improvement_over_null"].median()),
        "models_with_mc_beta_negative_ge_95pct": ";".join(mc_95_models),
        "models_not_negative_ge_95pct": ";".join(not_mc_95_models),
    }


def validate_residual_tables(tables: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    definitions = tables["compatible_residual_definitions"].set_index("branch_id")
    coefficients = tables["compatible_residual_coefficients"].set_index("branch_id")
    shared = definitions.index.intersection(coefficients.index)
    if len(shared) != len(definitions):
        return False, "branch mismatch"
    max_diff = 0.0
    tables_match = True
    for branch in shared:
        definition_values = definitions.loc[branch, DEG2_ORDER].to_numpy(dtype=float)
        coefficient_values = coefficients.loc[branch, DEG2_ORDER].to_numpy(dtype=float)
        diff = np.abs(definition_values - coefficient_values)
        max_diff = max(max_diff, float(diff.max()))
        if not np.allclose(definition_values, coefficient_values, rtol=0.0, atol=1e-18):
            tables_match = False
    return tables_match, f"max_abs_diff={max_diff:.3e}"


def validate_registry_coefficient_order(tables: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    registry = tables["compatible_reference_state_registry"]
    orders = registry["coefficient_order"].dropna().astype(str).unique().tolist()
    parsed_orders = []
    for order in orders:
        try:
            parsed_orders.append(json.loads(order))
        except json.JSONDecodeError:
            return False, f"unparseable coefficient_order={order}"
    passed = all(order == DEG2_ORDER for order in parsed_orders)
    return passed, f"orders={parsed_orders}"


def validate_against_public_archives(
    config: PublicConfig,
    tables: dict[str, pd.DataFrame],
    cov: np.ndarray,
    model_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    sample_by_model: dict[str, np.ndarray],
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    model_archive = primary_archive_rows(tables["signed_beta_significance"]).set_index("model_key").loc[MODEL_ORDER]
    model_calc = model_summary.set_index("model_key").loc[MODEL_ORDER]
    beta_diffs = pair_max_abs_diffs(
        model_calc.reset_index(),
        model_archive.reset_index(),
        [
            ("beta", "beta"),
            ("beta_sigma", "beta_sigma"),
            ("beta_ci_low", "beta_ci_low"),
            ("beta_ci_high", "beta_ci_high"),
            ("beta_z_score", "beta_z_score"),
            ("p_beta_less_zero_analytic", "p_beta_less_zero_analytic"),
        ],
    )
    misfit_diffs = pair_max_abs_diffs(
        model_calc.reset_index(),
        model_archive.reset_index(),
        [
            ("weighted_misfit_chi2", "weighted_misfit_chi2"),
            ("weighted_null_chi2", "weighted_null_chi2"),
            ("improvement_over_null", "improvement_over_null"),
        ],
    )
    misfit_chi_pairs = [
        ("weighted_misfit_chi2", "weighted_misfit_chi2"),
        ("weighted_null_chi2", "weighted_null_chi2"),
    ]
    beta_passed = (
        beta_diffs["beta"] < 1e-18
        and beta_diffs["beta_sigma"] < 1e-24
        and beta_diffs["beta_ci_low"] < 1e-18
        and beta_diffs["beta_ci_high"] < 1e-18
        and beta_diffs["beta_z_score"] < 1e-6
        and beta_diffs["p_beta_less_zero_analytic"] < 1e-15
        and columns_allclose(model_calc.reset_index(), model_archive.reset_index(), misfit_chi_pairs)
        and misfit_diffs["improvement_over_null"] < 1e-15
    )
    rows.append(
        {
            "validation": "signed_weighted_least_squares_matches_public_archive",
            "passed": beta_passed,
            "value": (
                "beta={beta:.3e}; sigma={sigma:.3e}; ci_low={ci_low:.3e}; ci_high={ci_high:.3e}; "
                "z={z:.3e}; p={p:.3e}; chi2={chi2:.3e}"
            ).format(
                beta=beta_diffs["beta"],
                sigma=beta_diffs["beta_sigma"],
                ci_low=beta_diffs["beta_ci_low"],
                ci_high=beta_diffs["beta_ci_high"],
                z=beta_diffs["beta_z_score"],
                p=beta_diffs["p_beta_less_zero_analytic"],
                chi2=max(misfit_diffs["weighted_misfit_chi2"], misfit_diffs["weighted_null_chi2"]),
            ),
        }
    )
    sign_match = model_calc["preferred_sign"].tolist() == model_archive["gravity_supported_sign"].tolist()
    rows.append(
        {
            "validation": "signed_beta_decision_rule_matches_public_archive",
            "passed": sign_match,
            "value": ";".join(model_calc["preferred_sign"].tolist()),
        }
    )

    constrained_archive = primary_archive_rows(tables["constrained_sign_hypothesis_comparison"]).set_index("model_key").loc[MODEL_ORDER]
    constrained_beta_diffs = pair_max_abs_diffs(
        model_calc.reset_index(),
        constrained_archive.reset_index(),
        [
            ("beta", "unconstrained_signed_beta"),
            ("adopted_beta_nonnegative", "adopted_hypothesis_beta_constrained_nonnegative"),
            ("reversed_beta_nonnegative", "reversed_hypothesis_beta_constrained_nonnegative"),
        ],
    )
    constrained_chi_diffs = pair_max_abs_diffs(
        model_calc.reset_index(),
        constrained_archive.reset_index(),
        [
            ("adopted_sign_weighted_residual_chi2", "chi2_adopted_constrained"),
            ("reversed_sign_weighted_residual_chi2", "chi2_reversed_constrained"),
            ("delta_chi2_adopted_minus_reversed", "delta_chi2_adopted_minus_reversed_positive_favors_reversed"),
        ],
    )
    constrained_chi_pairs = [
        ("adopted_sign_weighted_residual_chi2", "chi2_adopted_constrained"),
        ("reversed_sign_weighted_residual_chi2", "chi2_reversed_constrained"),
        ("delta_chi2_adopted_minus_reversed", "delta_chi2_adopted_minus_reversed_positive_favors_reversed"),
    ]
    constrained_passed = (
        constrained_beta_diffs["beta"] < 1e-18
        and constrained_beta_diffs["adopted_beta_nonnegative"] < 1e-18
        and constrained_beta_diffs["reversed_beta_nonnegative"] < 1e-18
        and columns_allclose(model_calc.reset_index(), constrained_archive.reset_index(), constrained_chi_pairs)
    )
    rows.append(
        {
            "validation": "nonnegative_amplitude_constrained_comparison_matches_public_archive",
            "passed": constrained_passed,
            "value": (
                "beta={beta:.3e}; adopted_beta={adopted_beta:.3e}; reversed_beta={reversed_beta:.3e}; "
                "chi2={chi2:.3e}; delta={delta:.3e}"
            ).format(
                beta=constrained_beta_diffs["beta"],
                adopted_beta=constrained_beta_diffs["adopted_beta_nonnegative"],
                reversed_beta=constrained_beta_diffs["reversed_beta_nonnegative"],
                chi2=max(
                    constrained_chi_diffs["adopted_sign_weighted_residual_chi2"],
                    constrained_chi_diffs["reversed_sign_weighted_residual_chi2"],
                ),
                delta=constrained_chi_diffs["delta_chi2_adopted_minus_reversed"],
            ),
        }
    )

    delta_archive = primary_archive_rows(tables["delta_chi_square_by_model"]).set_index("model_key").loc[MODEL_ORDER]
    delta_diff = max_abs_diff(
        model_calc.reset_index(),
        delta_archive.reset_index(),
        [("delta_chi2_adopted_minus_reversed", "delta_chi2_positive_favors_reversed")],
    )
    delta_close = columns_allclose(
        model_calc.reset_index(),
        delta_archive.reset_index(),
        [("delta_chi2_adopted_minus_reversed", "delta_chi2_positive_favors_reversed")],
    )
    rows.append(
        {
            "validation": "delta_chi_square_matches_public_archive",
            "passed": delta_close,
            "value": f"max_abs_diff={delta_diff:.3e}",
        }
    )
    favored_match = model_calc["favored_constrained_sign"].tolist() == delta_archive["favored_constrained_sign"].tolist()
    rows.append(
        {
            "validation": "constrained_favored_sign_matches_public_archive",
            "passed": favored_match,
            "value": ";".join(model_calc["favored_constrained_sign"].tolist()),
        }
    )

    mc_archive = primary_archive_rows(tables["monte_carlo_sign_confidence"]).set_index("model_key").loc[MODEL_ORDER]
    mc_calc = mc_summary.set_index("model_key").loc[MODEL_ORDER]
    mc_calc_for_archive = mc_calc.copy()
    mc_calc_for_archive["mc_beta_p05"] = [np.quantile(sample_by_model[model], 0.05) for model in MODEL_ORDER]
    mc_calc_for_archive["mc_beta_median"] = [np.quantile(sample_by_model[model], 0.50) for model in MODEL_ORDER]
    mc_calc_for_archive["mc_beta_p95"] = [np.quantile(sample_by_model[model], 0.95) for model in MODEL_ORDER]
    mc_diff = max_abs_diff(
        mc_calc_for_archive.reset_index(),
        mc_archive.reset_index(),
        [
            ("n_realizations", "n_monte_carlo"),
            ("fraction_beta_lt_0", "mc_frequency_beta_negative"),
            ("fraction_beta_gt_0", "mc_frequency_beta_positive"),
            ("mc_beta_p05", "mc_beta_p05"),
            ("mc_beta_median", "mc_beta_median"),
            ("mc_beta_p95", "mc_beta_p95"),
        ],
    )
    rows.append(
        {
            "validation": "monte_carlo_summary_matches_public_archive",
            "passed": mc_diff < 1e-18,
            "value": f"max_abs_diff={mc_diff:.3e}",
        }
    )

    summary_archive = tables["statistical_robustness_summary"]
    summary_archive = summary_archive[
        (summary_archive["branch_id"] == config.primary_branch)
        & (summary_archive["subset"] == config.primary_subset)
    ]
    if len(summary_archive) != 1:
        rows.append(
            {
                "validation": "statistical_robustness_public_archive_row_found",
                "passed": False,
                "value": f"rows={len(summary_archive)}",
            }
        )
    else:
        archive_row = summary_archive.iloc[0]
        calc_row = build_statistical_summary(model_summary, mc_summary)
        exact_count_columns = [
            "n_models",
            "n_beta_negative",
            "n_beta_positive",
            "n_mc_beta_negative_ge_95pct",
            "n_mc_beta_negative_ge_99pct",
        ]
        non_chi_numeric_columns = [
            "median_mc_frequency_beta_negative",
            "median_improvement_over_null",
        ]
        count_match = all(int(calc_row[col]) == int(archive_row[col]) for col in exact_count_columns)
        non_chi_diff = max(abs(float(calc_row[col]) - float(archive_row[col])) for col in non_chi_numeric_columns)
        median_delta_diff = abs(
            float(calc_row["median_delta_chi2_positive_favors_reversed"])
            - float(archive_row["median_delta_chi2_positive_favors_reversed"])
        )
        median_delta_close = bool(
            np.allclose(
                [float(calc_row["median_delta_chi2_positive_favors_reversed"])],
                [float(archive_row["median_delta_chi2_positive_favors_reversed"])],
                rtol=1e-12,
                atol=4.0,
            )
        )
        summary_diff = max(non_chi_diff, median_delta_diff)
        list_match = (
            calc_row["models_with_mc_beta_negative_ge_95pct"]
            == str(archive_row["models_with_mc_beta_negative_ge_95pct"])
            and calc_row["models_not_negative_ge_95pct"]
            == ("" if pd.isna(archive_row["models_not_negative_ge_95pct"]) else str(archive_row["models_not_negative_ge_95pct"]))
        )
        rows.append(
            {
                "validation": "statistical_robustness_summary_matches_public_archive",
                "passed": count_match and non_chi_diff < 1e-6 and median_delta_close and list_match,
                "value": (
                    f"max_abs_diff={summary_diff:.3e}; median_delta_abs_diff={median_delta_diff:.3e}; "
                    f"counts_match={count_match}; model_lists_match={list_match}"
                ),
            }
        )

    residual_tables_passed, residual_value = validate_residual_tables(tables)
    registry_passed, registry_value = validate_registry_coefficient_order(tables)
    rows.extend(
        [
            {
                "validation": "all_public_inputs_present",
                "passed": all(path.exists() for path in PUBLIC_INPUTS.values()),
                "value": str(len(PUBLIC_INPUTS)),
            },
            {
                "validation": "coefficient_ordering_verified",
                "passed": DEG2_ORDER == ["C20", "C21", "S21", "C22", "S22"],
                "value": ",".join(DEG2_ORDER),
            },
            {
                "validation": "primary_branch_and_subset_verified",
                "passed": config.primary_branch == PRIMARY_BRANCH and config.primary_subset == PRIMARY_SUBSET,
                "value": f"{config.primary_branch};{config.primary_subset}",
            },
            {
                "validation": "random_seed_and_monte_carlo_count_verified",
                "passed": config.random_seed == 42 and config.monte_carlo_count == 10_000,
                "value": f"seed={config.random_seed};n={config.monte_carlo_count}",
            },
            {
                "validation": "covariance_dimensions_verified",
                "passed": cov.shape == (5, 5) and np.allclose(cov, np.diag(np.diag(cov))),
                "value": "5x5 diagonal covariance",
            },
            {
                "validation": "residual_definition_and_coefficient_tables_match",
                "passed": residual_tables_passed,
                "value": residual_value,
            },
            {
                "validation": "reference_state_registry_coefficient_order_verified",
                "passed": registry_passed,
                "value": registry_value,
            },
            {
                "validation": "all_eight_models_present",
                "passed": model_summary["model_key"].tolist() == MODEL_ORDER,
                "value": ";".join(model_summary["model_key"].tolist()),
            },
            {
                "validation": "seven_velocity_models_and_one_native_density_model",
                "passed": (
                    int((model_summary["input_category"] == "velocity-based").sum()) == 7
                    and int((model_summary["input_category"] == "native density-related").sum()) == 1
                    and str(model_calc.loc["CSEM2_Globe_RHO", "input_category"]) == "native density-related"
                ),
                "value": ";".join(f"{m}:{MODEL_METADATA[m]['input_category']}" for m in MODEL_ORDER),
            },
            {
                "validation": "deterministic_result_is_7_of_8_reversed",
                "passed": int((model_summary["preferred_sign"] == "reversed").sum()) == 7,
                "value": str(int((model_summary["preferred_sign"] == "reversed").sum())),
            },
            {
                "validation": "monte_carlo_result_has_7_of_8_negative_ge_95pct",
                "passed": int((mc_summary["fraction_beta_lt_0"] >= 0.95).sum()) == 7,
                "value": str(int((mc_summary["fraction_beta_lt_0"] >= 0.95).sum())),
            },
            {
                "validation": "target_proximity_not_used_for_sign",
                "passed": bool(not model_summary["target_proximity_used_for_sign"].any()),
                "value": "target proximity columns are not read by the sign fit",
            },
            {
                "validation": "positive_normalization_invariance_check_passed",
                "passed": bool(
                    (
                        sensitivity.loc[
                            sensitivity["sensitivity_id"] == "positive_normalization_invariance",
                            "status",
                        ]
                        == "passed"
                    ).all()
                ),
                "value": "scale factors 0.1 and 10.0",
            },
            {
                "validation": "global_sign_reversal_identity_check_passed",
                "passed": bool(
                    (
                        sensitivity.loc[
                            sensitivity["sensitivity_id"] == "global_sign_reversal_identity",
                            "status",
                        ]
                        == "passed"
                    ).all()
                ),
                "value": "g to -g flips beta sign",
            },
            {
                "validation": "leave_one_coefficient_out_sensitivity_written",
                "passed": int((sensitivity["sensitivity_id"] == "leave_one_coefficient_out").sum()) == len(DEG2_ORDER),
                "value": str(int((sensitivity["sensitivity_id"] == "leave_one_coefficient_out").sum())),
            },
        ]
    )

    role_checks = [
        role_swap_check(model_prediction_vector(tables["mantle_model_degree2_predictions"], model, config.primary_subset))
        for model in MODEL_ORDER
    ]
    max_role_angle = max(
        max(
            float(row["min_to_reversed_max_angle_deg"]),
            float(row["mid_to_reversed_mid_angle_deg"]),
            float(row["max_to_reversed_min_angle_deg"]),
        )
        for row in role_checks
    )
    rows.extend(
        [
            {
                "validation": "sign_reversal_preserves_tensor_eigenvectors",
                "passed": max_role_angle < 1e-5,
                "value": f"{max_role_angle:.3e} deg",
            },
            {
                "validation": "sign_reversal_swaps_min_max_tensor_roles",
                "passed": all(bool(row["min_max_roles_swapped"]) for row in role_checks),
                "value": "min<->max; middle unchanged",
            },
        ]
    )
    return pd.DataFrame(rows)


def write_manifest(
    config: PublicConfig,
    validation: pd.DataFrame,
    model_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
) -> None:
    manifest = {
        "run_status": "completed" if validation["passed"].astype(bool).all() else "failed_validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "coefficient_order": DEG2_ORDER,
        "primary_branch": config.primary_branch,
        "primary_subset": config.primary_subset,
        "random_seed": config.random_seed,
        "monte_carlo_count": config.monte_carlo_count,
        "model_order": MODEL_ORDER,
        "public_inputs": [path.relative_to(REPO_ROOT).as_posix() for path in PUBLIC_INPUTS.values()],
        "outputs": OUTPUT_FILES,
        "deterministic_reversed_count": int((model_summary["preferred_sign"] == "reversed").sum()),
        "mc_negative_ge_95pct_count": int((mc_summary["fraction_beta_lt_0"] >= 0.95).sum()),
        "validation_passed": bool(validation["passed"].astype(bool).all()),
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_public_reproduction(config: PublicConfig | None = None) -> dict[str, Any]:
    config = config or PublicConfig()
    ensure_output_dir(config)
    tables = load_public_tables()
    cov, sigma = load_covariance(tables["degree2_gravity_product_summary"])
    weight = np.linalg.pinv(cov)
    model_summary = deterministic_summary(config, tables, weight)
    branch_samples = draw_branch_samples(config, tables, cov)
    mc_summary, sample_by_model = monte_carlo_summary(config, tables, weight, branch_samples)
    sensitivity = sensitivity_summary(config, tables, cov, weight)
    validation = validate_against_public_archives(
        config,
        tables,
        cov,
        model_summary,
        mc_summary,
        sample_by_model,
        sensitivity,
    )
    input_hashes = write_public_input_hashes(config)
    write_csv(model_summary, config.output_dir / "sign_test_model_summary.csv")
    write_csv(mc_summary, config.output_dir / "sign_test_monte_carlo_summary.csv")
    write_csv(sensitivity, config.output_dir / "sign_test_sensitivity_summary.csv")
    write_csv(validation, config.output_dir / "sign_test_validation.csv")
    write_manifest(config, validation, model_summary, mc_summary)

    unexpected_outputs = sorted(
        path.name
        for path in config.output_dir.iterdir()
        if path.is_file() and path.name not in OUTPUT_FILES
    )
    if unexpected_outputs:
        raise RuntimeError(
            "Unexpected file(s) already present in public output directory: " + ", ".join(unexpected_outputs)
        )
    failed = validation.loc[~validation["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError("Public reproduction validation failed; see sign_test_validation.csv.")
    return {
        "config": config,
        "tables": tables,
        "covariance": cov,
        "sigma": sigma,
        "model_summary": model_summary,
        "monte_carlo_summary": mc_summary,
        "sensitivity_summary": sensitivity,
        "validation": validation,
        "public_input_hashes": input_hashes,
    }


def main() -> None:
    results = run_public_reproduction()
    output_dir = results["config"].output_dir
    validation = results["validation"]
    print(f"Wrote public reproduction outputs to {output_dir.relative_to(REPO_ROOT)}")
    print(f"Validation checks passed: {int(validation['passed'].sum())}/{len(validation)}")


if __name__ == "__main__":
    main()

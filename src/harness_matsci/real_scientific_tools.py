from __future__ import annotations

import hashlib
import math
import platform
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


PAIRWISE_PATTERN = re.compile(
    r"Candidate A: Composition: (?P<a_formula>.*?); crystal system: (?P<a_crystal>.*?); "
    r"space group: (?P<a_spg>\d+).*?Candidate B: Composition: (?P<b_formula>.*?); "
    r"crystal system: (?P<b_crystal>.*?); space group: (?P<b_spg>\d+)",
    re.DOTALL,
)
UNIQUE_PATTERN = re.compile(
    r"Composition: (?P<formula>.*?); crystal system: (?P<crystal>.*?); space group: (?P<spg>\d+)",
    re.DOTALL,
)
SMILES_PATTERN = re.compile(r"target\s+(?P<target>C\d+):\s*(?P<smiles>.+)$")
CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


@dataclass(frozen=True)
class RealToolObservation:
    benchmark: str
    tool_name: str
    outputs: dict[str, float]
    provenance: dict[str, Any]
    duration_seconds: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class RealScientificToolSuite:
    """Local MatBot tool suite backed by executed chemistry/materials libraries.

    The material-property model is fit only from caller-supplied training records.
    Candidate outcomes are never read by :meth:`observe`.
    """

    def __init__(self, *, random_state: int = 1729, n_estimators: int = 192) -> None:
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self._regressor = None
        self._scaler = None
        self._neighbors = None
        self._novelty_scale = 1.0
        self._training_material_count = 0
        self._drd2_regressor = None
        self._drd2_residual_scale = 1.0
        self._training_molecule_count = 0
        self._drd2_calibration_count = 0

    @property
    def fitted(self) -> bool:
        return self._regressor is not None

    def fit(self, raw_training_rows: Iterable[dict[str, Any]]) -> "RealScientificToolSuite":
        import numpy as np
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        rows = list(raw_training_rows)
        materials: dict[str, tuple[list[float], float]] = {}
        for row in rows:
            for material_id, features, target in _training_materials(row):
                materials.setdefault(material_id, (features, target))
        if len(materials) < 100:
            raise ValueError("real material tool requires at least 100 unique training materials")
        matrix = np.asarray([item[0] for item in materials.values()], dtype=float)
        targets = np.asarray([item[1] for item in materials.values()], dtype=float)
        self._regressor = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(matrix, targets)
        self._scaler = StandardScaler().fit(matrix)
        scaled = self._scaler.transform(matrix)
        self._neighbors = NearestNeighbors(n_neighbors=2).fit(scaled)
        neighbor_distances = self._neighbors.kneighbors(scaled, return_distance=True)[0][:, 1]
        self._novelty_scale = float(max(1e-6, np.quantile(neighbor_distances, 0.9)))
        self._training_material_count = len(materials)
        self._fit_drd2_surrogate(rows)
        return self

    def _fit_drd2_surrogate(self, rows: Sequence[dict[str, Any]]) -> None:
        import numpy as np
        from sklearn.ensemble import ExtraTreesRegressor

        molecules: dict[str, tuple[list[float], float]] = {}
        for row in rows:
            training_example = _training_drd2_example(row)
            if training_example is not None:
                molecule_id, features, target = training_example
                molecules.setdefault(molecule_id, (features, target))
        if len(molecules) < 100:
            return
        ordered = sorted(
            molecules.items(),
            key=lambda item: hashlib.sha256(
                f"{self.random_state}|drd2|{item[0]}".encode()
            ).hexdigest(),
        )
        calibration_count = max(20, int(math.ceil(0.20 * len(ordered))))
        fit_rows = ordered[:-calibration_count]
        calibration_rows = ordered[-calibration_count:]
        fit_x = np.asarray([item[1][0] for item in fit_rows], dtype=float)
        fit_y = np.asarray([item[1][1] for item in fit_rows], dtype=float)
        self._drd2_regressor = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=self.random_state + 17,
            n_jobs=-1,
        ).fit(fit_x, fit_y)
        calibration_x = np.asarray(
            [item[1][0] for item in calibration_rows], dtype=float
        )
        calibration_y = np.asarray(
            [item[1][1] for item in calibration_rows], dtype=float
        )
        residuals = self._drd2_regressor.predict(calibration_x) - calibration_y
        self._drd2_residual_scale = float(
            max(1e-6, math.sqrt(float(np.mean(residuals**2))))
        )
        self._training_molecule_count = len(fit_rows)
        self._drd2_calibration_count = len(calibration_rows)

    def observe(self, row: dict[str, Any]) -> RealToolObservation:
        benchmark = str(row["benchmark"])
        started = time.perf_counter()
        if benchmark == "matbench_pairwise":
            outputs, tool_name = self._observe_pairwise(row)
        elif benchmark == "discover_unique":
            outputs, tool_name = self._observe_unique(row)
        elif benchmark == "extreme_properties":
            outputs, tool_name = self._observe_extreme(row)
        else:
            raise ValueError(f"unsupported scientific tool benchmark {benchmark}")
        duration = time.perf_counter() - started
        return RealToolObservation(
            benchmark=benchmark,
            tool_name=tool_name,
            outputs={key: float(value) for key, value in outputs.items()},
            provenance={
                "executed": True,
                "hidden_outcome_read": False,
                "runtime": "local_matbot_tool_runtime",
                "python": platform.python_version(),
                "implementation_digest": _implementation_digest(),
                "training_material_count": self._training_material_count,
                "training_molecule_count": self._training_molecule_count,
                "drd2_calibration_count": self._drd2_calibration_count,
                "drd2_calibration_rmse": self._drd2_residual_scale,
                "historical_training_labels_only": True,
                **_library_versions(),
            },
            duration_seconds=float(duration),
        )

    def _observe_pairwise(self, row: dict[str, Any]) -> tuple[dict[str, float], str]:
        if not self.fitted:
            raise ValueError("material tool must be fit before pairwise observation")
        match = PAIRWISE_PATTERN.search(str(row["visible_context"]))
        if match is None:
            raise ValueError("could not parse pairwise visible context")
        left_mean, left_std = self._predict_material(
            match.group("a_formula"), match.group("a_crystal"), int(match.group("a_spg"))
        )
        right_mean, right_std = self._predict_material(
            match.group("b_formula"), match.group("b_crystal"), int(match.group("b_spg"))
        )
        chosen = _chosen_side(str(row["candidate_action"]))
        chosen_mean, other_mean = (left_mean, right_mean) if chosen == "A" else (right_mean, left_mean)
        chosen_std, other_std = (left_std, right_std) if chosen == "A" else (right_std, left_std)
        margin = chosen_mean - other_mean
        margin_std = math.sqrt(chosen_std**2 + other_std**2)
        return {
            "chosen_property_mean": chosen_mean,
            "alternative_property_mean": other_mean,
            "predicted_margin": margin,
            "predictive_standard_deviation": margin_std,
            "positive_probability": _normal_cdf(margin / max(1e-6, margin_std)),
            "estimated_utility": _clip01(0.5 + margin),
        }, "pymatgen_extratrees_elasticity"

    def _observe_unique(self, row: dict[str, Any]) -> tuple[dict[str, float], str]:
        if not self.fitted:
            raise ValueError("material tool must be fit before uniqueness observation")
        match = UNIQUE_PATTERN.search(str(row["visible_context"]))
        if match is None:
            raise ValueError("could not parse unique-material visible context")
        features = material_feature_vector(
            match.group("formula"), match.group("crystal"), int(match.group("spg"))
        )
        performance, performance_std = self._predict_vector(features)
        scaled = self._scaler.transform([features])
        nearest = float(self._neighbors.kneighbors(scaled, n_neighbors=1)[0][0, 0])
        novelty = _clip01(nearest / self._novelty_scale)
        discovery = math.sqrt(max(0.0, performance) * max(0.0, novelty))
        uncertainty = min(1.0, performance_std + 0.25 / (1.0 + nearest))
        return {
            "predicted_performance": performance,
            "performance_standard_deviation": performance_std,
            "nearest_training_distance": nearest,
            "composition_structure_novelty": novelty,
            "estimated_discovery_utility": discovery,
            "predictive_standard_deviation": uncertainty,
            "positive_probability": _clip01(discovery * (1.0 - 0.5 * uncertainty)),
            "estimated_utility": discovery,
        }, "pymatgen_extratrees_novelty"

    def _observe_extreme(self, row: dict[str, Any]) -> tuple[dict[str, float], str]:
        raw = row.get("raw_record")
        if not isinstance(raw, dict):
            raise ValueError("extreme-property observation requires raw_record source metadata")
        match = SMILES_PATTERN.search(str(raw["candidate_action"]))
        if match is None:
            raise ValueError("could not parse candidate SMILES")
        candidate_smiles = raw.get("tool_outputs", {}).get("generated", {}).get("SMILES")
        if not candidate_smiles:
            candidate_smiles = match.group("smiles")
        target = raw.get("tool_outputs", {}).get("target", {})
        target_smiles = target.get("smiles")
        bounds = raw.get("source", {}).get("target_bounds", {})
        if not target_smiles or not bounds:
            raise ValueError("published target SMILES and bounds are required")
        try:
            candidate_values = rdkit_descriptors(str(candidate_smiles))
        except ValueError:
            return {
                "parse_success": 0.0,
                "estimated_utility": 0.0,
                "positive_probability": 0.0,
                "predictive_standard_deviation": 1.0,
                "drd2_not_observed": 1.0,
            }, "rdkit_published_target_descriptor_check"
        descriptor_keys = ("MW", "logP", "TPSA", "QED", "HBA", "HBD")
        target_values = {key: float(target[key]) for key in descriptor_keys}
        ratios = {
            key: abs(candidate_values[key] - target_values[key]) / max(1e-9, float(bounds[key]))
            for key in descriptor_keys
        }
        hit_count = sum(value <= 1.0 for value in ratios.values())
        hit_fraction = hit_count / len(ratios)
        clipped_error = sum(min(2.0, value) for value in ratios.values()) / len(ratios)
        drd2_mean, drd2_std = self._predict_drd2(str(candidate_smiles))
        drd2_target = float(target["DRD2"])
        drd2_bound = float(bounds["DRD2"])
        lower = drd2_target - drd2_bound
        upper = drd2_target + drd2_bound
        drd2_hit_probability = _normal_interval_probability(
            lower,
            upper,
            mean=drd2_mean,
            standard_deviation=drd2_std,
        )
        drd2_error_ratio = abs(drd2_mean - drd2_target) / max(1e-9, drd2_bound)
        expected_hit_fraction = (hit_count + drd2_hit_probability) / 7.0
        expected_clipped_error = (
            sum(min(2.0, value) for value in ratios.values())
            + min(2.0, drd2_error_ratio)
        ) / 7.0
        estimated_utility = _clip01(
            expected_hit_fraction * math.exp(-0.35 * expected_clipped_error)
        )
        boundary_distance = min(abs(value - 1.0) for value in ratios.values())
        descriptor_uncertainty = _clip01(math.exp(-2.0 * boundary_distance))
        normalized_drd2_uncertainty = _clip01(drd2_std / max(1e-9, drd2_bound))
        uncertainty = _clip01(
            math.sqrt(
                0.5 * descriptor_uncertainty**2
                + 0.5 * normalized_drd2_uncertainty**2
            )
        )
        all_seven_hit_probability = (
            drd2_hit_probability if hit_count == len(descriptor_keys) else 0.0
        )
        outputs = {
            "parse_success": 1.0,
            "candidate_artifact_retrieved": float(candidate_smiles != match.group("smiles")),
            "six_descriptor_hit_fraction": hit_fraction,
            "mean_clipped_error_ratio": clipped_error,
            "predicted_drd2": drd2_mean,
            "drd2_predictive_standard_deviation": drd2_std,
            "drd2_error_ratio": drd2_error_ratio,
            "drd2_hit_probability": drd2_hit_probability,
            "expected_seven_descriptor_hit_fraction": expected_hit_fraction,
            "all_seven_hit_probability": all_seven_hit_probability,
            "estimated_utility": estimated_utility,
            "positive_probability": all_seven_hit_probability,
            "predictive_standard_deviation": uncertainty,
            "drd2_not_observed": 1.0,
            "drd2_train_only_surrogate": 1.0,
        }
        for key in descriptor_keys:
            outputs[f"candidate_{key}"] = candidate_values[key]
            outputs[f"target_{key}"] = target_values[key]
            outputs[f"error_ratio_{key}"] = ratios[key]
        return outputs, "rdkit_train_only_drd2_surrogate_check"

    def _predict_drd2(self, smiles: str) -> tuple[float, float]:
        import numpy as np

        if self._drd2_regressor is None:
            raise ValueError("DRD2 surrogate requires historical molecular training records")
        features = molecular_feature_vector(smiles)
        tree_predictions = np.asarray(
            [tree.predict([features])[0] for tree in self._drd2_regressor.estimators_],
            dtype=float,
        )
        mean = _clip01(float(tree_predictions.mean()))
        standard_deviation = math.sqrt(
            float(tree_predictions.std(ddof=1)) ** 2 + self._drd2_residual_scale**2
        )
        return mean, standard_deviation

    def _predict_material(self, formula: str, crystal: str, spg: int) -> tuple[float, float]:
        return self._predict_vector(material_feature_vector(formula, crystal, spg))

    def _predict_vector(self, features: Sequence[float]) -> tuple[float, float]:
        import numpy as np

        tree_predictions = np.asarray(
            [tree.predict([features])[0] for tree in self._regressor.estimators_], dtype=float
        )
        return _clip01(float(tree_predictions.mean())), float(tree_predictions.std(ddof=1))


def material_feature_vector(formula: str, crystal: str, spg: int) -> list[float]:
    from pymatgen.core import Composition

    composition = Composition(formula)
    fractions = [(element, float(amount) / composition.num_atoms) for element, amount in composition.items()]
    properties = {
        "z": [float(element.Z) for element, _ in fractions],
        "mass": [float(element.atomic_mass) for element, _ in fractions],
        "x": [float(element.X or 0.0) for element, _ in fractions],
        "row": [float(element.row or 0.0) for element, _ in fractions],
        "group": [float(element.group or 0.0) for element, _ in fractions],
    }
    weights = [fraction for _, fraction in fractions]
    output = [len(fractions) / 10.0, min(1.0, float(composition.num_atoms) / 100.0)]
    entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
    output.append(entropy / math.log(20.0))
    scales = {"z": 100.0, "mass": 250.0, "x": 4.0, "row": 7.0, "group": 18.0}
    for key in ("z", "mass", "x", "row", "group"):
        mean = sum(weight * value for weight, value in zip(weights, properties[key]))
        variance = sum(weight * (value - mean) ** 2 for weight, value in zip(weights, properties[key]))
        output.extend([mean / scales[key], math.sqrt(variance) / scales[key]])
    output.append(float(spg) / 230.0)
    crystal_key = str(crystal).strip().lower()
    output.extend(float(crystal_key == item) for item in CRYSTAL_SYSTEMS)
    return output


def rdkit_descriptors(smiles: str) -> dict[str, float]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.error")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse candidate SMILES")
    properties = QED.properties(molecule)
    return {
        "MW": float(properties.MW),
        "logP": float(properties.ALOGP),
        "TPSA": float(properties.PSA),
        "QED": float(QED.qed(molecule)),
        "HBA": float(properties.HBA),
        "HBD": float(properties.HBD),
    }


def molecular_feature_vector(smiles: str) -> list[float]:
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.error")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse candidate SMILES")
    fingerprint = AllChem.GetMorganGenerator(
        radius=2,
        fpSize=1024,
    ).GetFingerprintAsNumPy(molecule).astype(np.float32)
    descriptors = rdkit_descriptors(smiles)
    scaled_descriptors = np.asarray(
        [
            math.log1p(descriptors["MW"]) / 8.0,
            math.tanh(descriptors["logP"] / 10.0),
            descriptors["TPSA"] / 400.0,
            descriptors["QED"],
            descriptors["HBA"] / 30.0,
            descriptors["HBD"] / 20.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([fingerprint, scaled_descriptors]).astype(float).tolist()


def _training_materials(row: dict[str, Any]):
    benchmark = row.get("benchmark")
    raw = row.get("raw_record", row)
    if benchmark == "matbench_pairwise":
        hidden = raw.get("tool_outputs", {}).get("hidden_evaluation", {})
        for key in ("candidate_a", "candidate_b"):
            item = hidden.get(key)
            if item:
                yield (
                    str(item["mbid"]),
                    material_feature_vector(item["composition"], item["crys_sys"], int(item["spg_num"])),
                    float(item["normalized_k_vrh"]),
                )
    elif benchmark == "discover_unique":
        item = raw.get("tool_outputs", {}).get("matbench", {})
        scores = raw.get("tool_outputs", {}).get("scores", {})
        if item and "performance_score" in scores:
            yield (
                str(item["mbid"]),
                material_feature_vector(item["composition"], item["crys_sys"], int(item["spg_num"])),
                float(scores["performance_score"]),
            )


def _training_drd2_example(
    row: dict[str, Any],
) -> tuple[str, list[float], float] | None:
    if row.get("benchmark") != "extreme_properties":
        return None
    raw = row.get("raw_record", row)
    generated = raw.get("tool_outputs", {}).get("generated", {})
    smiles = generated.get("SMILES")
    drd2 = generated.get("DRD2")
    if not smiles or drd2 is None:
        return None
    try:
        features = molecular_feature_vector(str(smiles))
    except ValueError:
        return None
    molecule_id = hashlib.sha256(str(smiles).encode()).hexdigest()
    return molecule_id, features, float(drd2)


def _chosen_side(action: str) -> str:
    match = re.search(r"candidate\s+([AB])", action, re.IGNORECASE)
    if match is None:
        raise ValueError("could not parse chosen candidate side")
    return match.group(1).upper()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_interval_probability(
    lower: float,
    upper: float,
    *,
    mean: float,
    standard_deviation: float,
) -> float:
    scale = max(1e-9, float(standard_deviation))
    return _clip01(
        _normal_cdf((upper - mean) / scale)
        - _normal_cdf((lower - mean) / scale)
    )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _library_versions() -> dict[str, str]:
    import rdkit
    import sklearn

    try:
        import pymatgen

        pymatgen_version = getattr(pymatgen, "__version__", "namespace-package")
    except Exception:
        pymatgen_version = "unavailable"
    return {
        "rdkit_version": str(getattr(rdkit, "__version__", "unknown")),
        "scikit_learn_version": str(getattr(sklearn, "__version__", "unknown")),
        "pymatgen_version": str(pymatgen_version),
    }


def _implementation_digest() -> str:
    return hashlib.sha256(b"real-scientific-tools-v2-drd2-surrogate").hexdigest()

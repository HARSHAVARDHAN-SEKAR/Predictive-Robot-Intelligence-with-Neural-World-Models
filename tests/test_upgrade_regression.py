"""Regression test for Stage 3 stress-result compatibility."""

import json
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_upgrade as upgrade


def test_stage3():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = root / "assets"
        results = root / "results"
        assets.mkdir()
        results.mkdir()

        state_path = results / "_upgrade_state.pkl"
        benchmark_path = results / "benchmark_results.json"

        benchmark_path.write_text(
            json.dumps({"existing_test": {"passed": True}})
        )

        state = {
            "corr": 0.49,
            "dis": np.array([0.1, 0.2, 0.3]),
            "err": np.array([0.2, 0.3, 0.5]),
            "fan": (
                np.zeros((7, 2, 2)),
                np.zeros((4, 2, 2)),
                np.zeros((5, 4, 2, 2)),
            ),
            "stress": {
                "Neural-MPC": {
                    "success_rate": 0.68,
                    "collision_rate": 0.24,
                    "energy": 17.20,
                },
                "Risk-MPC": {
                    "success_rate": 0.76,
                    "collision_rate": 0.12,
                    "energy": 18.84,
                },
            },
        }

        with state_path.open("wb") as f:
            pickle.dump(state, f)

        with (
            patch.object(upgrade, "ASSETS", assets),
            patch.object(upgrade, "RESULTS", results),
            patch.object(upgrade, "STATE", state_path),
        ):
            upgrade.stage3()

        data = json.loads(benchmark_path.read_text())
        rows = data["test5_risk_aware_stress"]

        assert data["existing_test"]["passed"]
        assert len(rows) == 2
        assert rows["Ensemble mean\n(5 futures)"]["success_rate"] == 0.68
        assert rows["Ensemble CVaR\n(risk-aware)"]["collision_rate"] == 0.12
        assert (assets / "uncertainty_ensemble.png").is_file()
        assert (assets / "risk_aware_stress.png").is_file()
        assert not state_path.exists()

    print("UPGRADE REGRESSION TEST PASSED")


if __name__ == "__main__":
    test_stage3()

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from microbench.config import load_yaml
from microbench.scenarios import materialize_official_suite
from microbench.scenarios.validation import (
    validate_scenario_config,
    validate_scenario_file,
    validate_suite_manifest_file,
)


class TestScenarioValidation(unittest.TestCase):
    def test_all_builtin_scenarios_validate(self):
        paths = sorted(Path("config/scenarios").glob("*.yaml"))
        self.assertGreater(len(paths), 0)
        reports = [validate_scenario_file(path) for path in paths]
        errors = {r.path: r.errors for r in reports if not r.ok}
        self.assertEqual(errors, {})

    def test_generated_suite_manifest_validates_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            generated = materialize_official_suite("official_alpha", out, overwrite=True)
            manifest_path = generated["manifest_path"]
            manifest = load_yaml(manifest_path)

            self.assertTrue(all(not Path(s["path"]).is_absolute() for s in manifest["scenarios"]))
            report = validate_suite_manifest_file(manifest_path)

        self.assertTrue(report.ok, report.errors)

    def test_validation_rejects_planar_world_marked_3d(self):
        cfg = {
            "scenario": {"name": "bad_3d", "duration_s": 1.0},
            "benchmark": {"dimension": "3d"},
            "world": {
                "planar": True,
                "bounds": {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0, "zmin": -1.0, "zmax": 1.0},
            },
            "agent_params": {"radius_m": 0.2, "v_max_mps": 1.0, "a_max_mps2": 1.0, "goal_tolerance_m": 0.1},
            "goals": {"min_goal_distance_m": 1.0},
            "spawn": {
                "type": "rect_to_rect",
                "start_region": {"center": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
                "goal_region": {"center": [1.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
            },
        }
        report = validate_scenario_config(cfg)

        self.assertFalse(report.ok)
        self.assertTrue(any("world.planar" in err for err in report.errors))

    def test_validation_rejects_3d_spawn_outside_bounds(self):
        cfg = {
            "scenario": {"name": "bad_bounds", "duration_s": 1.0},
            "benchmark": {"dimension": "3d"},
            "world": {
                "planar": False,
                "bounds": {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0},
            },
            "agent_params": {"radius_m": 0.2, "v_max_mps": 1.0, "a_max_mps2": 1.0, "goal_tolerance_m": 0.1},
            "goals": {"min_goal_distance_m": 1.0},
            "spawn": {
                "type": "sphere_swap",
                "center": [0.0, 0.0, 0.0],
                "radius_m": 10.0,
                "jitter_m": 0.0,
                "vertical_scale": 1.0,
                "min_abs_y_component": 0.1,
            },
        }
        report = validate_scenario_config(cfg)

        self.assertFalse(report.ok)
        self.assertTrue(any("spawn" in err and "world bounds" in err for err in report.errors))

    def test_validate_scenarios_cli_checks_builtins_and_generated_suite(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "microbench.cli",
                "validate-scenarios",
                "--all-builtins",
                "--generated-suite",
                "official_3d_stress",
                "--quiet",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("validation: PASS", proc.stdout)
        self.assertIn("scenarios=19", proc.stdout)
        self.assertIn("suite_manifests=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()

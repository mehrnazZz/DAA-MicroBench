from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from microbench.config import load_yaml
from microbench.core import EpisodeEngine
from microbench.scenarios import generate_spawns_goals, materialize_official_suite
from microbench.scenarios.families import SCENARIO_FAMILIES, suite_registry_dicts


class TestScenarioFamilies(unittest.TestCase):
    def test_sphere_swap_generates_true_3d_opposed_goals(self):
        cfg = {
            "goals": {"min_goal_distance_m": 40.0},
            "spawn": {
                "type": "sphere_swap",
                "center": [0.0, 0.0, 0.0],
                "radius_m": 30.0,
                "jitter_m": 0.0,
                "vertical_scale": 1.4,
                "min_abs_y_component": 0.15,
            },
        }
        spawns, goals = generate_spawns_goals(cfg, n_agents=32, rng=np.random.default_rng(7))

        self.assertGreater(float(np.ptp(spawns[:, 1])), 12.0)
        self.assertGreater(float(np.ptp(goals[:, 1])), 12.0)
        dots = np.sum(spawns * goals, axis=1)
        self.assertLess(float(np.mean(dots)), -500.0)

    def test_official_alpha_materializes_2d_and_3d_scenarios(self):
        with tempfile.TemporaryDirectory() as td:
            generated = materialize_official_suite("official_alpha", Path(td), overwrite=True)
            manifest = generated["manifest"]

            self.assertTrue(generated["manifest_path"].exists())
            self.assertEqual(len(generated["scenario_paths"]), 8)
            self.assertEqual({s["dimension"] for s in manifest["scenarios"]}, {"2d", "3d"})

            for scenario in manifest["scenarios"]:
                cfg = load_yaml(Path(td) / scenario["path"])
                self.assertEqual(cfg["scenario"]["name"], scenario["id"])
                self.assertEqual(cfg["benchmark"]["dimension"], scenario["dimension"])
                if scenario["dimension"] == "3d":
                    self.assertFalse(bool(cfg["world"]["planar"]))
                    bounds = cfg["world"].get("bounds", {})
                    self.assertGreater(float(bounds["ymax"]) - float(bounds["ymin"]), 0.0)

    def test_generated_3d_suite_loads_and_steps_in_volume(self):
        with tempfile.TemporaryDirectory() as td:
            generated = materialize_official_suite("official_3d_stress", Path(td), overwrite=True)
            scenario_path = generated["scenario_paths"][0]
            engine = EpisodeEngine(
                scenario_path=str(scenario_path),
                method="baseline_goal",
                n_agents=6,
                seed=0,
                comm_profile="ideal_50hz",
            )
            step = engine.step()
            engine.close()

        self.assertIsNotNone(step)
        self.assertFalse(engine.planar)
        self.assertGreater(float(np.ptp(engine.spawns[:, 1])), 4.0)

    def test_official_3d_families_are_not_planar_aliases(self):
        for family in SCENARIO_FAMILIES.values():
            if family.dimension != "3d":
                continue
            cfg = family.config
            self.assertFalse(bool(cfg["world"]["planar"]), family.scenario_id)
            spawn_type = cfg["spawn"]["type"]
            bounds = cfg["world"].get("bounds", {})
            y_span = float(bounds.get("ymax", 0.0)) - float(bounds.get("ymin", 0.0))
            self.assertGreater(y_span, 0.0, family.scenario_id)
            self.assertIn(spawn_type, {"sphere_swap", "four_way", "rect_to_rect"}, family.scenario_id)

    def test_new_agentic_families_include_noncooperative_and_priority_metadata(self):
        intruder = SCENARIO_FAMILIES["noncooperative_intruder_3d_hard"].config
        intruder_agent = intruder["agents"]["by_id"][0]
        self.assertEqual(intruder_agent["role"], "noncooperative_intruder")
        self.assertTrue(intruder_agent["failure_modes"]["noncooperative"])
        self.assertEqual(intruder["perception"]["mode"], "sensor")

        priority = SCENARIO_FAMILIES["heterogeneous_priority_crossing_3d_medium"].config
        self.assertGreater(priority["agents"]["by_id"][0]["priority"], priority["agents"]["by_id"][2]["priority"])
        self.assertTrue(priority["intent"]["enabled"])

    def test_suite_registry_lists_generated_and_handwritten_statuses(self):
        registry = {entry["suite"]: entry for entry in suite_registry_dicts()}

        self.assertEqual(registry["official_3d_stress"]["status"], "pre_v1_official")
        self.assertEqual(registry["official_3d_stress"]["source"], "generated")
        self.assertIn("merge_3d_hard", registry["official_3d_stress"]["scenarios"])
        self.assertIn("noncooperative_intruder_3d_hard", registry["official_3d_stress"]["scenarios"])
        self.assertEqual(registry["three_d"]["status"], "development")
        self.assertEqual(registry["primary"]["status"], "legacy_official")

    def test_materialize_suite_cli_prints_plan_without_running(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "suite"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "microbench.cli",
                    "materialize-suite",
                    "--suite",
                    "official_3d_stress",
                    "--out-dir",
                    str(out_dir),
                    "--print-plan",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("official_3d_stress", proc.stdout)
            self.assertTrue((out_dir / "suite_manifest.yaml").exists())
            self.assertEqual(len(list(out_dir.glob("*.yaml"))), 8)

    def test_list_suites_cli_reports_registry(self):
        proc = subprocess.run(
            [sys.executable, "-m", "microbench.cli", "list-suites"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("official_agentic_stress,pre_v1_official,generated,3d", proc.stdout)
        self.assertIn("three_d,development,hand_written,3d", proc.stdout)


if __name__ == "__main__":
    unittest.main()

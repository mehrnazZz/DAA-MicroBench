from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from microbench.config import load_yaml
from microbench.core import EpisodeEngine
from microbench.scenarios import generate_spawns_goals, materialize_official_suite, suite_defaults
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

        self.assertEqual(registry["official_smoke_generated"]["status"], "smoke")
        self.assertEqual(registry["official_smoke_generated"]["source"], "generated")
        self.assertEqual(registry["official_smoke_generated"]["acceptance_rule_count"], 10)
        self.assertIn("heterogeneous_priority_crossing_3d_medium", registry["official_smoke_generated"]["scenarios"])
        self.assertEqual(registry["official_3d_stress"]["status"], "pre_v1_official")
        self.assertEqual(registry["official_3d_stress"]["source"], "generated")
        self.assertEqual(registry["official_3d_stress"]["acceptance_rule_count"], 6)
        self.assertIn("merge_3d_hard", registry["official_3d_stress"]["scenarios"])
        self.assertIn("noncooperative_intruder_3d_hard", registry["official_3d_stress"]["scenarios"])
        self.assertIn("orca_with_staleness", registry["official_3d_stress"]["default_methods"])
        self.assertIn("orca_with_staleness", registry["official_agentic_stress"]["default_methods"])
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

    def test_generated_3d_stress_suite_carries_orca_acceptance_rules(self):
        with tempfile.TemporaryDirectory() as td:
            generated = materialize_official_suite("official_3d_stress", Path(td), overwrite=True)
            manifest = generated["manifest"]
            rules = manifest["acceptance"]["rules"]

            self.assertEqual(len(rules), 6)
            self.assertIn("orca_heuristic", manifest["default_methods"])
            self.assertIn("orca_with_staleness", manifest["default_methods"])
            self.assertTrue(any(rule["name"] == "orca_heuristic_3d_runtime_p95" for rule in rules))
            self.assertTrue(any(rule["name"] == "orca_with_staleness_3d_runtime_p95" for rule in rules))

    def test_generated_smoke_suite_manifest_carries_acceptance_rules(self):
        with tempfile.TemporaryDirectory() as td:
            generated = materialize_official_suite("official_smoke_generated", Path(td), overwrite=True)
            manifest = generated["manifest"]

            self.assertEqual(len(generated["scenario_paths"]), 3)
            self.assertEqual(manifest["status"], "smoke")
            self.assertEqual(manifest["dimensions"], ["2d", "3d"])
            self.assertEqual(manifest["default_methods"], ["baseline_goal", "orca_heuristic", "priority_yield"])
            self.assertEqual(manifest["n_agents"], [4])
            self.assertEqual(manifest["seeds"], [0])
            self.assertEqual(manifest["duration_override_s"], 8.0)
            self.assertEqual(manifest["acceptance"]["schema_version"], "0.1")
            self.assertEqual(len(manifest["acceptance"]["rules"]), 10)
            self.assertTrue(any(rule["name"] == "smoke_planner_fallbacks_clear" for rule in manifest["acceptance"]["rules"]))
            self.assertEqual(suite_defaults("official_smoke_generated")["acceptance"]["schema_version"], "0.1")
            cfg = load_yaml(generated["scenario_paths"][0])
            self.assertEqual(cfg["scenario"]["duration_s"], 8.0)

    def test_list_suites_cli_reports_registry(self):
        proc = subprocess.run(
            [sys.executable, "-m", "microbench.cli", "list-suites"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("official_agentic_stress,pre_v1_official,generated,3d", proc.stdout)
        self.assertIn("official_smoke_generated,smoke,generated,2d+3d,3,10", proc.stdout)
        self.assertIn("three_d,development,hand_written,3d", proc.stdout)

    def test_generated_smoke_suite_canonical_plan_is_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "microbench.cli",
                    "canonical-sweep",
                    "--suite",
                    "official_smoke_generated",
                    "--out-dir",
                    td,
                    "--print-plan",
                    "--no-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("suite: official_smoke_generated", proc.stdout)
        self.assertIn("total_runs: 9", proc.stdout)


if __name__ == "__main__":
    unittest.main()

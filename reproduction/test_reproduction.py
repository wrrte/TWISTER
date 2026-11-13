"""Scheduler and handoff regressions; no CUDA, training, Git or network calls."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce as r
import reproduction_cluster as cluster
from reproduction_seeds import occupied_seeds, workbook_seeds
from run_results import write_json


REAL_ROOT = r.ROOT
REFERENCES = r.references()
PLAN = cluster.load_plan()


def trial(seed, score=None, baseline=None, game="gopher", round_number=1,
          retained=True, status="complete", reference=(0, 100, 100)):
    return dict(game=game, seed=seed, round=round_number, retained=retained,
                status=status, stage="done" if status == "complete" else "warmup",
                full_flash=None if score is None else r.compare_score(score, reference),
                baseline=None if baseline is None else dict(metrics=dict(score=baseline)))


def state(trials, games=None):
    return dict(version=1, root=str(r.ROOT), search_id="search_test", round=1, trials=trials,
                config=dict(games=games or ["gopher"], start_seed=3090, seed_step=10,
                            references={"gopher": [0, 100, 100]}, override_config={}))


class GateTests(unittest.TestCase):
    def test_individual_cutoff_is_inclusive(self):
        for score, expected in ((90, True), (89.99, False)):
            item = trial(3090, score)
            gate = r.baseline_gate(state([item]), item)
            self.assertEqual(gate["accepted"], expected)
            self.assertFalse(gate["mean_passes"])

    def test_mean_includes_pruned_pairs_and_current_score(self):
        previous = [trial(3090, 150, 1, retained=False), trial(3100, 140, 999999)]
        current = trial(3110, 50)
        gate = r.baseline_gate(state(previous + [current]), current)
        self.assertFalse(gate["individual_passes"])
        self.assertTrue(gate["accepted"])
        self.assertTrue(gate["mean_passes"])
        self.assertEqual(gate["previous_pair_seeds"], [3090, 3100])
        self.assertAlmostEqual(gate["mean_score"], 340 / 3)

    def test_unpaired_other_game_and_incomplete_scores_are_excluded(self):
        previous = [trial(3090, 99999), trial(3100, 99999, 1, game="seaquest"),
                    trial(3110, 99999, 1, status="running")]
        current = trial(3120, 50)
        gate = r.baseline_gate(state(previous + [current]), current)
        self.assertFalse(gate["accepted"])
        self.assertEqual(gate["previous_pair_seeds"], [])

    def test_mean_cutoff_is_paper_plus_margin_and_inclusive(self):
        previous = trial(3090, 150, 1)
        for score, expected in ((70, True), (69.99, False)):
            current = trial(3100, score)
            self.assertEqual(r.baseline_gate(state([previous, current]), current)["accepted"], expected)

    def test_completed_pair_is_not_averaged_twice_on_resume(self):
        previous = trial(3090, 150, 1)
        current = trial(3100, 70, 1)
        gate = r.baseline_gate(state([previous, current]), current)
        self.assertEqual(gate["mean_score"], 110)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patches = [mock.patch.object(r, "ROOT", self.root),
                   mock.patch.object(r, "references", return_value=REFERENCES),
                   mock.patch("reproduction_seeds.occupied_seeds", return_value={})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.campaign = r.Campaign(self.root / "state.json")

    def save(self, trials):
        saved = state(trials)
        write_json(self.campaign.path, saved)
        return saved


class CampaignTests(WorkspaceTests):
    def test_mean_qualified_trial_actually_runs_baseline(self):
        previous = trial(3090, 150, 1)
        current = trial(3100, 70, status="running", round_number=2)
        saved = self.save([previous, current])
        saved["config"].update(python=sys.executable, no_wandb=True, override_config={})
        write_json(self.campaign.path, saved)
        _, _, paths = r.trial_paths(saved, current)

        def finish_baseline(command, environment, lease, logfile):
            self.assertEqual(command, [sys.executable, "baseline"])
            paths["baseline"].mkdir(parents=True)
            checkpoint = paths["baseline"] / "final.ckpt"
            checkpoint.write_text("test checkpoint")
            write_json(paths["baseline"] / "final_evaluation.json", dict(
                env_name="atari100k-gopher", seed=3100, retrieval_enabled=False,
                episodes=100, checkpoint=str(checkpoint), metrics=dict(score=1)))

        module = SimpleNamespace(branch_commands=lambda shared: (["python", "full"], ["python", "baseline"]),
                                 WARMUP_UNAVAILABLE_EXIT_CODE=75)
        lease = self.campaign._lease(current)
        self.addCleanup(lease.close)
        with mock.patch.dict(sys.modules, {"retrieval_runs": module}), \
                mock.patch.object(r, "run_command", side_effect=finish_baseline) as run:
            r.run_trial(self.campaign, saved, current, lease)
        run.assert_called_once()
        completed = self.campaign.read()["trials"][-1]
        self.assertTrue(completed["baseline_gate"]["mean_passes"])
        self.assertFalse(completed["full_flash"]["passes"])
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(len(r.successful_trials(self.campaign.read(), "gopher")), 2)

    def test_failed_gates_finish_without_running_baseline(self):
        current = trial(3090, 50, status="running")
        saved = self.save([current])
        saved["config"].update(python=sys.executable, no_wandb=True, override_config={})
        write_json(self.campaign.path, saved)
        module = SimpleNamespace(branch_commands=mock.Mock(), WARMUP_UNAVAILABLE_EXIT_CODE=75)
        lease = self.campaign._lease(current)
        self.addCleanup(lease.close)
        with mock.patch.dict(sys.modules, {"retrieval_runs": module}), mock.patch.object(r, "run_command") as run:
            r.run_trial(self.campaign, saved, current, lease)
        run.assert_not_called()
        self.assertEqual(self.campaign.read()["trials"][0]["status"], "complete")
        self.assertEqual(r.successful_trials(self.campaign.read(), "gopher"), [])


    def test_four_completed_pairs_stop_even_when_pruned_or_mean_qualified(self):
        trials = [trial(3090 + i * 10, 150, 1, round_number=i + 1) for i in range(3)]
        trials.append(trial(3120, 50, 1, round_number=4, retained=False))
        self.save(trials)
        kind, _, item, lease = self.campaign.claim()
        self.assertEqual(kind, "done")
        self.assertIsNone(item)
        self.assertIsNone(lease)

    def test_low_scoring_fifth_trial_keeps_warmup_until_baseline_finishes(self):
        trials = [trial(3090 + i * 10, 200 - i * 10, 1 if i < 3 else None,
                        round_number=i + 1) for i in range(4)]
        current = trial(3130, status="running", round_number=5)
        saved = self.save(trials + [current])
        directory = r.artifact_dir(saved, current)
        directory.mkdir(parents=True)
        write_json(directory / r.MARKER, r.identity(saved, current))
        (directory / "warmup-checkpoint").write_text("preserve until baseline completes")
        current = self.campaign.update(current, full_flash=r.compare_score(50, (0, 100, 100)))
        self.assertTrue(current["retained"])
        self.assertTrue(directory.exists())
        current = self.campaign.prepare_baseline(current)
        self.assertTrue(current["baseline_gate"]["accepted"])
        gate = copy.deepcopy(current["baseline_gate"])
        self.assertEqual(self.campaign.prepare_baseline(current)["baseline_gate"], gate)
        current = self.campaign.update(current, baseline=dict(metrics=dict(score=1)), status="complete")
        self.assertFalse(current["retained"])
        self.assertFalse(directory.exists())
        self.assertEqual(len(r.successful_trials(self.campaign.read(), "gopher")), 4)
        self.assertEqual(self.campaign.claim()[0], "done")

    def test_imported_attempts_advance_seed_and_round(self):
        saved = self.save([])
        saved["imported_history"] = {"gopher": [trial(3130, round_number=5, status="released")]}
        write_json(self.campaign.path, saved)
        kind, _, item, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(kind, "trial")
        self.assertEqual(item["seed"], 3140)
        self.assertEqual(item["round"], 6)


class WorkbookTests(unittest.TestCase):
    @unittest.skipUnless((REAL_ROOT / "results/performance_twister.xlsx").is_file(), "Excel export is local-only")
    def test_real_workbook_reserves_legacy_running_and_finished_gopher_seeds(self):
        workbook = workbook_seeds(REAL_ROOT / "results/performance_twister.xlsx")["gopher"]
        seeds = occupied_seeds(REAL_ROOT)["gopher"]
        self.assertTrue(workbook.issubset(seeds))
        self.assertTrue({3090, 3100, 3110, 3120, 9997}.issubset(seeds))
        saved = state([])
        seed = r.next_seed(saved, "gopher")
        self.assertGreaterEqual(seed, 3130)
        self.assertNotIn(seed, seeds)
        self.assertEqual((seed - 3090) % 10, 0)
        saved["config"]["start_seed"] = 9997
        saved["config"]["seed_step"] = 1
        self.assertEqual(r.next_seed(saved, "gopher"), 9998)

    def test_manifest_protects_seeds_on_a_server_without_excel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "reproduction/seed_reservations.json"
            manifest.parent.mkdir()
            manifest.write_bytes((REAL_ROOT / "reproduction/seed_reservations.json").read_bytes())
            self.assertIn(3120, occupied_seeds(root)["gopher"])
            with mock.patch.object(r, "ROOT", root):
                self.assertEqual(r.next_seed(state([]), "gopher"), 3130)


class RecoveryTests(WorkspaceTests):
    def setUp(self):
        super().setUp()
        self.metadata = mock.patch("reproduction_recovery.checkpoint_metadata", side_effect=self.checkpoint_metadata)
        self.metadata.start()
        self.addCleanup(self.metadata.stop)
        config = self.root / "configs/twister.py"
        config.parent.mkdir()
        config.write_text("# original model configuration")

    def checkpoint_metadata(self, python, path, modified, size, project_root):
        shared = Path(path).parent
        return dict(model_step=50000, source_dirs=[str(shared.parent / "ReplayBuffer")],
                    source_files=[str(shared.parent / "ReplayBuffer/50000.torch")])

    def resources(self, saved, item):
        directory, run_name, paths = r.trial_paths(saved, item)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / r.MARKER, r.identity(saved, item))
        shared = paths["warmup"] / "shared_warmup"
        shared.mkdir(parents=True)
        (shared / "warmup.ckpt").write_text("original model optimizer RNG checkpoint")
        replay = paths["warmup"] / "ReplayBuffer"
        replay.mkdir()
        (replay / "50000.torch").write_text("original replay")
        write_json(shared / "run.json", dict(
            version=1, run_name=run_name, env_name="atari100k-" + item["game"],
            override_config=dict(seed=item["seed"]), config_file="configs/twister.py", cli={},
            checkpoint="warmup.ckpt", warmup_model_steps=50000))
        return directory, paths

    def test_rejected_gate_is_recomputed_after_new_pair(self):
        old = trial(3090, 80)
        saved = self.save([old])
        old_gate = self.campaign.prepare_baseline(old)["baseline_gate"]
        self.assertFalse(old_gate["accepted"])
        saved = self.campaign.read()
        saved["trials"].append(trial(3100, 160, 1, round_number=2))
        write_json(self.campaign.path, saved)
        self.resources(saved, old)
        kind, _, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual((kind, recovered["seed"], recovered["stage"]), ("trial", 3090, "baseline"))
        self.assertEqual(len(self.campaign.read()["trials"]), 2)
        self.assertTrue(recovered["baseline_gate"]["mean_passes"])
        self.assertEqual(recovered["baseline_gate_history"], [old_gate])
        self.assertEqual(recovered["full_flash"], old["full_flash"])

    def test_full_budget_off_warmup_is_evaluated_without_replay_or_training(self):
        old = trial(3090, 80)
        saved = self.save([old, trial(3100, 160, 1, round_number=2)])
        directory, _, paths = r.trial_paths(saved, old)
        paths["warmup"].mkdir(parents=True)
        write_json(directory / r.MARKER, r.identity(saved, old))
        checkpoint = paths["warmup"] / "full_budget.ckpt"
        checkpoint.write_text("existing OFF weights")
        candidate = dict(version=1, env_name="atari100k-gopher", seed=3090,
                         model_step=100000, training_budget=100000, uninterrupted=True,
                         paused_environments=0, checkpoint=str(checkpoint))
        write_json(paths["warmup"] / "baseline_candidate.json", candidate)
        _, _, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(recovered["seed"], 3090)
        self.assertEqual(recovered["baseline_recovery"]["mode"], "evaluate_warmup")
        self.assertEqual(recovered["baseline_recovery"]["checkpoint"], str(checkpoint))

    def test_declined_gate_can_be_rechecked_but_accepted_gate_stays_fixed(self):
        old = trial(3090, 80)
        self.save([old])
        self.assertFalse(self.campaign.prepare_baseline(old)["baseline_gate"]["accepted"])
        saved = self.campaign.read()
        saved["trials"].append(trial(3100, 160, 1, round_number=2))
        write_json(self.campaign.path, saved)
        accepted = self.campaign.prepare_baseline(old)["baseline_gate"]
        self.assertTrue(accepted["accepted"])
        saved = self.campaign.read()
        saved["trials"].append(trial(3110, 50, 1, round_number=3))
        write_json(self.campaign.path, saved)
        self.assertEqual(self.campaign.prepare_baseline(old)["baseline_gate"], accepted)

    def test_highest_missing_replay_is_skipped_for_next_usable_seed(self):
        best = trial(3090, 80)
        second = trial(3100, 70, round_number=2)
        paired = trial(3110, 200, 1, round_number=3)
        saved = self.save([best, second, paired])
        _, paths = self.resources(saved, best)
        (paths["warmup"] / "ReplayBuffer/50000.torch").unlink()
        self.resources(saved, second)
        _, _, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(recovered["seed"], 3100)
        self.assertIn("Replay files", self.campaign.read()["trials"][0]["baseline_recovery_skipped"]["reason"])

    def test_recovery_runs_only_saved_baseline_branch(self):
        old = trial(3090, 80)
        paired = trial(3100, 160, 1, round_number=2)
        saved = self.save([old, paired])
        saved["config"].update(python=sys.executable, no_wandb=True, override_config={})
        write_json(self.campaign.path, saved)
        directory, paths = self.resources(saved, old)
        checkpoint = paths["warmup"] / "shared_warmup/warmup.ckpt"
        original = checkpoint.read_bytes()
        _, claimed_state, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)

        def finish(command, environment, lease, logfile):
            self.assertEqual(command, [sys.executable, "OFF"])
            paths["baseline"].mkdir(parents=True)
            final = paths["baseline"] / "final.ckpt"
            final.write_text("baseline final weights")
            write_json(paths["baseline"] / "final_evaluation.json", dict(
                env_name="atari100k-gopher", seed=3090, retrieval_enabled=False,
                episodes=100, checkpoint=str(final), metrics=dict(score=10)))

        commands = mock.Mock(return_value=(["python", "ON"], ["python", "OFF"]))
        module = SimpleNamespace(branch_commands=commands, WARMUP_UNAVAILABLE_EXIT_CODE=75)
        with mock.patch.dict(sys.modules, {"retrieval_runs": module}), \
                mock.patch.object(r, "run_command", side_effect=finish) as run:
            r.run_trial(self.campaign, claimed_state, recovered, lease)
        run.assert_called_once()
        self.assertEqual(checkpoint.read_bytes(), original)
        self.assertEqual(self.campaign.read()["trials"][0]["full_flash"], old["full_flash"])
        self.assertEqual(len(r.successful_trials(self.campaign.read(), "gopher")), 2)

    def test_backfill_in_score_order_stops_at_four_pairs(self):
        candidates = [trial(3090 + i * 10, score, round_number=i + 1) for i, score in enumerate((80, 70, 60))]
        paired = [trial(3120 + i * 10, score, 1, round_number=i + 4) for i, score in enumerate((200, 190))]
        saved = self.save(candidates + paired)
        for item in candidates:
            self.resources(saved, item)
        for seed in (3090, 3100):
            _, _, item, lease = self.campaign.claim()
            self.assertEqual(item["seed"], seed)
            self.campaign.update(item, baseline=dict(metrics=dict(score=1)), status="complete", stage="done")
            lease.close()
        self.assertEqual(len(r.successful_trials(self.campaign.read(), "gopher")), 4)
        self.assertEqual(self.campaign.claim()[0], "done")
        self.assertIsNone(self.campaign.read()["trials"][2]["baseline"])

    def test_updated_average_can_stop_further_backfill(self):
        candidates = [trial(3090, 80), trial(3100, 70, round_number=2)]
        saved = self.save(candidates + [trial(3110, 160, 1, round_number=3)])
        for item in candidates:
            self.resources(saved, item)
        _, _, item, lease = self.campaign.claim()
        self.assertEqual(item["seed"], 3090)
        self.campaign.update(item, baseline=dict(metrics=dict(score=1)), status="complete")
        lease.close()
        _, _, next_item, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(next_item["seed"], 3120)
        self.assertIsNone(next_item["full_flash"])

    def test_pending_cleanup_is_cancelled_when_historical_baseline_qualifies(self):
        old = trial(3090, 80, retained=False)
        old["cleanup_pending"] = True
        saved = self.save([old, trial(3100, 160, 1, round_number=2)])
        directory, _ = self.resources(saved, old)
        _, _, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(recovered["seed"], 3090)
        self.assertTrue(directory.is_dir())
        self.assertFalse(recovered["cleanup_pending"])

    def test_new_full_score_does_not_delete_history_before_its_pair_completes(self):
        old = trial(3090, 80, retained=False)
        old["cleanup_pending"] = True
        current = trial(3100, status="running", round_number=2)
        saved = self.save([old, current])
        directory, _ = self.resources(saved, old)
        current = self.campaign.update(current, full_flash=r.compare_score(160, (0, 100, 100)))
        self.assertTrue(directory.exists())
        self.campaign.update(current, baseline=dict(metrics=dict(score=1)), status="complete")
        _, _, recovered, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(recovered["seed"], 3090)
        self.assertTrue(directory.exists())

    def test_no_baseline_is_claimed_after_four_pairs_even_if_interrupted(self):
        pairs = [trial(3090 + i * 10, 150, 1, round_number=i + 1) for i in range(4)]
        self.save(pairs + [trial(3130, 80, round_number=5, status="interrupted")])
        self.assertEqual(self.campaign.claim()[0], "done")

    def test_imported_scores_without_artifacts_never_start_from_scratch(self):
        old = trial(3090, 80)
        old.update(imported_from="b", artifact_search_id="search_source")
        self.save([old, trial(3100, 160, 1, round_number=2)])
        _, _, item, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(item["seed"], 3110)
        self.assertIsNone(item["full_flash"])

    def test_imported_original_artifacts_can_be_used_when_accessible(self):
        old = trial(3090, 80)
        old.update(imported_from="b", artifact_search_id="search_source")
        saved = self.save([old, trial(3100, 160, 1, round_number=2)])
        directory, _ = self.resources(saved, old)
        _, _, item, lease = self.campaign.claim()
        self.addCleanup(lease.close)
        self.assertEqual(item["seed"], 3090)
        self.assertIn("search_source", str(directory))
        self.assertEqual(r.identity(saved, item)["search_id"], "search_source")


class HandoffTests(WorkspaceTests):
    def setUp(self):
        super().setUp()
        # Keep testing the original single-owner handoff independently of the
        # shared-game overlay, which has its own integration checks below.
        plan = copy.deepcopy(PLAN)
        plan.pop("shared_games", None)
        write_json(cluster.plan_path(), plan)
        self.b = r.Campaign(self.root / "b.json")
        self.c = r.Campaign(self.root / "c.json")
        old_b = [game for game in PLAN["servers"]["b"]["games"] if game not in {"road_runner", "up_n_down"}] + ["gopher"]
        old_c = ["road_runner", "seaquest", "up_n_down"]
        for campaign, server, games in ((self.b, "b", old_b), (self.c, "c", old_c)):
            saved = state([], games)
            saved["search_id"] = "search_" + server
            saved["config"].update(server=server, references={game: REFERENCES[game] for game in games})
            write_json(campaign.path, saved)
        bstate = self.b.read()
        bstate["trials"] = [trial(3090 + i * 10, 3000 if i < 3 else None, round_number=i + 1,
                                 reference=REFERENCES["gopher"], status="complete" if i < 3 else "interrupted")
                            for i in range(4)]
        write_json(self.b.path, bstate)
        cstate = self.c.read()
        cstate["trials"] = [trial(3090, 18303, 18000, game="road_runner", reference=REFERENCES["road_runner"]),
                            trial(3100, 32661, 30000, game="road_runner", round_number=2, reference=REFERENCES["road_runner"]),
                            trial(3090, 13402, 13000, game="up_n_down", reference=REFERENCES["up_n_down"]),
                            trial(3100, game="seaquest", status="running", round_number=2)]
        write_json(self.c.path, cstate)
        self.bsettings = dict(server="b", state=str(self.b.path))
        self.csettings = dict(server="c", state=str(self.c.path))

    def release_both(self):
        cluster.release_history(self.c, self.csettings)
        cluster.release_history(self.b, self.bsettings)

    def test_exchange_preserves_source_artifacts_and_continues_seeds(self):
        cstate = self.c.read()
        original = r.artifact_dir(cstate, cstate["trials"][0])
        original.mkdir(parents=True)
        (original / "checkpoint").write_text("original weights")
        seaquest = cstate["trials"][-1]
        lease = self.c._lease(seaquest)
        self.addCleanup(lease.close)
        # C can release the two idle games while Seaquest continues running.
        self.release_both()
        cluster.attach_server(self.b, "b")
        cluster.attach_server(self.c, "c")
        self.assertEqual((original / "checkpoint").read_text(), "original weights")
        self.assertEqual(next(t for t in self.c.read()["trials"] if t["game"] == "seaquest"), seaquest)
        self.assertEqual(len(r.successful_trials(self.b.read(), "road_runner")), 2)
        self.assertEqual(r.next_seed(self.b.read(), "road_runner"), 3110)
        self.assertEqual(r.next_seed(self.b.read(), "up_n_down"), 3100)
        kind, _, item, gopher_lease = self.c.claim()
        self.addCleanup(gopher_lease.close)
        self.assertEqual((kind, item["game"], item["seed"], item["round"]), ("trial", "gopher", 3130, 5))
        self.assertEqual(self.b.read()["config"]["games"], PLAN["servers"]["b"]["games"])
        self.assertEqual(self.c.read()["config"]["games"], ["seaquest", "gopher"])
        # The interrupted Gopher trial left on B is never resumed there.
        _, _, item, blease = self.b.claim()
        self.addCleanup(blease.close)
        self.assertNotEqual(item["game"], "gopher")

    def test_live_outgoing_worker_prevents_any_release(self):
        active = self.b.read()["trials"][-1]
        lease = self.b._lease(active)
        with self.assertRaisesRegex(ValueError, "still running"):
            cluster.release_history(self.b, self.bsettings)
        self.assertNotIn("released_games", self.b.read())
        self.assertFalse(cluster.handoff_path("gopher").exists())
        lease.close()

    def test_other_game_keeps_running_during_gopher_handoff(self):
        saved = self.b.read()
        other = trial(3090, game="hero", status="running", reference=REFERENCES["hero"])
        saved["trials"].append(other)
        write_json(self.b.path, saved)
        lease = self.b._lease(other)
        self.addCleanup(lease.close)
        self.release_both()
        cluster.attach_server(self.b, "b")
        updated = self.b.read()
        self.assertEqual(next(t for t in updated["trials"] if t["game"] == "hero"), other)
        # Pre-update workers require every active-table game to be assigned.
        self.assertTrue(all(t["game"] in updated["config"]["games"] for t in updated["trials"]))
        self.assertEqual(len(updated["released_trials"]["gopher"]), 4)
        # Their seed allocator sees imported attempts, preserving continuity.
        attempts = sum(t["game"] == "road_runner" for t in updated["trials"])
        self.assertEqual(updated["config"]["start_seed"] + attempts * 10, 3110)
        self.b.update(other, stage="full_flash")
        self.assertEqual(next(t for t in self.b.read()["trials"] if t["game"] == "hero")["stage"], "full_flash")

    def test_missing_source_snapshot_blocks_adoption(self):
        cluster.release_history(self.c, self.csettings)
        with self.assertRaisesRegex(ValueError, "gopher history is not ready"):
            cluster.attach_server(self.c, "c")
        self.assertEqual(self.c.read()["config"]["games"], ["road_runner", "seaquest", "up_n_down"])

    def test_release_and_import_are_idempotent(self):
        self.release_both()
        original = cluster.handoff_path("gopher").read_bytes()
        cluster.release_history(self.b, self.bsettings)
        self.assertEqual(cluster.handoff_path("gopher").read_bytes(), original)
        cluster.attach_server(self.c, "c")
        cluster.attach_server(self.c, "c")
        self.assertEqual(len(r.game_trials(self.c.read(), "gopher")), 4)

    def test_imported_pairs_participate_in_the_mean_gate(self):
        self.release_both()
        cluster.attach_server(self.b, "b")
        current = trial(3110, 15000, game="road_runner", reference=REFERENCES["road_runner"])
        gate = r.baseline_gate(self.b.read(), current)
        self.assertFalse(gate["individual_passes"])
        self.assertTrue(gate["mean_passes"])
        self.assertEqual(gate["previous_pair_seeds"], [3090, 3100])

    def test_duplicate_seed_snapshot_is_rejected(self):
        self.release_both()
        path = cluster.handoff_path("gopher")
        snapshot = json.loads(path.read_text())
        snapshot["trials"].append(snapshot["trials"][0])
        write_json(path, snapshot)
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            cluster.attach_server(self.c, "c")


class SharedGamesTests(WorkspaceTests):
    def setUp(self):
        super().setUp()
        write_json(cluster.plan_path(), PLAN)
        for source in (REAL_ROOT / "reproduction/handoffs").glob("*.json"):
            target = cluster.handoff_path(source.stem)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    def server(self, name):
        campaign = r.Campaign(self.root / f"{name}.json")
        saved = state([], list(PLAN["servers"][name]["games"]))
        saved["search_id"] = "search_" + name
        saved["config"].update(server=name, references=copy.deepcopy(REFERENCES))
        write_json(campaign.path, saved)
        cluster.attach_server(campaign, name)
        return campaign

    def claim(self, campaign):
        result = campaign.claim()
        if result[3] is not None:
            self.addCleanup(result[3].close)
        return result

    def pair(self, campaign, game, seed):
        saved = campaign.read()
        number = 1 + max((item["round"] for item in r.game_trials(saved, game)), default=0)
        saved["trials"].append(trial(seed, 1000000, 1, game=game, round_number=number,
                                    reference=REFERENCES[game]))
        write_json(campaign.path, saved)

    def finish_others(self, campaign, remaining):
        for game in campaign.read()["config"]["games"]:
            if game not in remaining:
                for seed in (5000, 5010, 5020, 5030):
                    self.pair(campaign, game, seed)

    def report(self, server, game, seeds=None, count=0):
        progress = dict(successful_pairs=count if seeds is None else len(seeds))
        if seeds is not None:
            progress["successful_pair_seeds"] = seeds
        write_json(self.root / "reproduction/reports" / f"{server}.json",
                   dict(version=1, server=server, games={game: progress}))

    def test_every_server_includes_both_shared_games_once(self):
        for name in PLAN["servers"]:
            campaign = self.server(name)
            games = campaign.read()["config"]["games"]
            for game in ("gopher", "krull"):
                self.assertEqual(games.count(game), 1)
            self.assertEqual(set(games) - {"gopher", "krull"},
                             set(PLAN["servers"][name]["games"]) - {"gopher", "krull"})
            self.assertFalse(cluster.apply_assignment(campaign.read()))

    def test_shared_games_have_normal_priority(self):
        saved = self.save([])
        saved["config"]["games"] = ["alien", "gopher", "krull"]
        saved["config"]["references"] = copy.deepcopy(REFERENCES)
        write_json(self.campaign.path, saved)
        self.assertEqual([self.claim(self.campaign)[2]["game"] for _ in range(3)],
                         ["alien", "gopher", "krull"])

    def test_both_games_allow_four_concurrent_distinct_trials(self):
        for game in ("gopher", "krull"):
            with self.subTest(game=game):
                campaign = r.Campaign(self.root / f"parallel_{game}.json")
                saved = state([], [game])
                saved["config"]["references"] = copy.deepcopy(REFERENCES)
                write_json(campaign.path, saved)
                trials = [self.claim(campaign)[2] for _ in range(4)]
                self.assertEqual([item["seed"] for item in trials], [3090, 3100, 3110, 3120])
                self.assertEqual(len({item["round"] for item in trials}), 4)
                self.assertEqual(self.claim(campaign)[0], "wait")
                campaign.update(trials[0], status="complete", full_flash=r.compare_score(0, (0, 100, 100)))
                self.assertEqual(self.claim(campaign)[2]["seed"], 3130)

    def test_other_games_remain_serial(self):
        saved = self.save([])
        saved["config"].update(games=["alien"], references=copy.deepcopy(REFERENCES))
        write_json(self.campaign.path, saved)
        self.assertEqual(self.claim(self.campaign)[0], "trial")
        self.assertEqual(self.claim(self.campaign)[0], "wait")

    def test_server_seed_streams_preserve_owner_and_avoid_overlap(self):
        expected = {"gopher": {"a": 3091, "b": 3092, "c": 3130},
                    "krull": {"a": 3091, "b": 3090, "c": 3092}}
        for game in expected:
            seeds = []
            for name in PLAN["servers"]:
                saved = self.server(name).read()
                seed = r.next_seed(saved, game)
                self.assertEqual(seed, expected[game][name])
                seeds.extend([seed, seed + saved["config"]["seed_step"]])
            self.assertEqual(len(seeds), len(set(seeds)))

    def test_owner_seed_stream_skips_manifest_and_workbook_reservations(self):
        campaign = self.server("b")
        with mock.patch("reproduction_seeds.occupied_seeds", return_value={"krull": {3090, 3100, 3110}}):
            self.assertEqual(r.next_seed(campaign.read(), "krull"), 3120)

    def test_global_pair_goals_are_independent_and_ignore_own_stale_report(self):
        campaign = self.server("a")
        self.finish_others(campaign, {"gopher", "krull"})
        self.report("a", "gopher", count=100)
        self.report("c", "gopher", seeds=[3130, 3140, 3150, 3160])
        self.report("b", "krull", seeds=[3090, 3100, 3110, 3120])
        counts = r.successful_pair_counts(campaign.read())
        self.assertEqual((counts["gopher"], counts["krull"]), (4, 4))
        self.assertEqual(self.claim(campaign)[0], "done")

    def test_global_goal_for_one_game_leaves_other_games_available(self):
        campaign = self.server("a")
        self.finish_others(campaign, {"gopher", "krull"})
        self.report("c", "gopher", seeds=[3130, 3140, 3150, 3160])
        self.assertEqual(self.claim(campaign)[2]["game"], "krull")

    def test_local_and_remote_pairs_reserve_remaining_slots_until_baseline_finishes(self):
        campaign = self.server("a")
        self.finish_others(campaign, {"krull"})
        self.pair(campaign, "krull", 3091)
        self.pair(campaign, "krull", 3101)
        self.report("b", "krull", seeds=[3090])
        item = self.claim(campaign)[2]
        self.assertEqual(item["seed"], 3111)
        campaign.update(item, stage="baseline", full_flash=r.compare_score(1000000, REFERENCES["krull"]))
        self.assertEqual(r.successful_pair_counts(campaign.read())["krull"], 3)
        self.assertEqual(self.claim(campaign)[0], "wait")
        campaign.update(item, status="complete", baseline=dict(metrics=dict(score=1)))
        self.assertEqual(self.claim(campaign)[0], "done")

    def test_inflight_shared_trial_can_resume_after_remote_goal_is_met(self):
        campaign = self.server("a")
        self.finish_others(campaign, {"gopher"})
        _, _, item, lease = self.claim(campaign)
        self.report("c", "gopher", seeds=[3130, 3140, 3150, 3160])
        self.assertEqual(self.claim(campaign)[0], "done")
        campaign.update(item, status="interrupted")
        lease.close()
        self.assertEqual(self.claim(campaign)[2]["seed"], item["seed"])

    def test_source_keeps_new_shared_trials_separate_from_released_history(self):
        campaign = self.server("b")
        saved = campaign.read()
        snapshot = json.loads(cluster.handoff_path("gopher").read_text())
        saved["search_id"] = snapshot["search_id"]
        saved["released_games"] = {"gopher": snapshot["id"]}
        saved["released_trials"] = {"gopher": [trial(3090, 1000000, 1, reference=REFERENCES["gopher"])]}
        write_json(campaign.path, saved)
        self.finish_others(campaign, {"gopher"})
        _, _, item, _ = self.claim(campaign)
        self.assertEqual(item["seed"], 3092)
        self.assertTrue(item["shared_trial"])
        cluster.attach_server(campaign, "b")
        self.assertIn(item["seed"], [record["seed"] for record in campaign.read()["trials"] if record["game"] == "gopher"])
        self.assertEqual(self.claim(campaign)[2]["seed"], 3102)
        report = json.loads(cluster.export_report(campaign, "b").read_text())
        self.assertEqual(report["games"]["gopher"]["successful_pair_seeds"], [])
        self.assertEqual(report["games"]["gopher"]["successful_pairs"], 0)
        original = cluster.handoff_path("gopher").read_bytes()
        # Re-running the frozen handoff does not reserve or export new shared
        # trials, including ones whose leases are held by live workers.
        cluster.release_history(campaign, {"server": "b"})
        self.assertEqual(cluster.handoff_path("gopher").read_bytes(), original)

    def test_imported_pairs_and_old_source_report_are_not_counted_twice(self):
        snapshot = json.loads(cluster.handoff_path("gopher").read_text())
        snapshot["trials"][0]["baseline"] = dict(metrics=dict(score=1))
        write_json(cluster.handoff_path("gopher"), snapshot)
        campaign = self.server("c")
        self.report("b", "gopher", count=1)
        self.assertEqual(r.successful_pair_counts(campaign.read())["gopher"], 1)
        exported = json.loads(cluster.export_report(campaign, "c").read_text())
        self.assertEqual(exported["games"]["gopher"]["successful_pair_seeds"], [3090])
        self.assertEqual(r.successful_pair_counts(self.server("a").read())["gopher"], 1)

    def test_remote_pair_seed_lists_are_deduplicated(self):
        campaign = self.server("a")
        self.report("b", "gopher", seeds=[3090, 3092])
        self.report("c", "gopher", seeds=[3090, 3130])
        self.assertEqual(r.successful_pair_counts(campaign.read())["gopher"], 3)

    def test_legacy_owner_counts_still_contribute(self):
        campaign = self.server("a")
        self.pair(campaign, "krull", 3091)
        self.report("b", "krull", count=3)
        self.assertEqual(r.successful_pair_counts(campaign.read())["krull"], 4)

    def test_status_projects_shared_assignment_without_writing_state(self):
        campaign = self.server("a")
        saved = campaign.read()
        saved["config"]["games"] = list(PLAN["servers"]["a"]["games"])
        for key in ("shared_games", "shared_seed_offsets"):
            saved["config"].pop(key)
        write_json(campaign.path, saved)
        original = campaign.path.read_bytes()
        output = io.StringIO()
        with redirect_stdout(output):
            r.main(["--state", str(campaign.path), "--status"])
        for game in ("gopher", "krull"):
            self.assertIn(f"{game}: 0/4 successful pairs", output.getvalue())
        self.assertEqual(campaign.path.read_bytes(), original)

    def test_shared_games_cannot_be_moved(self):
        for name, game in (("b", "krull"), ("c", "gopher")):
            with self.assertRaisesRegex(ValueError, "Shared games"):
                cluster.move_game(self.server(name), {"server": name}, game, "a")

    def test_invalid_shared_game_plan_is_rejected(self):
        plan = copy.deepcopy(PLAN)
        plan["shared_games"] = ["gopher", "gopher"]
        write_json(cluster.plan_path(), plan)
        with self.assertRaisesRegex(ValueError, "Shared games"):
            cluster.load_plan()


if __name__ == "__main__":
    unittest.main()

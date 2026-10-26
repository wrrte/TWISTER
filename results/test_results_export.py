"""Final-score extraction and Excel aggregation without W&B network calls.

Run from the TWISTER root: python -m unittest discover -s results -p 'test_*.py'
"""

import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(RESULTS))
import classify_wandb_runs as export
from convert_csv_to_excel import (
    HNS_DELTA_COLUMN, PAIRED_MEAN_COLUMN, SCORE_DELTA_COLUMN,
    comparison_metrics, convert_csv_to_excel, reference_scores,
)


def make_run(**changes):
    values = dict(
        name="callbacks/krull_seed9997/atari100k-krull", id="cfmagzmy",
        config={}, summary={export.SCORE_KEY: 5465.9}, state="finished",
        commit="abc1234", created_at="2026-10-21T01:11:28Z",
        url="https://wandb.ai/choemj-kaist/nnet/runs/cfmagzmy", metadata={},
        scan_history=mock.Mock(return_value=iter([])),
    )
    values.update(changes)
    return SimpleNamespace(**values)


class ExtractionTests(unittest.TestCase):
    def test_summary_score_identity_and_no_retrieval_guess(self):
        run = make_run()
        row = export.classify_run(run)
        self.assertEqual((row["Game"], row["Seed"], row[export.SCORE_KEY]),
                         ("krull", 9997, 5465.9))
        self.assertEqual(row["Retrieval Enable"], "Unknown")
        self.assertEqual(row["Score Source"], "summary")
        run.scan_history.assert_not_called()

    def test_confirmed_historical_run_overrides_do_not_change_other_runs(self):
        overrides = export.load_run_overrides()
        confirmed = export.classify_run(make_run(), config_overrides=overrides.get("cfmagzmy"))
        self.assertEqual(confirmed["Retrieval Enable"], "False")
        self.assertEqual(confirmed["Config"], "Retrieval OFF")
        unconfirmed = export.classify_run(make_run(id="future-run"),
                                         config_overrides=overrides.get("future-run"))
        self.assertEqual(unconfirmed["Retrieval Enable"], "Unknown")

    def test_only_confirmed_roadrunner_runs_are_recorded_as_baseline(self):
        overrides = export.load_run_overrides()
        for run_id in ("xetdccow", "6pn495jh"):
            row = export.classify_run(make_run(
                id=run_id, name="callbacks/road_runner_seed9999/atari100k-road_runner",
            ), config_overrides=overrides.get(run_id))
            self.assertEqual(row["Retrieval Enable"], "False")
            self.assertEqual(row["Config"], "Retrieval OFF")
        self.assertNotIn("pwa555yc", overrides)  # Different run, missing final score.
        self.assertNotIn("f4uu5knq", overrides)  # Amidar remains Unknown.

    def test_zero_negative_and_last_summary(self):
        for value in (0, -20.7, {"last": 0}):
            row = export.classify_run(make_run(summary={export.SCORE_KEY: value}))
            self.assertEqual(row["Score Status"], "ok")
            self.assertEqual(row[export.SCORE_KEY], value.get("last") if isinstance(value, dict) else value)

    def test_history_fallback_uses_last_finite_exact_metric(self):
        run = make_run(summary={export.SCORE_KEY: float("nan"), "eval/score": 999})
        run.scan_history.return_value = iter([
            {export.SCORE_KEY: 40}, {export.SCORE_KEY: 0},
            {export.SCORE_KEY: -3}, {export.SCORE_KEY: float("inf")},
        ])
        row = export.classify_run(run)
        self.assertEqual(row[export.SCORE_KEY], -3)
        self.assertEqual(row["Score Source"], "history")
        run.scan_history.assert_called_once_with(keys=[export.SCORE_KEY], page_size=1000)

    def test_missing_never_substitutes_periodic_or_tensorboard_scores(self):
        run = make_run(summary={"eval/score": 12, "Evaluation-final/0/score": 42})
        row = export.classify_run(run)
        self.assertEqual(row[export.SCORE_KEY], "N/A")
        self.assertEqual(row["Score Status"], "missing")
        run.scan_history.reset_mock()
        export.classify_run(run, history_fallback=False)
        run.scan_history.assert_not_called()

    def test_incomplete_history_is_an_error_even_after_a_valid_row(self):
        def broken_history(**kwargs):
            yield {export.SCORE_KEY: 1}
            raise RuntimeError("download interrupted")
        row = export.classify_run(make_run(summary={}, scan_history=broken_history))
        self.assertEqual(row["Score Status"], "error")
        self.assertEqual(row[export.SCORE_KEY], "N/A")
        self.assertIn("download interrupted", row["Error"])

    def test_branch_suffixes_config_precedence_and_warmup(self):
        for suffix, expected in (("O", "True"), ("X", "False"), ("warmup", "Both")):
            run = make_run(name=f"callbacks/krull_seed9997_{suffix}/atari100k-krull",
                           config={"seed": 42, "retrieval_enabled": "Both"}, summary={})
            row = export.classify_run(run)
            self.assertEqual(row["Seed"], 42)
            self.assertEqual(row["Retrieval Enable"], expected)
            if suffix == "warmup":
                self.assertEqual(row["Score Status"], "warmup")
                run.scan_history.assert_not_called()

    def test_config_settings_remain_separate_and_seed_metadata_is_supported(self):
        run = make_run(name="callbacks/test/dmc-Acrobot-swingup",
                       config={"retrieval_enabled": True, "retrieval.target": 16,
                               "retrieval": {"warmup_steps": 50000}},
                       metadata={"args": ["--seed=9"]})
        row = export.classify_run(run)
        self.assertEqual(row["Seed"], 9)
        self.assertEqual(row["Environment"], "dmc-Acrobot-swingup")
        self.assertEqual(row["Retrieval Target"], 16)
        self.assertIn('"target": 16', row["Config"])
        self.assertIn('"warmup_steps": 50000', row["Config"])

    def test_cli_uses_api_filters_and_leaves_previous_csv_on_failed_listing(self):
        sdk = mock.Mock()
        api = sdk.Api.return_value
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.csv"
            api.runs.return_value = [make_run()]
            with mock.patch.dict(sys.modules, {"wandb": sdk}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = export.main(["--entity", "team", "--project", "test", "--run-id",
                                    "cfmagzmy", "--states", "finished", "--output", str(output)])
                self.assertEqual(code, 0)
                filters = api.runs.call_args.kwargs["filters"]
                self.assertEqual(filters["name"], {"$in": ["cfmagzmy"]})
                self.assertEqual(filters["state"], {"$in": ["finished"]})
                before = output.read_bytes()
                def broken_runs(*args, **kwargs):
                    yield make_run()
                    raise RuntimeError("project unavailable")
                api.runs.side_effect = broken_runs
                self.assertEqual(export.main(["--output", str(output)]), 1)
                self.assertEqual(output.read_bytes(), before)
            sdk.init.assert_not_called()


class WorkbookTests(unittest.TestCase):
    def test_custom_overrides_keep_distinct_on_configurations(self):
        from openpyxl import load_workbook
        rows = [export.classify_run(make_run(
            id=f"custom{target}", config={"seed": 1, "retrieval_enabled": True,
                                          "retrieval": {"target": target}},
        )) for target in (16, 32)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = export.write_csv(rows, root / "scores.csv")
            corrections = root / "overrides.json"
            corrections.write_text(
                '{"custom16": {"retrieval_enabled": true}, '
                '"custom32": {"retrieval_enabled": true}}', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                target = convert_csv_to_excel(source, root / "scores.xlsx", corrections)
            book = load_workbook(target)
            try:
                labels = {row[1] for row in book["Results"].values}
                self.assertIn('Retrieval ON {"target": 16}', labels)
                self.assertIn('Retrieval ON {"target": 32}', labels)
            finally:
                book.close()

    def test_old_csv_roadrunner_unknowns_are_corrected_before_aggregation(self):
        from openpyxl import load_workbook
        rows = [export.classify_run(make_run(
            id=run_id, name=f"callbacks/road_runner_seed{seed}/atari100k-road_runner",
            summary={export.SCORE_KEY: score},
        )) for run_id, seed, score in (
            ("xetdccow", 9999, 17628), ("6pn495jh", 9998, 23434),
            ("pwa555yc", 9999, None),
        )]
        self.assertTrue(all(row["Config"] == "Retrieval Unknown" for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            source = export.write_csv(rows, Path(directory) / "old.csv")
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                target = convert_csv_to_excel(source, Path(directory) / "corrected.xlsx")
            self.assertEqual(source.read_bytes(), before)
            book = load_workbook(target)
            try:
                raw = list(book["Runs"].values)
                records = [dict(zip(raw[0], values)) for values in raw[1:]]
                for row in records[:2]:
                    self.assertEqual(row["Retrieval Enable"], "False")
                    self.assertEqual(row["Config"], "Retrieval OFF")
                self.assertEqual(records[2]["Config"], "Retrieval Unknown")
                results = list(book["Results"].values)
                headers = list(results[0])
                baseline = next(row for row in results if row[1] == "Retrieval OFF")
                self.assertEqual(baseline[headers.index(9998)], 23434)
                self.assertEqual(baseline[headers.index(9999)], 17628)
                self.assertEqual(baseline[headers.index(PAIRED_MEAN_COLUMN)], 20531)
            finally:
                book.close()

    def test_duplicate_runs_are_averaged_per_seed_and_nonfinal_runs_are_excluded(self):
        from openpyxl import load_workbook
        rows = []
        for index, (seed, score, state) in enumerate([
                (1, 10, "finished"), (1, 30, "finished"), (2, 40, "finished"),
                (2, 999, "running"), (3, None, "running"), (4, None, "finished"),
                (None, 999, "finished")]):
            rows.append(export.classify_run(make_run(
                id=f"run{index}", name="callbacks/test/atari100k-krull",
                config={"seed": seed, "retrieval_enabled": False},
                metadata={}, summary={export.SCORE_KEY: score}, state=state,
            )))
        with tempfile.TemporaryDirectory() as directory:
            source = export.write_csv(rows, Path(directory) / "scores.csv")
            with source.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 7)
            with contextlib.redirect_stdout(io.StringIO()):
                target = convert_csv_to_excel(source, Path(directory) / "scores.xlsx")
            book = load_workbook(target)
            self.addCleanup(book.close)
            self.assertEqual(book.sheetnames, ["Results", "Runs", "Missing"])
            sheet = book["Results"]
            headers = {cell.value: cell.column for cell in sheet[1]}
            row_index = next(row[0].row for row in sheet.iter_rows(min_row=2)
                             if row[1].value == "Retrieval OFF")
            paper_row = next(row[0].row for row in sheet.iter_rows(min_row=2)
                             if row[1].value == "TWISTER (논문): 8839")
            self.assertEqual(sheet.cell(row_index, headers[1]).value, 20)
            self.assertEqual(sheet.cell(row_index, headers[2]).value, 40)
            self.assertEqual(sheet.cell(row_index, headers[3]).value, "RUNNING")
            self.assertEqual(sheet.cell(row_index, headers[4]).value, "MISSING")
            self.assertEqual(sheet.cell(row_index, headers[PAIRED_MEAN_COLUMN]).value, 30)
            self.assertEqual(sheet.cell(paper_row, headers[SCORE_DELTA_COLUMN]).value, -8809)
            self.assertAlmostEqual(sheet.cell(paper_row, headers[HNS_DELTA_COLUMN]).value,
                                   -8809 / (2666 - 1598))
            self.assertIn("run0", sheet.cell(row_index, headers[1]).comment.text)
            self.assertIn("run1", sheet.cell(row_index, headers[1]).comment.text)
            self.assertEqual(book["Runs"].max_row, 8)
            self.assertEqual(book["Missing"].max_row, 3)

    def test_paired_metrics_exclude_unmatched_seeds_and_keep_signed_hns(self):
        records = [
            {"config": "Retrieval OFF", "retrieval": "false",
             "scores": {"1": 10, "2": 20, "3": 1000}},
            {"config": "Retrieval ON", "retrieval": "true",
             "scores": {"1": 0, "2": 10, "4": 2000}},
        ]
        metrics, paper, note = comparison_metrics(records, (-21, 15, 20))
        self.assertEqual(metrics["Retrieval OFF"], [15, None, None])
        self.assertEqual(metrics["Retrieval ON"][:2], [5, 10])
        self.assertAlmostEqual(metrics["Retrieval ON"][2], -10 / 36)
        self.assertEqual(paper[:2], [None, -5])
        self.assertAlmostEqual(paper[2], -5 / 36)
        self.assertEqual(note, "OFF와 모든 ON 설정의 공통 시드: 1, 2")

    def test_unpaired_or_unknown_runs_cannot_produce_comparison_metrics(self):
        for records in ([
                {"config": "OFF", "retrieval": "false", "scores": {"1": 10}},
                {"config": "ON", "retrieval": "true", "scores": {"2": 20}},
            ], [{"config": "Unknown", "retrieval": "unknown", "scores": {"1": 10}}]):
            metrics, paper, _ = comparison_metrics(records, (0, 12, 88))
            self.assertTrue(all(values == [None, None, None] for values in metrics.values()))
            self.assertEqual(paper, [None, None, None])

    def test_paper_references_use_image_values_and_normalize_game_names(self):
        self.assertEqual(reference_scores("atari100k-pong", "Pong"), (-21, 15, 20))
        self.assertEqual(reference_scores("atari100k-jamesbond", "James Bond"), (29, 303, 573))
        self.assertEqual(reference_scores("atari100k-bankheist", "Bank_Heist"), (14, 753, 942))
        self.assertIsNone(reference_scores("dmc-pong", "pong"))

    def test_empty_export_creates_readable_empty_workbook(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as directory:
            source = export.write_csv([], Path(directory) / "empty.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                target = convert_csv_to_excel(source, Path(directory) / "empty.xlsx")
            book = load_workbook(target)
            try:
                self.assertEqual(book["Results"].max_row, 1)
                self.assertEqual(book["Runs"].max_row, 1)
            finally:
                book.close()


if __name__ == "__main__":
    unittest.main()

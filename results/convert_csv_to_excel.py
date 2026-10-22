#!/usr/bin/env python3
"""Convert exported TWISTER scores to an Excel game/config/seed table."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import mean
import sys
import tempfile

from classify_wandb_runs import HERE, SCORE_KEY, number


# (Random, Human, TWISTER paper): transcribed from the user's table, 2026-10-22.
# Keep the displayed integer values, including Pong's -21 and Boxing's 0.
REFERENCE_SCORES = {
    "alien": (228, 7128, 970),
    "amidar": (6, 1720, 184),
    "assault": (222, 742, 721),
    "asterix": (210, 8503, 1306),
    "bankheist": (14, 753, 942),
    "battlezone": (2360, 37188, 9920),
    "boxing": (0, 12, 88),
    "breakout": (2, 30, 35),
    "choppercommand": (811, 7388, 910),
    "crazyclimber": (10780, 35829, 81880),
    "demonattack": (152, 1971, 289),
    "freeway": (0, 30, 32),
    "frostbite": (65, 4335, 305),
    "gopher": (258, 2412, 22234),
    "hero": (1027, 30826, 8773),
    "jamesbond": (29, 303, 573),
    "kangaroo": (52, 3035, 6016),
    "krull": (1598, 2666, 8839),
    "kungfumaster": (258, 22736, 23442),
    "mspacman": (307, 6952, 2206),
    "pong": (-21, 15, 20),
    "privateeye": (25, 69571, 1608),
    "qbert": (164, 13455, 3197),
    "roadrunner": (12, 7845, 17832),
    "seaquest": (68, 42055, 532),
    "upndown": (533, 11693, 7068),
}
PAIRED_MEAN_COLUMN = "Mean (공통 시드)"
SCORE_DELTA_COLUMN = "Δ Score (행별 비교)"
HNS_DELTA_COLUMN = "Δ HNS (행별 비교)"
METRIC_COLUMNS = [PAIRED_MEAN_COLUMN, SCORE_DELTA_COLUMN, HNS_DELTA_COLUMN]


def reference_scores(environment, game):
    key = re.sub(r"[^a-z0-9]", "", game.lower())
    return REFERENCE_SCORES.get(key) if environment.startswith("atari100k-") else None


def comparison_metrics(records, reference):
    """Compare baseline and ON configurations on the same completed seeds."""
    metrics = {record["config"]: [None, None, None] for record in records}
    paper_metrics = [None, None, None]
    baselines = [record for record in records if record["retrieval"] == "false"]
    enabled = [record for record in records if record["retrieval"] == "true"]
    if len(baselines) != 1:
        return metrics, paper_metrics, "Retrieval OFF baseline이 확인되지 않아 비교하지 않습니다."
    baseline = baselines[0]
    shared_seeds = set(baseline["scores"])
    for record in enabled:
        shared_seeds.intersection_update(record["scores"])
    if not shared_seeds:
        return metrics, paper_metrics, "비교할 완료 run의 공통 시드가 없습니다."
    baseline_mean = mean(baseline["scores"][seed] for seed in sorted(shared_seeds, key=int))
    metrics[baseline["config"]][0] = baseline_mean
    denominator = reference[1] - reference[0] if reference else None
    if reference:
        paper_delta = baseline_mean - reference[2]
        paper_metrics[1] = paper_delta
        if denominator:
            paper_metrics[2] = paper_delta / denominator
    for record in enabled:
        on_mean = mean(record["scores"][seed] for seed in sorted(shared_seeds, key=int))
        delta = on_mean - baseline_mean
        metrics[record["config"]] = [on_mean, abs(delta),
                                      delta / denominator if denominator else None]
    description = ("OFF와 모든 ON 설정의 공통 시드" if enabled else
                   "ON 결과가 없어 완료된 baseline의 유효 시드")
    note = f"{description}: {', '.join(sorted(shared_seeds, key=int))}"
    return metrics, paper_metrics, note


def convert_csv_to_excel(csv_path=None, output_path=None):
    try:
        from openpyxl import Workbook
        from openpyxl.comments import Comment
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as error:
        raise RuntimeError("Excel export requires: python -m pip install openpyxl") from error
    csv_path = Path(csv_path or HERE / "wandb_runs_classification.csv").expanduser()
    output = Path(output_path or HERE / "converted_results.xlsx").expanduser().resolve()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"Environment", "Game", "Seed", "Config", "State", SCORE_KEY}
        if required - set(fields):
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(required - set(fields)))}")
        rows = list(reader)
    if output == csv_path.resolve():
        raise ValueError("Excel output must differ from the input CSV path")

    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("Score Status") == "warmup":
            continue
        key = (row["Environment"], row["Game"], row["Config"])
        groups[key][row["Seed"]].append(row)
    seeds = sorted({seed for group in groups.values() for seed in group if seed.isdigit()}, key=int)

    workbook = Workbook()
    results = workbook.active
    results.title = "Results"
    results.append(["Game", "Config", *[int(seed) for seed in seeds], *METRIC_COLUMNS])
    game_records = defaultdict(list)
    for key, by_seed in sorted(groups.items()):
        cells, seed_scores = [], {}
        comments = []
        for seed in seeds:
            entries = by_seed.get(seed, [])
            scores = [number(row[SCORE_KEY]) for row in entries
                      if row["State"] == "finished" and number(row[SCORE_KEY]) is not None]
            if scores:
                value = mean(scores)
                seed_scores[seed] = value
            else:
                value = ", ".join(sorted({
                    "RUNNING" if row["State"] == "running" else
                    "ERROR" if row.get("Score Status") == "error" else
                    "MISSING" if row["State"] == "finished" else row["State"].upper()
                    for row in entries
                }))
            cells.append(value)
            comments.append("\n".join(
                f"{row.get('Run ID', '')} | {row['State']} | {row[SCORE_KEY]}\n"
                f"{row.get('Run URL', '')}" for row in entries))
        retrieval = {row.get("Retrieval Enable", "").lower()
                     for entries in by_seed.values() for row in entries}
        game_records[key[:2]].append({
            "config": key[2], "cells": cells, "scores": seed_scores, "comments": comments,
            "retrieval": next(iter(retrieval)) if len(retrieval) == 1 else "unknown",
        })
    mean_column = 3 + len(seeds)
    game_blocks, paper_rows = [], []
    for (environment, game), records in sorted(game_records.items()):
        reference = reference_scores(environment, game)
        metrics, paper_metrics, comparison_note = comparison_metrics(records, reference)
        first_row = results.max_row + 1
        label = game
        if reference:
            random_score, human_score, paper_score = reference
            label = f"{game}\nRandom: {random_score}\nHuman: {human_score}"
            results.append([label, f"TWISTER (논문): {paper_score}",
                            *[None for _ in seeds], *paper_metrics])
            paper_rows.append(results.max_row)
            results.cell(results.max_row, 2).comment = Comment(
                "출처: 사용자가 첨부한 Atari100k 표 (2026-10-22)의 TWISTER (ours) 열. "
                "Random/Human도 같은 이미지에 표시된 정수값을 사용합니다.", "TWISTER")
            results.cell(results.max_row, mean_column + 1).comment = Comment(comparison_note, "TWISTER")
        for record in sorted(records, key=lambda item: (
                {"false": 0, "true": 1}.get(item["retrieval"], 2), item["config"])):
            results.append([label, record["config"], *record["cells"], *metrics[record["config"]]])
            for column, comment in enumerate(record["comments"], start=3):
                if comment:
                    results.cell(results.max_row, column).comment = Comment(comment, "TWISTER")
            results.cell(results.max_row, mean_column).comment = Comment(comparison_note, "TWISTER")
        game_blocks.append((first_row, results.max_row, label))
    for offset, description in enumerate((
        "게임별 OFF와 모든 ON 설정에 유효한 최종 점수가 있는 공통 시드를 사용합니다. "
        "ON 결과가 아직 없으면 완료된 baseline의 모든 유효 시드를 사용합니다. "
        "같은 시드의 중복 run은 먼저 평균하며, 실행 중·점수 누락·시드 미확인은 제외합니다.",
        "TWISTER (논문) 행: OFF 평균 − TWISTER 논문 점수 (부호 유지). "
        "ON 행: |ON 평균 − OFF 평균|. OFF 행은 빈칸입니다.",
        "TWISTER (논문) 행: (OFF 평균 − TWISTER 논문 점수) / (Human − Random). "
        "ON 행: (ON 평균 − OFF 평균) / (Human − Random). "
        "부호를 유지하며 백분율이 아닙니다. Random/Human은 첨부 이미지의 정수값입니다.",
    )):
        results.cell(1, mean_column + offset).comment = Comment(description, "TWISTER")

    def add_raw_sheet(title, entries):
        sheet = workbook.create_sheet(title)
        sheet.append(fields)
        for entry in entries:
            sheet.append([entry.get(field, "") for field in fields])
        score_columns = {index for index, field in enumerate(fields, start=1)
                         if field in (SCORE_KEY, "Eval Return")}
        for cells in sheet.iter_rows(min_row=2):
            for cell in cells:
                if cell.column in score_columns and number(cell.value) is not None:
                    cell.value = number(cell.value)
                if fields[cell.column - 1] == "Run URL" and str(cell.value).startswith("https://"):
                    cell.hyperlink = cell.value
        return sheet

    add_raw_sheet("Runs", rows)
    add_raw_sheet("Missing", [row for row in rows if number(row[SCORE_KEY]) is None])
    for sheet in workbook:
        sheet.freeze_panes = "C2" if sheet.title == "Results" else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24465C")
            cell.alignment = Alignment(horizontal="left", vertical="center")
        for cells in sheet.iter_rows(min_row=2):
            for cell in cells:
                # CSV labels and run names should remain literal Excel text.
                if isinstance(cell.value, str):
                    cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if isinstance(cell.value, (float, int)):
                    cell.number_format = "0.00"
                if cell.value == "RUNNING":
                    cell.fill = PatternFill("solid", fgColor="FFF2CC")
        for index, cells in enumerate(sheet.columns, start=1):
            longest = max((len(str(cell.value or "")) for cell in cells), default=0)
            sheet.column_dimensions[get_column_letter(index)].width = min(65, max(12, longest + 2))
    for offset, width in enumerate((30, 34, 34)):
        column = mean_column + offset
        results.column_dimensions[get_column_letter(column)].width = width
        for cells in results.iter_rows(min_row=2, min_col=column, max_col=column):
            cells[0].number_format = "+0.0000;-0.0000;0.0000" if offset == 2 else "0.00"
    for row in paper_rows:
        results.cell(row, mean_column + 1).number_format = "+0.00;-0.00;0.00"
        for cell in list(results[row])[1:]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="E9F0F7")
    for first_row, last_row, label in game_blocks:
        if last_row > first_row:
            results.merge_cells(start_row=first_row, start_column=1, end_row=last_row, end_column=1)
        game_cell = results.cell(first_row, 1)
        game_cell.value = label
        game_cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        for row in range(first_row, last_row + 1):
            results.row_dimensions[row].height = max(24, 72 / (last_row - first_row + 1))
        for cell in results[first_row]:
            cell.border = Border(top=Side(style="medium", color="24465C"))
    results.column_dimensions["A"].width = max(24, results.column_dimensions["A"].width)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        workbook.save(temporary)
        temporary.replace(output)
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)
    print(f"Excel: {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "wandb_runs_classification.csv")
    parser.add_argument("--output", type=Path, default=HERE / "converted_results.xlsx")
    args = parser.parse_args(argv)
    try:
        convert_csv_to_excel(args.input, args.output)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Excel conversion failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

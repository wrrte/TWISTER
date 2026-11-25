#!/usr/bin/env python3
"""Export update_tex.py's selected TWISTER scores to the matcher's Excel layout."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
import tempfile

from update_tex import (ComparisonScores, METRICS, format_value, game_key, load_results,
                        main_table_rows, metric_name, update_table)


HERE = Path(__file__).resolve().parent
MATCHER = HERE.parent.parent / "episode_score_matcher"


def load_external_scores(path):
    """Read confirmed external pairs without inventing training seed IDs.

    Each game replaces its CSV-selected scores to avoid counting overlapping
    external/local observations twice. Array order preserves the supplied pairs.
    """
    results, sources = {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("External scores must be an object keyed by game")
    for game, entry in data.items():
        key = game_key(game)
        if key in results or not isinstance(entry, dict):
            raise ValueError(f"{game}: invalid or duplicate external score entry")
        baseline, enabled = entry.get("baseline"), entry.get("flash")
        source = entry.get("source")
        if (not isinstance(baseline, list) or not isinstance(enabled, list)
                or not 1 <= len(baseline) == len(enabled) <= 5):
            raise ValueError(f"{game}: external baseline/flash must contain 1-5 aligned pairs")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{game}: external scores need a source description")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in baseline + enabled):
            raise ValueError(f"{game}: external scores must be finite numbers")
        unknown_seeds = (None,) * len(baseline)
        results[key] = ComparisonScores(unknown_seeds, unknown_seeds, tuple(baseline), tuple(enabled))
        sources[key] = f"{source}\nExternal score file: {path}"
    return results, sources


def workbook_layout(sheet):
    """Find the five display slots, game/method rows and aggregate cells."""
    required = {"game", "method", "1", "2", "3", "4", "5", "mean"}
    for row in sheet.iter_rows(max_row=min(15, sheet.max_row)):
        columns = {str(cell.value).strip().lower(): cell.column
                   for cell in row if cell.value is not None}
        if required.issubset(columns):
            header_row = row[0].row
            break
    else:
        raise ValueError("TWISTER: expected Game, Method, 1, 2, 3, 4, 5, Mean columns")

    rows, current_game = {}, None
    for row in sheet.iter_rows(min_row=header_row + 1):
        game = row[columns["game"] - 1].value
        if str(game).strip().lower() == "metric":
            method_columns = {str(cell.value).strip().lower(): cell.column for cell in row
                              if str(cell.value).strip().lower() in ("baseline", "flash")}
            summary_row = row[0].row
            break
        if game is not None:
            current_game = game_key(str(game))
        method = str(row[columns["method"] - 1].value or "").strip().lower()
        if current_game and method in ("baseline", "flash"):
            key = current_game, method
            if key in rows:
                raise ValueError(f"TWISTER: duplicate game/method row: {key}")
            rows[key] = row[0].row
    else:
        raise ValueError("TWISTER: missing Metric / baseline / flash aggregate header")
    if set(method_columns) != {"baseline", "flash"}:
        raise ValueError("TWISTER: missing baseline/flash aggregate columns")
    metrics = {}
    for row in sheet.iter_rows(min_row=summary_row + 1):
        name = metric_name(str(row[0].value or ""))
        if name:
            if name in metrics:
                raise ValueError(f"TWISTER: duplicate aggregate row: {name}")
            metrics[name] = row[0].row
    if set(metrics) != set(METRICS):
        raise ValueError("TWISTER: expected all five aggregate rows")
    return columns, rows, method_columns, metrics


def export_workbook(template, output, results, tex_lines, csv_path, *, all_seed=False, external_sources=None):
    from openpyxl import load_workbook
    from openpyxl.comments import Comment

    # Reuse the TeX updater's means and HNS calculations, including external pairs.
    updated_rows = main_table_rows(update_table(tex_lines, results))
    summaries = {metric_name(parts[0]): parts[-3:-1] for _, parts, _ in updated_rows
                 if metric_name(parts[0])}
    book = load_workbook(template)
    try:
        if "TWISTER" not in book.sheetnames:
            raise ValueError("Template must contain a TWISTER sheet")
        sheet = book["TWISTER"]
        columns, rows, method_columns, metrics = workbook_layout(sheet)
        for game, scores in results.items():
            for method, values in (("baseline", scores.baseline), ("flash", scores.enabled)):
                if (game, method) not in rows:
                    raise ValueError(f"TWISTER: template is missing {game}/{method}")
                if len(values) > 5:
                    raise ValueError(f"{game}/{method}: {len(values)} scores exceed the five Excel slots")

        filled = 0
        for (game, method), rownum in rows.items():
            # Remove old template scores, including games no longer in the comparison.
            for column in [columns[str(slot)] for slot in range(1, 6)] + [columns["mean"]]:
                cell = sheet.cell(rownum, column)
                cell.value = None
                cell.comment = None
            scores = results.get(game)
            if scores is None:
                continue
            seeds, values = ((scores.baseline_seeds, scores.baseline) if method == "baseline"
                             else (scores.enabled_seeds, scores.enabled))
            for slot, (seed, value) in enumerate(zip(seeds, values), 1):
                cell = sheet.cell(rownum, columns[str(slot)])
                cell.value = value  # Retain unrounded scores, including zero and negatives.
                cell.number_format = "#,##0.###############"
                provenance = ((external_sources or {}).get(game) or
                              f"Selection: update_tex.py{' --all-seed' if all_seed else ''}\n"
                              f"Source CSV (with run overrides and baseline pairings): {csv_path}")
                cell.comment = Comment(
                    f"Actual training seed: {seed if seed is not None else 'not provided'}\n"
                    f"Display slot: {slot}; method: {method}\n"
                    f"{provenance}", "Source")
                filled += 1
            if values:
                cell = sheet.cell(rownum, columns["mean"])
                cell.value = float(format_value(mean(values)))
                cell.number_format = "#,##0.#"

        for name, rownum in metrics.items():
            for method, text in zip(("baseline", "flash"), summaries[name]):
                cell = sheet.cell(rownum, method_columns[method])
                text = text.strip()
                cell.value = None if text == "-" else (int(text) if name == "#Superhuman" else float(text))
                cell.number_format = "0" if name == "#Superhuman" else "0.000"
                cell.comment = None

        # Preserve the template's other sheets, merged cells and formatting.
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".xlsx", delete=False) as stream:
                temporary = Path(stream.name)
            book.save(temporary)
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return filled
    finally:
        book.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "wandb_runs_classification.csv")
    parser.add_argument("--tex", type=Path, default=HERE.parent.parent / "iclr2027_conference.tex",
                        help="Read Random/Human references for the same aggregate metrics as update_tex.py")
    parser.add_argument("--template", type=Path, default=MATCHER / "input_scores_matched.xlsx")
    parser.add_argument("--output", type=Path, default=MATCHER / "input_scores_twister.xlsx")
    parser.add_argument("--external-scores", type=Path, default=HERE / "external_scores.json",
                        help="Confirmed external pairs; each listed game replaces its CSV selection ({} disables)")
    parser.add_argument("--run-overrides", type=Path, help="Same corrections as update_tex.py")
    parser.add_argument("--on-config", help="Exact ON config label, as in update_tex.py")
    parser.add_argument("--all-seed", action="store_true",
                        help="Use update_tex.py's balanced seed selection instead of matched seed pairs")
    parser.add_argument("--baseline-pairings", type=Path, help="Explicit cross-seed pair directory or JSON file")
    args = parser.parse_args(argv)
    for name in ("csv", "tex", "template", "output", "run_overrides", "baseline_pairings", "external_scores"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.output.suffix.lower() != ".xlsx":
        raise ValueError("--output must be an .xlsx file")
    if args.output in (args.csv, args.tex, args.template, args.run_overrides, args.baseline_pairings, args.external_scores):
        raise ValueError("Output must differ from every input file")
    results = load_results(args.csv, args.run_overrides, args.on_config, all_seed=args.all_seed,
                           baseline_pairings_path=args.baseline_pairings)
    external, external_sources = load_external_scores(args.external_scores)
    results.update(external)
    filled = export_workbook(args.template, args.output, results,
                             args.tex.read_text(encoding="utf-8").splitlines(keepends=True),
                             args.csv, all_seed=args.all_seed, external_sources=external_sources)
    print(f"Saved: {args.output}")
    print(f"TWISTER: {len(results)} games, {filled} scores; slots 1-5 follow selected/supplied pair order.")
    if external:
        print(f"Confirmed external pairs: {', '.join(sorted(external))} (replace CSV selections for these games).")
    print("Unused slots are blank. Seed IDs, when provided, are in cell comments. STORM/DRAMA retain the template's data.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"TWISTER seed export failed: {error}", file=sys.stderr)
        raise SystemExit(1)

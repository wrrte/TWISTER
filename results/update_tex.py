#!/usr/bin/env python3
"""Update only the TWISTER cells in tab:main_performance from CSV scores."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path
import re
from statistics import mean, median
import sys
import tempfile

from classify_wandb_runs import SCORE_KEY, apply_run_overrides, load_run_overrides, number
from convert_csv_to_excel import comparison_metrics, comparison_seeds, seed_label, seed_sort_key
from baseline_pairs import load_pairings, prepare_records


HERE = Path(__file__).resolve().parent
METRICS = ("#Superhuman", "Mean", "Median", "IQM", "Optimality Gap")


def game_key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class ComparisonScores:
    baseline_seeds: tuple
    enabled_seeds: tuple
    baseline: tuple
    enabled: tuple


def load_results(csv_path, run_overrides_path=None, on_config=None, all_seed=False, baseline_pairings_path=None):
    """Use Excel's duplicate means and shared/balanced seed selection."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Environment", "Game", "Seed", "Config", "State", "Retrieval Enable", SCORE_KEY}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    apply_run_overrides(rows, load_run_overrides(run_overrides_path))
    pairings = load_pairings(path=baseline_pairings_path)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("Score Status") == "warmup" or not row["Environment"].startswith("atari100k-"):
            continue
        groups[row["Environment"], row["Game"], row["Config"]][seed_label(row)].append(row)

    games = defaultdict(list)
    for (environment, game, config), by_seed in sorted(groups.items()):
        scores = {}
        for seed, entries in by_seed.items():
            if seed_sort_key(seed) is None:
                continue
            values = [number(row[SCORE_KEY]) for row in entries
                      if row["State"] == "finished" and number(row[SCORE_KEY]) is not None]
            if values:
                scores[seed] = mean(values)
        retrieval = {row.get("Retrieval Enable", "").lower()
                     for entries in by_seed.values() for row in entries}
        games[environment, game].append(dict(
            config=config, scores=scores,
            retrieval=next(iter(retrieval)) if len(retrieval) == 1 else "unknown"))

    on_configs = {record["config"] for records in games.values()
                  for record in records if record["retrieval"] == "true"}
    if on_config is None and len(on_configs) > 1:
        raise ValueError("Multiple Retrieval ON configs; select one with --on-config:\n" +
                         "\n".join(sorted(on_configs)))
    if on_config is not None and on_config not in on_configs:
        raise ValueError("--on-config does not match a Retrieval ON config in the CSV")
    selected = on_config if on_config is not None else next(iter(on_configs), None)
    results = {}
    for (environment, game), records in sorted(games.items()):
        records = prepare_records(records, pairings.get(environment.removeprefix("atari100k-"), {}))
        baselines = [record for record in records if record["retrieval"] == "false"]
        enabled = [record for record in records if record["retrieval"] == "true"]
        ours = next((record for record in enabled if record["config"] == selected), None)
        if not baselines or ours is None:
            continue  # Excel's baseline-only mean is intentionally NOT a pair.
        if len(baselines) != 1:
            raise ValueError(f"{game}: expected one Retrieval OFF configuration")
        baseline = baselines[0]
        seeds, _ = comparison_seeds(records, all_seed)
        if not seeds:
            continue
        baseline_seeds, enabled_seeds = seeds[baseline["config"]], seeds[ours["config"]]
        scores = ComparisonScores(
            baseline_seeds, enabled_seeds,
            tuple(baseline["scores"][seed] for seed in baseline_seeds),
            tuple(ours["scores"][seed] for seed in enabled_seeds))
        # Keep both modes tied to Excel's comparison means.
        excel_means, _, _ = comparison_metrics(records, reference=None, all_seed=all_seed)
        if (mean(scores.baseline) != excel_means[baseline["config"]][0]
                or mean(scores.enabled) != excel_means[ours["config"]][0]):
            raise ValueError(f"{game}: comparison means disagree with Excel")
        key = game_key(game)
        if key != game_key(environment.split("-", 1)[1]) or key in results:
            raise ValueError(f"Ambiguous game/environment identity: {environment}, {game}")
        results[key] = scores
    return results


def metric_name(label):
    label = label.strip().lstrip("\\")
    return next((name for name in METRICS if label == name or label.startswith(name + " (")), None)


def main_table(lines, allow_missing_twister=False):
    """Locate and validate the main table, optionally before adding TWISTER."""
    labels = [index for index, line in enumerate(lines)
              if not line.lstrip().startswith("%") and r"\label{tab:main_performance}" in line]
    if len(labels) != 1:
        raise ValueError("Expected exactly one active tab:main_performance label")
    header = None
    rows = []
    names = set()
    for index in range(labels[0] + 1, len(lines)):
        line = lines[index]
        if line.lstrip().startswith("%"):
            continue
        if r"\end{table}" in line or r"\end{tabular}" in line or r"\bottomrule" in line:
            break
        if "&" not in line:
            continue
        body, separator, tail = line.partition(r"\\")
        parts = body.split("&")
        if parts[0].strip() == "Game":
            if header is not None or not separator:
                raise ValueError("Malformed or duplicate main performance header")
            header = [part.strip() for part in parts]
            header_index = index
            if header[:3] != ["Game", "Random", "Human"]:
                raise ValueError("Expected Game / Random / Human as the first three columns")
            has_twister = any("TWISTER" in column for column in header)
            if has_twister:
                if (header.count("TWISTER") != 1 or header.count("TWISTER+ours") != 1
                        or header[-3:] != ["TWISTER", "TWISTER+ours", r"$\Delta$"]):
                    raise ValueError("Expected a complete TWISTER / TWISTER+ours / $\\Delta$ column group at the end")
            elif not allow_missing_twister:
                raise ValueError("Expected a TWISTER / TWISTER+ours / $\\Delta$ column group")
        elif header is not None:
            if not separator or len(parts) != len(header):
                raise ValueError(f"Malformed main performance row at line {index + 1}")
            name = metric_name(parts[0]) or game_key(parts[0].strip())
            if name in names:
                raise ValueError(f"Duplicate main performance row: {name}")
            names.add(name)
            rows.append((index, parts, separator + tail))
    if header is None or not set(METRICS).issubset(names):
        raise ValueError("Main performance table must contain a header and all five metric rows")
    return labels[0], header_index, header, rows


def main_table_rows(lines):
    return main_table(lines)[3]


def ensure_twister_columns(lines):
    """Append the TWISTER group and three tabular columns when absent."""
    label_index, header_index, header, rows = main_table(lines, allow_missing_twister=True)
    updated = list(lines)
    if "TWISTER" in header:
        return updated
    tabulars = []
    for index in range(label_index + 1, header_index):
        if lines[index].lstrip().startswith("%"):
            continue
        match = re.search(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{([lcr| \t]+)\}", lines[index])
        if match:
            tabulars.append((index, match))
    if len(tabulars) != 1:
        raise ValueError("Adding TWISTER columns requires one tabular with an l/c/r column specification")
    index, match = tabulars[0]
    if len(re.findall(r"[lcr]", match[1])) != len(header):
        raise ValueError("Main performance tabular column count does not match its header")
    updated[index] = lines[index][:match.end(1)] + "rrr" + lines[index][match.end(1):]
    append_rows = [(header_index, ("TWISTER", "TWISTER+ours", r"$\Delta$"))]
    append_rows.extend((index, ("-", "-", "-")) for index, _, _ in rows)
    for index, values in append_rows:
        body, separator, tail = lines[index].partition(r"\\")
        updated[index] = body + "& " + " & ".join(values) + " " + separator + tail
    return updated


def calc_iqm(values):
    """Pool per-(game, seed) HNS and trim floor(N/4) samples from each tail."""
    values = sorted(values)
    trim = len(values) // 4
    return mean(values[trim:len(values) - trim]) if values else None


def aggregate_metrics(game_hns, seed_hns):
    """Game-balanced HNS summaries; IQM pools unrounded retained seed scores."""
    if not game_hns:
        return dict.fromkeys(METRICS)
    return {
        "#Superhuman": sum(value > 1.0 for value in game_hns),
        "Mean": mean(game_hns),
        "Median": median(game_hns),
        "IQM": calc_iqm(seed_hns),
        "Optimality Gap": mean(max(0.0, 1.0 - value) for value in game_hns),
    }


def format_value(value, metric=None):
    if value is None:
        return "-"
    if metric == "#Superhuman":
        return str(int(value))
    if metric:
        text = f"{value:.3f}"
        return "0.000" if text == "-0.000" else text
    text = f"{value:.1f}"
    if text == "-0.0":
        return "0"
    return text[:-2] if text.endswith(".0") else text


def format_delta(baseline, enabled, metric=None):
    if "-" in (baseline, enabled):
        return "-"
    delta = Decimal(enabled) - Decimal(baseline)
    text = f"{delta:.3f}" if metric and metric != "#Superhuman" else format(delta, "f")
    if not metric or metric == "#Superhuman":
        text = text.rstrip("0").rstrip(".") if "." in text else text
    if not delta:
        return text.lstrip("-")
    if delta > 0:
        text = "+" + text
    improves = delta < 0 if metric == "Optimality Gap" else delta > 0
    return r"\textcolor{" + ("green" if improves else "red") + "}{" + text + "}"


def update_table(lines, results):
    """Add or update TWISTER's columns while preserving the other cells."""
    lines = ensure_twister_columns(lines)
    rows = main_table_rows(lines)
    game_hns = [[], []]
    seed_hns = [[], []]
    table_games = {}
    for _, parts, _ in rows:
        if metric_name(parts[0]):
            continue
        key = game_key(parts[0].strip())
        random, human = (float(part.strip().replace(",", "")) for part in parts[1:3])
        if not all(math.isfinite(value) for value in (random, human)) or human <= random:
            raise ValueError(f"{parts[0].strip()}: invalid Random/Human references")
        table_games[key] = parts[0].strip()
        if key in results:
            for column, scores in enumerate((results[key].baseline, results[key].enabled)):
                game_hns[column].append((mean(scores) - random) / (human - random))
                seed_hns[column].extend((score - random) / (human - random) for score in scores)
    if set(results) - set(table_games):
        raise ValueError("CSV comparison games absent from main table: " +
                         ", ".join(sorted(set(results) - set(table_games))))
    metrics = [aggregate_metrics(games, seeds) for games, seeds in zip(game_hns, seed_hns)]
    updated = list(lines)
    for index, parts, ending in rows:
        metric = metric_name(parts[0])
        key = game_key(parts[0].strip())
        values = ([column[metric] for column in metrics] if metric else
                  [mean(results[key].baseline), mean(results[key].enabled)] if key in results else [None, None])
        baseline, enabled = (format_value(value, metric) for value in values)
        parts[-3:] = [" " + value + " " for value in
                      (baseline, enabled, format_delta(baseline, enabled, metric))]
        updated[index] = "&".join(parts) + ending
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "wandb_runs_classification.csv")
    parser.add_argument("--tex", type=Path, default=HERE.parent.parent / "iclr2027_conference.tex")
    parser.add_argument("--run-overrides", type=Path, help="Use the same confirmed corrections as Excel")
    parser.add_argument("--on-config", help="Exact ON config label; needed only when several ON configs exist")
    parser.add_argument("--all-seed", action="store_true",
                        help="Ignore seed IDs; equalize OFF/all ON counts by removing lowest/highest scores alternately")
    parser.add_argument("--baseline-pairings", type=Path, help="Explicit cross-seed pair directory or JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Report seed coverage and metrics without writing TeX")
    args = parser.parse_args(argv)
    args.tex = args.tex.expanduser().resolve()
    args.csv = args.csv.expanduser().resolve()
    if args.csv.resolve() == args.tex.resolve():
        raise ValueError("TeX output must differ from the input CSV")
    print(f"TeX: {args.tex}")
    results = load_results(args.csv, args.run_overrides, args.on_config, all_seed=args.all_seed,
                           baseline_pairings_path=args.baseline_pairings)
    original = args.tex.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    updated = update_table(lines, results)
    if args.all_seed:
        print("--all-seed: equalize OFF/all ON seed counts per game; ignore seed IDs. "
              "Remove lowest/highest/next-lowest/next-highest scores alternately.")
    for key, scores in sorted(results.items()):
        seed_note = (f"OFF seeds {', '.join(scores.baseline_seeds)}; "
                     f"ON seeds {', '.join(scores.enabled_seeds)}" if args.all_seed or scores.baseline_seeds != scores.enabled_seeds else
                     f"seeds {', '.join(scores.baseline_seeds)}")
        print(f"{key}: {seed_note}; OFF={mean(scores.baseline):.6g}, ON={mean(scores.enabled):.6g}")
    rows = main_table_rows(updated)
    games = sum(metric_name(parts[0]) is None for _, parts, _ in rows)
    coverage = "Balanced coverage" if args.all_seed else "Paired coverage"
    unit = "retained seeds per OFF/ON" if args.all_seed else "game/seed pairs"
    print(f"{coverage}: {len(results)}/{games} games, "
          f"{sum(len(scores.baseline_seeds) for scores in results.values())} {unit}. "
          "Missing games are '-' and excluded from aggregates.")
    for _, parts, _ in rows:
        if metric_name(parts[0]):
            print(f"{metric_name(parts[0])}: " + " | ".join(part.strip() for part in parts[-3:]))
    if args.dry_run:
        print("Dry run: TeX unchanged.")
    else:
        if args.tex.read_bytes() != original:
            raise RuntimeError("TeX changed while calculating; rerun to preserve the concurrent edit")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.tex.parent, prefix=args.tex.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write("".join(updated).encode("utf-8"))
            temporary.chmod(args.tex.stat().st_mode)
            if args.tex.read_bytes() != original:
                raise RuntimeError("TeX changed while calculating; rerun to preserve the concurrent edit")
            temporary.replace(args.tex)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Updated the TWISTER columns in {args.tex}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"TWISTER table update failed: {error}", file=sys.stderr)
        raise SystemExit(1)

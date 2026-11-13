"""Editable legacy baseline pool and explicit cross-seed pair provenance."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import mean
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SCORE_KEY = "Evaluation-final/score"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def finite_score(value):
    if value is None or isinstance(value, bool):
        raise ValueError("A baseline/pair score must be finite")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A baseline/pair score must be finite") from error
    if not math.isfinite(score):
        raise ValueError("A baseline/pair score must be finite")
    return score


def valid_game(game):
    if not isinstance(game, str) or not re.fullmatch(r"[a-z]+(?:_[a-z]+)*", game):
        raise ValueError("Invalid baseline pool game")
    return game


def valid_seed(seed, legacy=False):
    if type(seed) is not int or not 0 <= seed < 2 ** 32 or (legacy and not 9990 <= seed <= 9999):
        raise ValueError("Reusable baseline seeds must be 999N" if legacy else "Invalid pair seed")
    return seed


def pool_path(root=ROOT):
    return Path(root) / "results/reusable_baselines.json"


def pairings_path(root=ROOT):
    return Path(root) / "reproduction/baseline_pairings"


def pairing_path(game, seed, root=ROOT):
    return pairings_path(root) / valid_game(game) / f"seed{valid_seed(seed)}.json"


def load_pool(root=ROOT, path=None):
    path = Path(path) if path is not None else pool_path(root)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("games"), dict):
        raise ValueError("Unsupported reusable baseline pool")
    result = {}
    for game, entry in data["games"].items():
        valid_game(game)
        if game == "gopher":
            continue
        baselines = entry.get("baselines", [])
        if not isinstance(baselines, list):
            raise ValueError(f"{game}: baselines must be a list")
        seen = set()
        result[game] = []
        for baseline in baselines:
            seed = valid_seed(baseline["seed"], legacy=True)
            if seed in seen:
                raise ValueError(f"{game}: duplicate reusable baseline seed {seed}")
            seen.add(seed)
            score = finite_score(baseline["score"])
            if baseline.get("enabled", True):
                result[game].append(dict(baseline, score=score))
    return result


def validate_pairing(record):
    if record.get("version") != 1 or record.get("kind") != "reused_baseline_cross_seed":
        raise ValueError("Unsupported baseline pairing")
    valid_game(record["game"])
    if record["game"] == "gopher":
        raise ValueError("Gopher is excluded from baseline reuse")
    valid_seed(record["full_flash_seed"])
    finite_score(record["full_flash_score"])
    valid_seed(record["baseline"]["seed"], legacy=True)
    finite_score(record["baseline"]["score"])
    if not record.get("gate", {}).get("accepted"):
        raise ValueError("A reused baseline requires an accepted full FLASH gate")
    return dict(record, full_flash_score=finite_score(record["full_flash_score"]),
                baseline=dict(record["baseline"], score=finite_score(record["baseline"]["score"])))


def load_pairings(root=ROOT, path=None):
    directory = Path(path) if path is not None else pairings_path(root)
    if not directory.exists():
        return {}
    files = [directory] if directory.is_file() else sorted(directory.glob("*/seed*.json"))
    result, used = {}, set()
    for path in files:
        record = validate_pairing(json.loads(path.read_text(encoding="utf-8")))
        game, target = record["game"], str(record["full_flash_seed"])
        source = (game, record["baseline"]["seed"])
        if target in result.get(game, {}) or source in used:
            raise ValueError(f"{game}: baseline and full FLASH seeds must be paired one-to-one")
        result.setdefault(game, {})[target] = record
        used.add(source)
    return result


@contextmanager
def allocation_lock(root=ROOT):
    path = Path(root) / "results/reusable_baselines.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare_records(records, pairings):
    """Keep actual OFF seed identities; add frozen source scores, never fake runs."""
    if not pairings:
        return records
    records = [dict(record, scores=dict(record["scores"])) for record in records]
    baselines = [record for record in records if record["retrieval"] == "false"]
    if len(baselines) > 1:
        raise ValueError("Cross-seed pairing requires one Retrieval OFF configuration")
    if not baselines:
        baseline = dict(config="Retrieval OFF", retrieval="false", scores={})
        records.insert(0, baseline)
    else:
        baseline = baselines[0]
    baseline["pairings"] = {}
    for target, record in pairings.items():
        observed = [item["scores"][target] for item in records
                    if item["retrieval"] == "true" and target in item["scores"]]
        if observed and not any(math.isclose(value, record["full_flash_score"], rel_tol=1e-9, abs_tol=1e-9)
                                for value in observed):
            raise ValueError(f"{record['game']} seed {target}: full FLASH score differs from its saved pairing")
        baseline["scores"][str(record["baseline"]["seed"])] = record["baseline"]["score"]
        baseline["pairings"][target] = str(record["baseline"]["seed"])
    return records


def workbook_rows(path):
    """Read the Runs sheet with the standard library; preserve original seed IDs."""
    with ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(node.itertext()) for node in
                       ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        sheets = ET.fromstring(book.read("xl/workbook.xml"))
        sheet = next((node for node in sheets.findall("s:sheets/s:sheet", NS)
                      if node.get("name") == "Runs"), None)
        if sheet is None:
            raise ValueError("The workbook must contain its Runs sheet")
        relation = sheet.get("{" + REL_NS + "}id")
        links = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target = next(node.get("Target") for node in links if node.get("Id") == relation)
        member = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)

        def cells(row):
            result = {}
            for cell in row.findall("s:c", NS):
                column = re.sub(r"\d", "", cell.get("r"))
                value, inline = cell.find("s:v", NS), cell.find("s:is", NS)
                value = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
                result[column] = strings[int(value)] if cell.get("t") == "s" else value
            return result

        rows = ET.fromstring(book.read(member)).findall("s:sheetData/s:row", NS)
        headers = cells(rows[0])
        required = {"Environment", "Seed", "Config", "State", "Retrieval Enable", SCORE_KEY}
        if required - set(headers.values()):
            raise ValueError("Workbook Runs sheet is missing score/identity columns")
        return [{headers[column]: value for column, value in cells(row).items() if column in headers}
                for row in rows[1:]]


def build_pool(workbook):
    rows = workbook_rows(workbook)
    grouped = {}
    for row in rows:
        environment = row["Environment"]
        if not environment.startswith("atari100k-") or environment == "atari100k-gopher":
            continue
        game = valid_game(environment[len("atari100k-"):])
        grouped.setdefault(game, {"false": {}, "true": {}})
        retrieval = row["Retrieval Enable"].lower()
        if (retrieval not in {"false", "true"} or row["State"] != "finished"
                or row.get("Score Status") == "warmup" or not row["Seed"].isdigit()):
            continue
        try:
            score = finite_score(row[SCORE_KEY])
        except (KeyError, ValueError):
            continue
        grouped[game][retrieval].setdefault(row["Config"], {}).setdefault(int(row["Seed"]), []).append((score, row))
    games = {}
    for game, configurations in sorted(grouped.items()):
        if any(len(configurations[mode]) > 1 for mode in ("false", "true")):
            raise ValueError(f"{game}: select one OFF/full FLASH configuration before building the pool")
        off = next(iter(configurations["false"].values()), {})
        on = next(iter(configurations["true"].values()), {})
        paired = sorted(off.keys() & on.keys())
        threshold = mean(mean(score for score, _ in off[seed]) for seed in paired) if paired else None
        baselines = []
        for seed, entries in off.items():
            score = mean(score for score, _ in entries)
            if not 9990 <= seed <= 9999 or seed in paired or threshold is None or score < threshold:
                continue
            baselines.append(dict(seed=seed, score=score, enabled=True,
                                  run_ids=sorted({row.get("Run ID", "") for _, row in entries} - {""}),
                                  run_names=sorted({row.get("Run Name", "") for _, row in entries} - {""}),
                                  run_urls=sorted({row.get("Run URL", "") for _, row in entries} - {""})))
        games[game] = dict(paired_baseline_mean=threshold, paired_seeds=paired,
                           baselines=sorted(baselines, key=lambda entry: (-entry["score"], entry["seed"])))
    return dict(version=1, source=dict(workbook="results/performance_twister.xlsx",
                                      sha256=hashlib.sha256(Path(workbook).read_bytes()).hexdigest()),
                selection="Unpaired finished 999N OFF scores >= mean OFF score of existing same-seed ON/OFF pairs",
                excluded_games=["gopher"], games=games)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=ROOT / "results/performance_twister.xlsx")
    parser.add_argument("--output", type=Path, default=pool_path())
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing edited pool explicitly")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise ValueError("Pool already exists; use --overwrite only to replace your edits")
    pool = build_pool(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(pool, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for game, entry in pool["games"].items():
        if entry["baselines"]:
            print(f"{game}: paired OFF mean={entry['paired_baseline_mean']:.6g}; "
                  + ", ".join(f"seed{item['seed']}={item['score']:g}" for item in entry["baselines"]))
    print(f"Saved {sum(len(entry['baselines']) for entry in pool['games'].values())} reusable baselines: {args.output}")


if __name__ == "__main__":
    main()

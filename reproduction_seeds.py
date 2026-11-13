"""Read occupied experiment seeds without an Excel/PyTorch dependency."""

from functools import lru_cache
import json
from pathlib import PurePosixPath
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def occupied_seeds(root):
    # Excel exports are Git-ignored. Ship their seed reservations separately so
    # every server protects the known runs even without its own workbook.
    manifest = json.loads((root / "reproduction/seed_reservations.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported seed reservation manifest")
    result = {game: set(seeds) for game, seeds in manifest["games"].items()}
    workbook = root / "results/performance_twister.xlsx"
    if workbook.is_file():
        for game, seeds in workbook_seeds(workbook).items():
            result.setdefault(game, set()).update(seeds)
    return result


def workbook_seeds(path):
    stamp = path.stat()
    return _workbook_seeds(str(path), stamp.st_mtime_ns, stamp.st_size)


@lru_cache(maxsize=4)
def _workbook_seeds(path, modified, size):
    # Include warmup, running, failed and legacy rows as reservations. An old
    # baseline alone must also prevent a new ON/OFF experiment reusing its seed.
    with ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(node.itertext()) for node in
                       ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        sheets = ET.fromstring(book.read("xl/workbook.xml"))
        sheet = next((node for node in sheets.findall("s:sheets/s:sheet", NS)
                      if node.get("name") == "Runs"), None)
        if sheet is None:
            raise ValueError("Seed protection requires the workbook's Runs sheet")
        relation = sheet.get("{" + REL_NS + "}id")
        links = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target = next(node.get("Target") for node in links if node.get("Id") == relation)
        member = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        rows = ET.fromstring(book.read(member)).findall("s:sheetData/s:row", NS)

        def values(row):
            result = {}
            for cell in row.findall("s:c", NS):
                column = re.sub(r"\d", "", cell.get("r"))
                value = cell.find("s:v", NS)
                inline = cell.find("s:is", NS)
                value = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
                result[column] = strings[int(value)] if cell.get("t") == "s" else value
            return result

        headers = values(rows[0])
        if not {"Environment", "Seed"}.issubset(headers.values()):
            raise ValueError("Seed protection requires Environment and Seed columns")
        occupied = {}
        for raw in rows[1:]:
            row = {headers.get(column): value for column, value in values(raw).items()}
            environment = row.get("Environment", "")
            seed = re.fullmatch(r"(\d+)(?:-old)?", row.get("Seed", ""))
            if environment.startswith("atari100k-") and seed:
                occupied.setdefault(environment.removeprefix("atari100k-"), set()).add(int(seed[1]))
        return {game: frozenset(seeds) for game, seeds in occupied.items()}

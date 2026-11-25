"""Find the closest possible mean of N scores sampled with replacement.

Searches the full attainable support using exact integer bitset dynamic
programming. This finds a target-selected resample, not a confidence interval,
p-value, or an estimate of how likely that resample is. Equal-distance ties
prefer the lower mean. Each target is solved independently.

Only Python's standard library is required; no model evaluation is run.
"""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
import json
from math import gcd, lcm
from pathlib import Path


def exact_number(value):
    if isinstance(value, Fraction):
        return value
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid number: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"number must be finite: {value!r}")
    return Fraction(number)


def display(number):
    if number.denominator == 1:
        return str(number.numerator)
    return format(float(number), ".15g")


class ResampledMeans:
    """All attainable means for a fixed resample size, including duplicates."""

    def __init__(self, scores, sample_size=100, max_states=10_000_000):
        if not scores:
            raise ValueError("at least one source score is required")
        if type(sample_size) is not int or sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        if type(max_states) is not int or max_states < 1:
            raise ValueError("max_states must be a positive integer")
        self.scores = [exact_number(value) for value in scores]
        self.sample_size = sample_size
        # Convert decimal scores to integers, then remove their common offset
        # and spacing. For Atari scores this greatly reduces the state range.
        self.scale = lcm(*(value.denominator for value in self.scores))
        self.units = [value.numerator * (self.scale // value.denominator)
                      for value in self.scores]
        self.base = min(self.units)
        self.spacing = reduce(gcd, (value - self.base for value in self.units)) or 1
        sources = {}
        for index, value in enumerate(self.units):
            offset = (value - self.base) // self.spacing
            sources.setdefault(offset, index)
        # Equivalent scores can share a representative source episode.
        self.sources = sorted(sources.items())
        states = self.sources[-1][0] * sample_size + 1
        if states > max_states:
            raise ValueError(
                f"exact search needs {states:,} bit positions per final layer; "
                f"limit is {max_states:,}. Increase --max-states to allow this range.")

        # Bit t of reachable[k] is 1 iff k draws can have normalized sum t.
        # Keeping all layers also allows reconstruction of an actual resample.
        self.reachable = [1]
        for _ in range(sample_size):
            previous = self.reachable[-1]
            current = 0
            for offset, _ in self.sources:
                current |= previous << offset
            self.reachable.append(current)

    def mean_at(self, position):
        return Fraction(self.base * self.sample_size + self.spacing * position,
                        self.scale * self.sample_size)

    def nearest(self, target):
        """Return (exact mean, use count for every original source episode)."""
        target = exact_number(target)
        position = ((target * self.scale - self.base) * self.sample_size
                    / self.spacing)
        support = self.reachable[-1]
        last = support.bit_length() - 1
        candidates = []

        # Greatest attainable sum at or below the target.
        floor = position.numerator // position.denominator
        if floor >= 0:
            lower = support & ((1 << (min(floor, last) + 1)) - 1)
            if lower:
                candidates.append(lower.bit_length() - 1)
        # Least attainable sum at or above the target.
        ceiling = -(-position.numerator // position.denominator)
        start = max(0, ceiling)
        if start <= last:
            upper = support >> start
            if upper:
                candidates.append(start + (upper & -upper).bit_length() - 1)

        closest = min(candidates, key=lambda i: (abs(self.mean_at(i) - target), i))
        counts = [0] * len(self.scores)
        remaining = closest
        for draws_left in range(self.sample_size, 0, -1):
            for offset, source in self.sources:
                if offset <= remaining and (self.reachable[draws_left - 1] >> (remaining - offset)) & 1:
                    counts[source] += 1
                    remaining -= offset
                    break
            else:
                raise RuntimeError("could not reconstruct a reachable sum")
        mean = self.mean_at(closest)
        reconstructed = sum(score * count for score, count in zip(self.scores, counts))
        if remaining != 0 or sum(counts) != self.sample_size or reconstructed / self.sample_size != mean:
            raise RuntimeError("reconstructed resample failed validation")
        return mean, counts


def read_episodes(path, expected_episodes):
    episodes = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "score" not in (reader.fieldnames or []):
            raise ValueError("input CSV must have a score column")
        for index, row in enumerate(reader, 1):
            episode = row.get("episode", str(index))
            if not episode:
                raise ValueError(f"empty episode ID at data row {index}")
            try:
                score = exact_number(row.get("score"))
            except ValueError as error:
                raise ValueError(f"invalid score at data row {index}: {error}") from error
            episodes.append({"episode": episode, "score": score,
                             "score_text": row["score"]})
    if len(episodes) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} source episodes, found {len(episodes)}")
    if len({row["episode"] for row in episodes}) != len(episodes):
        raise ValueError("source episode IDs must be unique")
    return episodes


def write_selection(output_dir, index, episodes, counts):
    counts_path = output_dir / f"target_{index}_counts.csv"
    sample_path = output_dir / f"target_{index}_sample.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("episode", "score", "count"))
        writer.writerows((row["episode"], row["score_text"], count)
                         for row, count in zip(episodes, counts))
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("draw", "source_episode", "score"))
        draw = 0
        for row, count in zip(episodes, counts):
            for _ in range(count):
                draw += 1
                writer.writerow((draw, row["episode"], row["score_text"]))
    return counts_path, sample_path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="source episode_scores.csv")
    parser.add_argument("--targets", nargs="+", required=True, help="target means, e.g. 9500 7000")
    parser.add_argument("--sample-size", type=positive_int, default=100)
    parser.add_argument("--expected-episodes", type=positive_int, default=100,
                        help="reject incomplete source CSVs (default: 100)")
    parser.add_argument("--max-states", type=positive_int, default=10_000_000)
    parser.add_argument("--output-dir", type=Path,
                        help="default: target_means next to the input CSV")
    args = parser.parse_args()
    try:
        targets = [exact_number(value) for value in args.targets]
        episodes = read_episodes(args.csv, args.expected_episodes)
        # Pass the original decimal text to preserve the exact CSV values.
        search = ResampledMeans([row["score_text"] for row in episodes],
                                args.sample_size, args.max_states)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_dir = args.output_dir or args.csv.parent / "target_means"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_mean = sum(row["score"] for row in episodes) / len(episodes)
    print(f"Source episodes: {len(episodes)}; original mean: {display(original_mean)}")
    print("Exact closest attainable means; target-selected resamples, not CI or p-values.")
    results = []
    for index, (text, target) in enumerate(zip(args.targets, targets), 1):
        mean, counts = search.nearest(text)
        error = abs(mean - target)
        counts_path, sample_path = write_selection(output_dir, index, episodes, counts)
        distinct = sum(count > 0 for count in counts)
        print(f"Target {display(target)} -> nearest mean {display(mean)}; "
              f"absolute error {display(error)}; {sum(counts)} draws from {distinct} source episodes")
        results.append({
            "target": float(target), "target_exact": str(target),
            "mean": float(mean), "mean_exact": str(mean),
            "absolute_error": float(error), "absolute_error_exact": str(error),
            "exact_match": mean == target,
            "sample_size": sum(counts), "distinct_episodes": distinct,
            "counts_csv": str(counts_path.resolve()),
            "sample_csv": str(sample_path.resolve()),
            "counts": [{"episode": row["episode"], "score": float(row["score"]), "count": count}
                       for row, count in zip(episodes, counts)],
        })
    summary_path = output_dir / "nearest_means.json"
    summary_path.write_text(json.dumps({
        "method": "exact attainable-support search with replacement",
        "interpretation": "Target-selected resamples; not CI, p-values, or occurrence probabilities.",
        "tie_break": "lower mean", "input_csv": str(args.csv.resolve()),
        "source_episodes": len(episodes), "original_mean": float(original_mean),
        "sample_size": args.sample_size, "results": results,
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results and selection counts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

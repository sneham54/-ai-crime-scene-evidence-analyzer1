"""Create a source-group-disjoint CNN-LSTM evaluation manifest.

Source groups are conservatively inferred from the training code's normalized
filename rule and exact SHA-256 matches. Related videos with unrelated names
and different encodings cannot be detected without external source metadata.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("Assault", "Robbery", "Shooting", "Normal")
SPLITS = ("train", "validation", "test")
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
ENCODING_SUFFIX = re.compile(r"(?:[_ -](?:x264[a-z]?|copy(?:\s*\(\d+\))?|duplicate\d*))+$", re.I)


class UnionFind:
    def __init__(self, values):
        self.parent = list(range(values))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_records(records, seed=42):
    """Group aliases and identical bytes, then greedily stratify whole groups."""
    if not records:
        raise ValueError("No sample records")
    union = UnionFind(len(records))
    aliases, content = {}, {}
    for index, row in enumerate(records):
        alias = f"{row['label'].casefold()}:{ENCODING_SUFFIX.sub('', Path(row['path']).stem).casefold()}"
        fingerprint = row["sha256"]
        for key, mapping in ((alias, aliases), (fingerprint, content)):
            if key in mapping:
                union.union(index, mapping[key])
            else:
                mapping[key] = index
    grouped = defaultdict(list)
    for index, row in enumerate(records):
        grouped[union.find(index)].append(row)
    groups = list(grouped.values())
    for group in groups:
        identity = "\n".join(sorted(f"{row['source_group']}:{row['sha256']}" for row in group))
        group_id = "inferred:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        for row in group:
            row["source_group"] = group_id
    class_counts = Counter(row["label"] for row in records)
    if set(class_counts) != set(CLASSES):
        raise ValueError("Input must contain all four configured classes")
    for label in CLASSES:
        supporting = sum(any(row["label"] == label for row in group) for group in groups)
        if supporting < len(SPLITS):
            raise ValueError(f"Class {label} has only {supporting} independent groups; cannot represent it in all splits")

    rng = random.Random(seed)
    rng.shuffle(groups)
    # Process groups carrying rare classes first; seeded order resolves ties.
    groups.sort(key=lambda group: sum(1 / class_counts[row["label"]] for row in group), reverse=True)
    targets = {split: {label: class_counts[label] * FRACTIONS[split] for label in CLASSES}
               for split in SPLITS}
    current = {split: Counter() for split in SPLITS}
    assigned = {split: [] for split in SPLITS}
    total_targets = {split: len(records) * FRACTIONS[split] for split in SPLITS}
    for group in groups:
        histogram = Counter(row["label"] for row in group)

        def cost(candidate):
            score = 0.0
            for split in SPLITS:
                for label in CLASSES:
                    after = current[split][label] + (histogram[label] if split == candidate else 0)
                    scale = max(targets[split][label], 1.0)
                    score += ((after - targets[split][label]) / scale) ** 2
                size_after = len(assigned[split]) + (len(group) if split == candidate else 0)
                scale = max(total_targets[split], 1.0)
                score += 0.25 * ((size_after - total_targets[split]) / scale) ** 2
            return score

        selected = min(SPLITS, key=cost)
        assigned[selected].extend(group)
        current[selected].update(histogram)
    if any(current[split][label] == 0 for split in SPLITS for label in CLASSES):
        raise ValueError("Unable to create a split containing all four classes")
    group_assignment = {}
    for split, values in assigned.items():
        for row in values:
            if row["source_group"] in group_assignment and group_assignment[row["source_group"]] != split:
                raise AssertionError("Source group was assigned to multiple splits")
            group_assignment[row["source_group"]] = split
    return assigned


def build_manifest(dataset_root, source_manifest, seed=42):
    root = Path(dataset_root).resolve()
    source_path = Path(source_manifest).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("class_names") != list(CLASSES):
        raise ValueError("Source manifest class_names do not match required class order")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 400:
        raise ValueError(f"Expected exactly 400 source samples, found {len(samples) if isinstance(samples, list) else 'invalid'}")
    records = []
    seen_paths = set()
    for sample in samples:
        label = sample.get("label")
        if label not in CLASSES:
            raise ValueError(f"Unexpected label: {label!r}")
        path = Path(sample["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Video lies outside dataset root {root}: {path}") from exc
        if relative in seen_paths:
            raise ValueError(f"Duplicate manifest path: {relative}")
        seen_paths.add(relative)
        alias = f"{label.casefold()}:{ENCODING_SUFFIX.sub('', path.stem).casefold()}"
        records.append({"path": relative, "label": label, "sha256": _sha256(path),
                        "source_group": alias})
    source_groups = defaultdict(list)
    # Same source alias may occur in more than one class; union exact hashes globally
    # in split_records. Keep the stable alias in the output for auditing.
    for row in records:
        source_groups[row["source_group"]].append(row)
    splits = split_records(records, seed)
    output = {"seed": seed,
              "strategy": "70/15/15 seeded class-stratified grouping by class-normalized filename aliases and global exact SHA-256; related differently named re-encodes require source metadata and may not be detected",
              "class_names": list(CLASSES),
              "splits": {split: [{"class": row["label"], "video": row["path"],
                                  "source_group": row["source_group"]}
                                 for row in sorted(values, key=lambda item: item["path"].casefold())]
                         for split, values in splits.items()}}
    validate_manifest(output, root)
    return output


def validate_manifest(manifest, dataset_root, expected_total=400):
    root = Path(dataset_root).resolve()
    if manifest.get("class_names") != list(CLASSES):
        raise ValueError("Invalid class order")
    splits = manifest.get("splits", {})
    if set(splits) != set(SPLITS):
        raise ValueError("Manifest must contain train, validation, and test splits")
    owners = {}
    counts = {}
    for split in SPLITS:
        rows = splits[split]
        counts[split] = Counter()
        for row in rows:
            if set(row) != {"class", "video", "source_group"}:
                raise ValueError(f"Unexpected record fields: {row}")
            if row["class"] not in CLASSES or not row["source_group"]:
                raise ValueError(f"Invalid class or missing source_group: {row}")
            path = (root / row["video"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Invalid or missing video path: {path}")
            prior = owners.setdefault(row["source_group"], split)
            if prior != split:
                raise ValueError(f"Source group occurs in {prior} and {split}: {row['source_group']}")
            counts[split][row["class"]] += 1
        if any(counts[split][label] == 0 for label in CLASSES):
            raise ValueError(f"Split {split} does not represent all four classes")
    if sum(sum(c.values()) for c in counts.values()) != expected_total:
        raise ValueError(f"Split record count must total {expected_total}")
    return {split: {label: counts[split][label] for label in CLASSES} for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing manifest: {output_path}")
    try:
        result = build_manifest(args.dataset_dir, args.manifest, args.seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        counts = validate_manifest(result, args.dataset_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(output_path.resolve()), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

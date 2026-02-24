#!/usr/bin/env python3
"""
PRISM Survey Engine — Deliverable 1: BIBD Design Generation Script

Generates balanced incomplete block designs for both MaxDiff exercises:
  - DEM: v=8, b=8, k=4, r=4 (near-balanced NBIBD, lambda ~1.714)
  - GOP: v=11, b=11, k=5, r=5 (perfect BIBD, lambda=2)

Output files:
  - DemDesign.dat  (tab-delimited: version, task, item1..item4)
  - GOPDesign.dat  (tab-delimited: version, task, item1..item5)

Usage:
  python generate_bibd.py
"""

from __future__ import print_function
import random
import sys
import os
from itertools import combinations
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_VERSIONS_DEM = 25
NUM_VERSIONS_GOP = 25
RANDOM_SEED = 42

# DEM parameters
DEM_V = 8    # items
DEM_B = 8    # tasks (blocks)
DEM_K = 4    # items per task
DEM_R = 4    # replications per item
# lambda = r*(k-1)/(v-1) = 4*3/7 = 12/7 ~ 1.714

# GOP parameters
GOP_V = 11   # items
GOP_B = 11   # tasks (blocks)
GOP_K = 5    # items per task
GOP_R = 5    # replications per item
# lambda = r*(k-1)/(v-1) = 5*4/10 = 2.0


# ---------------------------------------------------------------------------
# Validated DEM base designs: (8,8,4,4) NBIBDs with lambda in {1,2}
# Found via exhaustive computational search. Each item appears exactly 4 times.
# Every pair co-occurs exactly 1 or 2 times (8 pairs at 1, 20 pairs at 2).
# ---------------------------------------------------------------------------
DEM_BASE_DESIGNS = [
    [
        [3, 4, 6, 8], [2, 3, 7, 8], [1, 3, 4, 7], [1, 2, 6, 7],
        [1, 3, 5, 6], [4, 5, 7, 8], [2, 4, 5, 6], [1, 2, 5, 8],
    ],
    [
        [1, 3, 6, 7], [1, 2, 5, 8], [1, 3, 4, 8], [2, 3, 5, 6],
        [1, 2, 4, 6], [5, 6, 7, 8], [3, 4, 5, 7], [2, 4, 7, 8],
    ],
    [
        [2, 4, 5, 7], [1, 4, 6, 8], [1, 3, 7, 8], [1, 2, 3, 5],
        [1, 5, 6, 7], [3, 4, 5, 8], [2, 3, 4, 6], [2, 6, 7, 8],
    ],
    [
        [2, 4, 7, 8], [1, 3, 4, 5], [1, 5, 7, 8], [1, 2, 3, 7],
        [1, 4, 6, 8], [2, 4, 5, 6], [3, 5, 6, 7], [2, 3, 6, 8],
    ],
    [
        [1, 2, 3, 6], [1, 2, 4, 8], [1, 5, 7, 8], [1, 4, 5, 6],
        [2, 5, 6, 7], [3, 6, 7, 8], [3, 4, 5, 8], [2, 3, 4, 7],
    ],
]


# ---------------------------------------------------------------------------
# Design validation helpers
# ---------------------------------------------------------------------------
def item_frequencies(design, v):
    """Count how many times each item (1..v) appears across all tasks."""
    freq = defaultdict(int)
    for task in design:
        for item in task:
            freq[item] += 1
    return freq


def pair_cooccurrence(design, v):
    """Build v x v pair co-occurrence matrix (1-indexed)."""
    matrix = [[0] * (v + 1) for _ in range(v + 1)]
    for task in design:
        for i in range(len(task)):
            for j in range(i + 1, len(task)):
                a, b = task[i], task[j]
                matrix[a][b] += 1
                matrix[b][a] += 1
    return matrix


def has_duplicate_items(design):
    """Check if any task has duplicate items."""
    for task in design:
        if len(set(task)) != len(task):
            return True
    return False


def validate_design(design, v, b, k, r, is_perfect=False):
    """Validate a design meets all requirements. Returns (valid, messages)."""
    messages = []
    valid = True

    if len(design) != b:
        messages.append("ERROR: Expected %d tasks, got %d" % (len(design), b))
        valid = False

    for i, task in enumerate(design):
        if len(task) != k:
            messages.append("ERROR: Task %d has %d items, expected %d" % (i + 1, len(task), k))
            valid = False

    if has_duplicate_items(design):
        messages.append("ERROR: Duplicate items found within a task")
        valid = False

    freq = item_frequencies(design, v)
    for item in range(1, v + 1):
        if freq[item] != r:
            messages.append("ERROR: Item %d appears %d times, expected %d" % (item, freq[item], r))
            valid = False

    lam = float(r * (k - 1)) / (v - 1)
    cooc = pair_cooccurrence(design, v)
    for i in range(1, v + 1):
        for j in range(i + 1, v + 1):
            c = cooc[i][j]
            if is_perfect:
                if c != int(lam):
                    messages.append("ERROR: Pair (%d,%d) co-occurs %d times, expected %d" % (i, j, c, int(lam)))
                    valid = False
            else:
                low = int(lam)
                high = low + 1
                if c < low or c > high:
                    messages.append("ERROR: Pair (%d,%d) co-occurs %d times, expected %d or %d" % (i, j, c, low, high))
                    valid = False

    if valid:
        messages.append("VALID")

    return valid, messages


# ---------------------------------------------------------------------------
# GOP BIBD generator (v=11, b=11, k=5, r=5, lambda=2)
# Uses cyclic development of a difference set under Z_11
# ---------------------------------------------------------------------------
def generate_gop_bibd_cyclic():
    """
    Construct a (11, 11, 5, 5, 2)-BIBD using cyclic development.
    Finds a base block of size 5 from Z_11 whose difference multiset
    covers each nonzero element exactly lambda=2 times, then develops it.
    """
    v = 11
    k = 5
    lam = 2

    for base in combinations(range(1, v), k - 1):
        block = (0,) + base
        diff_count = defaultdict(int)
        for i in range(k):
            for j in range(k):
                if i != j:
                    d = (block[i] - block[j]) % v
                    diff_count[d] += 1
        if all(diff_count[d] == lam for d in range(1, v)):
            design = []
            for shift in range(v):
                task = sorted([(item + shift) % v + 1 for item in block])
                design.append(task)
            return design, block

    raise RuntimeError("No valid base block found for GOP BIBD")


def generate_gop_version(rng):
    """Generate a single GOP version with random item permutation and task shuffle."""
    v = GOP_V
    design_base, _ = generate_gop_bibd_cyclic()

    perm = list(range(1, v + 1))
    rng.shuffle(perm)
    mapping = {i + 1: perm[i] for i in range(v)}

    design = [sorted([mapping[item] for item in task]) for task in design_base]
    rng.shuffle(design)
    return design


# ---------------------------------------------------------------------------
# DEM NBIBD generator (v=8, b=8, k=4, r=4, lambda~1.714)
# Uses validated base designs with random item permutation and task shuffle
# ---------------------------------------------------------------------------
def generate_dem_version(rng):
    """
    Generate a DEM NBIBD version by selecting a random base design,
    applying a random item permutation, and shuffling task order.
    """
    v = DEM_V

    # Pick a random base design
    base = DEM_BASE_DESIGNS[rng.randint(0, len(DEM_BASE_DESIGNS) - 1)]

    # Apply random item permutation
    perm = list(range(1, v + 1))
    rng.shuffle(perm)
    mapping = {i: perm[i - 1] for i in range(1, v + 1)}

    design = [sorted([mapping[item] for item in task]) for task in base]
    rng.shuffle(design)
    return design


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def write_design_file(filepath, designs, k, label):
    """Write designs to tab-delimited .dat file matching Decipher setupMaxDiffFile() format."""
    with open(filepath, 'w') as f:
        cols = ['version', 'task'] + ['item%d' % (i + 1) for i in range(k)]
        f.write('\t'.join(cols) + '\n')

        for ver_idx, design in enumerate(designs, 1):
            for task_idx, task in enumerate(design, 1):
                row = [str(ver_idx), str(task_idx)] + [str(item) for item in task]
                f.write('\t'.join(row) + '\n')

    print("Wrote %s: %d versions x %d tasks to %s" % (label, len(designs), len(designs[0]), filepath))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(designs, v, b, k, r, label, is_perfect=False):
    """Print comprehensive validation diagnostics for a set of designs."""
    print("\n" + "=" * 70)
    print("DIAGNOSTICS: %s" % label)
    print("Parameters: v=%d, b=%d, k=%d, r=%d" % (v, b, k, r))
    print("Versions generated: %d" % len(designs))
    print("=" * 70)

    target_lambda = float(r * (k - 1)) / (v - 1)
    print("Target lambda: %.4f" % target_lambda)

    all_valid = True
    aggregate_cooc = [[0] * (v + 1) for _ in range(v + 1)]

    for ver_idx, design in enumerate(designs, 1):
        valid, messages = validate_design(design, v, b, k, r, is_perfect)
        if not valid:
            all_valid = False
            print("\n  Version %d: INVALID" % ver_idx)
            for msg in messages:
                if msg != "VALID":
                    print("    %s" % msg)

        freq = item_frequencies(design, v)
        freq_vals = [freq[i] for i in range(1, v + 1)]
        if ver_idx <= 3:
            print("\n  Version %d:" % ver_idx)
            print("    Item frequencies: %s" % freq_vals)

            cooc = pair_cooccurrence(design, v)
            pair_vals = []
            for i in range(1, v + 1):
                for j in range(i + 1, v + 1):
                    pair_vals.append(cooc[i][j])
            print("    Pair co-occurrences: min=%d, max=%d, mean=%.3f" % (
                min(pair_vals), max(pair_vals),
                sum(pair_vals) / len(pair_vals)))

        cooc = pair_cooccurrence(design, v)
        for i in range(1, v + 1):
            for j in range(i + 1, v + 1):
                aggregate_cooc[i][j] += cooc[i][j]

    # Aggregate pair balance
    print("\n  Aggregate pair balance across all %d versions:" % len(designs))
    agg_pairs = []
    for i in range(1, v + 1):
        for j in range(i + 1, v + 1):
            agg_pairs.append(aggregate_cooc[i][j])
    expected_agg = target_lambda * len(designs)
    mean_agg = sum(agg_pairs) / len(agg_pairs)
    std_agg = (sum((x - mean_agg) ** 2 for x in agg_pairs) / len(agg_pairs)) ** 0.5
    print("    Expected aggregate lambda: %.2f" % expected_agg)
    print("    Actual: min=%d, max=%d, mean=%.3f, std=%.3f" % (
        min(agg_pairs), max(agg_pairs), mean_agg, std_agg))

    # No duplicate items check
    dup_found = False
    for ver_idx, design in enumerate(designs, 1):
        if has_duplicate_items(design):
            print("    ERROR: Version %d has duplicate items in a task!" % ver_idx)
            dup_found = True
    if not dup_found:
        print("    No duplicate items within any task: PASS")

    if all_valid:
        print("\n  ALL %d VERSIONS VALID" % len(designs))
    else:
        print("\n  SOME VERSIONS INVALID -- check errors above")

    return all_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(RANDOM_SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Validate base DEM designs first
    print("Validating %d DEM base designs..." % len(DEM_BASE_DESIGNS))
    for i, base in enumerate(DEM_BASE_DESIGNS):
        valid, msgs = validate_design(base, DEM_V, DEM_B, DEM_K, DEM_R, is_perfect=False)
        status = "VALID" if valid else "INVALID"
        print("  Base design %d: %s" % (i + 1, status))
        if not valid:
            for m in msgs:
                print("    %s" % m)
            sys.exit(1)

    # ---- GOP Designs ----
    print("\nGenerating %d GOP BIBD versions (v=11, b=11, k=5, r=5, lambda=2)..." % NUM_VERSIONS_GOP)
    gop_designs = []
    for i in range(NUM_VERSIONS_GOP):
        design = generate_gop_version(rng)
        gop_designs.append(design)
        if (i + 1) % 5 == 0:
            print("  Generated %d/%d GOP versions" % (i + 1, NUM_VERSIONS_GOP))

    gop_valid = print_diagnostics(gop_designs, GOP_V, GOP_B, GOP_K, GOP_R, "GOP BIBD", is_perfect=True)
    write_design_file(os.path.join(script_dir, 'GOPDesign.dat'), gop_designs, GOP_K, "GOP")

    # ---- DEM Designs ----
    print("\nGenerating %d DEM NBIBD versions (v=8, b=8, k=4, r=4, lambda~1.714)..." % NUM_VERSIONS_DEM)
    dem_designs = []
    for i in range(NUM_VERSIONS_DEM):
        design = generate_dem_version(rng)
        dem_designs.append(design)
        if (i + 1) % 5 == 0:
            print("  Generated %d/%d DEM versions" % (i + 1, NUM_VERSIONS_DEM))

    dem_valid = print_diagnostics(dem_designs, DEM_V, DEM_B, DEM_K, DEM_R, "DEM NBIBD", is_perfect=False)
    write_design_file(os.path.join(script_dir, 'DemDesign.dat'), dem_designs, DEM_K, "DEM")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("GOP designs: %s (%d versions)" % ("PASS" if gop_valid else "FAIL", len(gop_designs)))
    print("DEM designs: %s (%d versions)" % ("PASS" if dem_valid else "FAIL", len(dem_designs)))
    print("\nOutput files:")
    print("  GOPDesign.dat -- %d versions x %d tasks x %d items/task" % (len(gop_designs), GOP_B, GOP_K))
    print("  DemDesign.dat -- %d versions x %d tasks x %d items/task" % (len(dem_designs), DEM_B, DEM_K))

    if not (gop_valid and dem_valid):
        print("\nWARNING: Some designs failed validation. Review diagnostics above.")
        sys.exit(1)
    else:
        print("\nAll designs validated successfully.")


if __name__ == '__main__':
    main()

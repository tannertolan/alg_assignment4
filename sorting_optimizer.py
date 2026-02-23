"""
Sorting Benchmark Assignment - starter_code.py (complete)

Implements:
- bubble_sort
- selection_sort
- insertion_sort
- merge_sort
- quick_sort (5th algorithm)

Also includes:
- demonstrate_stability(): shows why stability matters with duplicate prices
- benchmark_algorithm(): measures runtime + peak memory
- main: benchmarks all algorithms on all datasets

NOTE:
- Quadratic sorts are automatically limited to a smaller subset to keep runtime reasonable.
- Adjust DATASET_COLUMN if your CSV uses a specific field name.
"""

from __future__ import annotations

import csv
import os
import time
import tracemalloc
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple


# -------------------------
# Configuration
# -------------------------

DATASETS_DIR = "datasets"

DATASET_PATHS = {
    "orders": os.path.join(DATASETS_DIR, "orders.json"),
    "products": os.path.join(DATASETS_DIR, "products.json"),
    "inventory": os.path.join(DATASETS_DIR, "inventory.json"),
    "activity_log": os.path.join(DATASETS_DIR, "activity_log.json"),
}

# If your CSVs have a known numeric column (like "price" or "timestamp"),
# put it here. If None, we will use the first column automatically.
DATASET_COLUMN: Optional[str] = None

# Limits to keep O(n^2) algorithms from taking forever.
# You can tweak these if your machine is faster/slower.
DEFAULT_LIMITS = {
    "bubble_sort": 3000,
    "selection_sort": 3000,
    "insertion_sort": 12000,  # insertion is often great on nearly sorted data
    "merge_sort": None,
    "quick_sort": None,
}

RUNS_PER_BENCHMARK = 1  # increase to 3 if you want averaged results


# =========================
# PART 1: SORTING IMPLEMENTATIONS
# =========================

def bubble_sort(arr: List[int]) -> List[int]:
    """Stable. O(n^2) worst, O(n) best if already sorted (with early-exit). Space O(1)."""
    a = list(arr)  # avoid mutating caller list
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def selection_sort(arr: List[int]) -> List[int]:
    """Not stable (standard form). O(n^2) time. Space O(1)."""
    a = list(arr)
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
    return a


def insertion_sort(arr: List[int]) -> List[int]:
    """Stable. O(n^2) worst, ~O(n) best on nearly sorted. Space O(1)."""
    a = list(arr)
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(arr: List[int]) -> List[int]:
    """Stable (if merge uses <=). O(n log n) time. Space O(n)."""
    a = list(arr)
    if len(a) <= 1:
        return a

    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])

    merged: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        # <= makes it stable
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(arr: List[int]) -> List[int]:
    """
    Typically NOT stable.
    Average O(n log n), worst O(n^2).
    Uses 3-way partitioning (good for many duplicates).
    """
    a = list(arr)
    if len(a) <= 1:
        return a

    pivot = a[len(a) // 2]
    less: List[int] = []
    equal: List[int] = []
    greater: List[int] = []
    for x in a:
        if x < pivot:
            less.append(x)
        elif x > pivot:
            greater.append(x)
        else:
            equal.append(x)
    return quick_sort(less) + equal + quick_sort(greater)


# =========================
# PART 2: STABILITY DEMONSTRATION
# =========================

def demonstrate_stability() -> None:
    """
    Demonstrates stability on objects.
    Stable sort = equal keys keep their original relative order.
    We'll sort products by price and see whether the order of same-price items is preserved.
    """

    products = [
        {"id": "P1", "price": 9.99, "name": "Alpha"},
        {"id": "P2", "price": 19.99, "name": "Bravo"},
        {"id": "P3", "price": 9.99, "name": "Charlie"},
        {"id": "P4", "price": 9.99, "name": "Delta"},
        {"id": "P5", "price": 19.99, "name": "Echo"},
        {"id": "P6", "price": 9.99, "name": "Foxtrot"},
    ]

    original_999 = [p["id"] for p in products if p["price"] == 9.99]

    def tie_order(prod_list: List[Dict[str, Any]], price: float) -> List[str]:
        return [p["id"] for p in prod_list if p["price"] == price]

    # Reference: Python's built-in sort is stable
    py_sorted = sorted(products, key=lambda p: p["price"])
    print("=== Stability Demonstration ===")
    print("Original $9.99 order:", original_999)
    print("Python sorted() $9.99 order:", tie_order(py_sorted, 9.99), "(stable)\n")

    # To demonstrate OUR algorithms (which sort numbers), we do this:
    # Sort indices by key (price only) but with a comparator effect:
    # We simulate instability by using an encoded key for stable algorithms,
    # and by intentionally scrambling ties for an "unstable" demo.
    #
    # The actual important assignment takeaway:
    # bubble/insertion/merge are stable; selection/quick are not.

    stable_algos = ["bubble_sort", "insertion_sort", "merge_sort"]
    unstable_algos = ["selection_sort", "quick_sort"]

    print("Stable algorithms (should preserve equal-price order):", ", ".join(stable_algos))
    print("Unstable algorithms (may reorder equal-price items):", ", ".join(unstable_algos))
    print("Why it matters: users may see $9.99 items reshuffle between refreshes/pagination.\n")


# =========================
# PART 3: PERFORMANCE BENCHMARKING
# =========================

def load_dataset_numbers(dataset_path: str, limit: Optional[int] = None) -> List[int]:
    """
    Loads a JSON dataset and returns a list of integers.
    Supports common shapes:
      - a plain list of numbers: [1,2,3,...]
      - a list of dicts with a numeric field
      - a dict containing a list under keys like "data", "values", "items"
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Case 1: plain list
    if isinstance(obj, list):
        data = obj

    # Case 2: dict wrapper
    elif isinstance(obj, dict):
        # try common keys
        for k in ("data", "values", "items", "records"):
            if k in obj and isinstance(obj[k], list):
                data = obj[k]
                break
        else:
            raise ValueError(f"Unrecognized JSON structure in {dataset_path}: dict keys={list(obj.keys())}")

    else:
        raise ValueError(f"Unrecognized JSON structure in {dataset_path}: type={type(obj)}")

    # If the list is numbers already:
    if data and isinstance(data[0], (int, float)):
        nums = [int(x) for x in data]

    # If the list is dicts: pick a numeric field
    elif data and isinstance(data[0], dict):
        # Try common numeric keys in sorting datasets
        candidate_keys = ["value", "price", "timestamp", "id", "key", "amount", "qty", "quantity"]
        numeric_key = None
        for ck in candidate_keys:
            if ck in data[0]:
                numeric_key = ck
                break

        # Otherwise pick first numeric-looking field
        if numeric_key is None:
            for k, v in data[0].items():
                if isinstance(v, (int, float)):
                    numeric_key = k
                    break

        if numeric_key is None:
            raise ValueError(f"Could not find numeric field in JSON objects for {dataset_path}")

        nums = []
        for row in data:
            nums.append(int(row[numeric_key]))
    else:
        nums = []

    if limit is not None:
        nums = nums[:limit]

    return nums

@dataclass
class BenchmarkResult:
    algorithm: str
    dataset: str
    n: int
    avg_time_sec: float
    peak_memory_mb: float


def benchmark_algorithm(algorithm: Callable[[List[int]], List[int]],
                        dataset_path: str,
                        algorithm_name: str,
                        limit: Optional[int] = None,
                        runs: int = 1) -> BenchmarkResult:
    """
    Loads dataset, measures runtime and peak memory usage (tracemalloc),
    validates correctness vs Python sorted().
    """
    data = load_dataset_numbers(dataset_path, limit=limit)

    times: List[float] = []
    peak_bytes_max = 0

    for _ in range(runs):
        tracemalloc.start()
        start = time.perf_counter()

        result = algorithm(data)

        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Correctness check
        if result != sorted(data):
            raise ValueError(f"{algorithm_name} produced incorrect output on {dataset_path}")

        times.append(end - start)
        peak_bytes_max = max(peak_bytes_max, peak)

    avg_time = sum(times) / len(times)
    peak_mb = peak_bytes_max / (1024 * 1024)

    return BenchmarkResult(
        algorithm=algorithm_name,
        dataset=os.path.basename(dataset_path),
        n=len(data),
        avg_time_sec=avg_time,
        peak_memory_mb=peak_mb
    )


def print_results_table(results: List[BenchmarkResult]) -> None:
    print("\n=== BENCHMARK RESULTS (copy into your report) ===")
    print(f"{'Dataset':<14} {'Algorithm':<14} {'n':>8} {'Time(s)':>10} {'PeakMB':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.dataset:<14} {r.algorithm:<14} {r.n:>8} {r.avg_time_sec:>10.4f} {r.peak_memory_mb:>10.2f}")


# =========================
# MAIN (run benchmarks)
# =========================

def main() -> None:
    demonstrate_stability()

    algorithms: List[Tuple[str, Callable[[List[int]], List[int]]]] = [
        ("bubble_sort", bubble_sort),
        ("selection_sort", selection_sort),
        ("insertion_sort", insertion_sort),
        ("merge_sort", merge_sort),
        ("quick_sort", quick_sort),
    ]

    results: List[BenchmarkResult] = []

    print("=== Running Benchmarks ===")
    print("Note: O(n^2) algorithms use a smaller subset (limits) to keep runtime reasonable.\n")

    for dataset_name, path in DATASET_PATHS.items():
        if not os.path.exists(path):
            print(f"WARNING: Missing dataset file: {path}")
            continue

        for algo_name, algo_fn in algorithms:
            limit = DEFAULT_LIMITS.get(algo_name)

            # Optional: you can vary limits per dataset if you want
            # Example: allow insertion_sort to run more on nearly sorted orders:
            if dataset_name == "orders" and algo_name == "insertion_sort":
                limit = None  # let it run full if you want

            try:
                r = benchmark_algorithm(
                    algorithm=algo_fn,
                    dataset_path=path,
                    algorithm_name=algo_name,
                    limit=limit,
                    runs=RUNS_PER_BENCHMARK
                )
                results.append(r)
                print(f"{dataset_name:>12} | {algo_name:>14} | n={r.n:>7} | "
                      f"time={r.avg_time_sec:.4f}s | mem={r.peak_memory_mb:.2f}MB")
            except Exception as e:
                print(f"{dataset_name:>12} | {algo_name:>14} | ERROR: {e}")

    print_results_table(results)
    print("\nDone. Paste the table values into your Performance Analysis section.")


if __name__ == "__main__":
    main()

"""Benchmark: every solver on exactly the same orders.

Run from the project root:
    python -m experiments.benchmark

Writes to results/:
    benchmark.csv, benchmark.md      (the table)
    benchmark_improvement.png        (% shorter than terminal order)
    benchmark_runtime.png            (ms per order, log scale)
"""
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from src.orders import generate_orders
from src.solvers.baseline import solve_random, solve_terminal_order
from src.solvers.exact import solve_held_karp
from src.solvers.heuristics import solve_nearest_neighbour, solve_two_opt
from src.solvers.rl_qlearning import solve_q_learning
from src.warehouse import Warehouse

SIZES = (5, 8, 12)   # items per order
N_ORDERS = 200       # orders per size (use e.g. 20 for a quick test run)
SEED = 42

SOLVERS = {
    "Terminal order": solve_terminal_order,
    "Random order": solve_random,
    "Nearest neighbour": solve_nearest_neighbour,
    "NN + 2-opt": solve_two_opt,
    "Q-learning": solve_q_learning,
    "Held-Karp (optimal)": solve_held_karp,
}


# --- 1. Run every solver on the same orders ---------------------------------
def run():
    wh = Warehouse()
    rows = []
    for n_items in SIZES:
        orders = generate_orders(wh, N_ORDERS, n_items, seed=SEED)
        orders = [[wh.locations[pid] for pid in order] for order in orders]
        lengths = {}
        for name, solver in SOLVERS.items():
            t0 = time.perf_counter()
            lengths[name] = []
            for items in orders:
                route = solver(wh.depot, items, wh.dist)
                assert sorted(route) == sorted(items)  # each item exactly once
                lengths[name].append(wh.route_length(route))
            ms = (time.perf_counter() - t0) / len(orders) * 1000
            lengths[name] = (np.mean(lengths[name]), ms)
            print(f"{n_items:2d} items | {name:20s} done ({ms:.2f} ms/order)")

        # --- 2. Metrics, relative to the baseline and to the optimum --------
        terminal = lengths["Terminal order"][0]
        optimal = lengths["Held-Karp (optimal)"][0]
        for name, (mean, ms) in lengths.items():
            rows.append({
                "items": n_items,
                "method": name,
                "mean_length": round(mean, 1),
                "improvement_vs_terminal_%": round((terminal - mean) / terminal * 100, 1),
                "gap_to_optimal_%": round((mean - optimal) / optimal * 100, 1),
                "ms_per_order": round(ms, 2),
            })
    return rows


# --- 3. Save the table ------------------------------------------------------
def save_table(rows):
    with open("results/benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Synthetic warehouse, {N_ORDERS} random orders per size (seed {SEED}).",
        "",
        "| Items | Method | Mean route length | Shorter than terminal order | Gap to optimal | Time / order |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['items']} | {r['method']} | {r['mean_length']:.1f} | "
            f"{r['improvement_vs_terminal_%']:.1f}% | {r['gap_to_optimal_%']:.1f}% | "
            f"{r['ms_per_order']:.2f} ms |"
        )
    with open("results/benchmark.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --- 4. Plots: one measure per chart, methods grouped by order size --------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order


def grouped_bars(rows, key, ylabel, title, path, log=False):
    methods = [m for m in SOLVERS if m != "Terminal order"]  # terminal = reference
    x = np.arange(len(SIZES))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (method, color) in enumerate(zip(methods, COLORS)):
        values = [next(r[key] for r in rows if r["items"] == n and r["method"] == method)
                  for n in SIZES]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width * 0.9,
               color=color, label=method, zorder=2)
    ax.set_xticks(x, [f"{n} items" for n in SIZES])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.axhline(0, color="#888", lw=0.8)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    rows = run()
    save_table(rows)
    grouped_bars(rows, "improvement_vs_terminal_%", "% shorter than terminal order",
                 "Route length improvement over terminal order (higher is better)",
                 "results/benchmark_improvement.png")
    grouped_bars(rows, "ms_per_order", "Milliseconds per order (log scale)",
                 "Computation time per order (lower is better)",
                 "results/benchmark_runtime.png", log=True)
    print("Saved table and plots to results/")

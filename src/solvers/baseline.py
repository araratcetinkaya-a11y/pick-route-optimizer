"""Baseline solvers: the routes a smarter method has to beat.

All solvers share the same interface:
    solve(start, items, dist) -> route
- start: (x, y) of the depot
- items: list of (x, y) item locations, in the order the terminal shows them
- dist:  distance function dist(a, b)
- route: the same items, in the order they should be picked
"""
import numpy as np


def solve_terminal_order(start, items, dist):
    """Pick items exactly in the order the terminal lists them."""
    return list(items)


def solve_random(start, items, dist, seed=0):
    """Pick items in a random (but reproducible) order."""
    rng = np.random.default_rng(seed)
    return [items[i] for i in rng.permutation(len(items))]


if __name__ == "__main__":
    from src.orders import generate_orders
    from src.warehouse import Warehouse

    wh = Warehouse()
    orders = generate_orders(wh, n_orders=200, n_items=8)

    for name, solver in [("Terminal order", solve_terminal_order),
                         ("Random order", solve_random)]:
        lengths = []
        for order in orders:
            items = [wh.locations[pid] for pid in order]
            route = solver(wh.depot, items, wh.dist)
            assert sorted(route) == sorted(items)  # every item picked exactly once
            lengths.append(wh.route_length(route))
        print(f"{name:15s} mean length = {np.mean(lengths):.1f}")

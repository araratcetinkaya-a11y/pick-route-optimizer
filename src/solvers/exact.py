"""Exact solver: Held-Karp dynamic programming.

Same interface as every solver:
    solve(start, items, dist) -> route

Guaranteed optimal, but time grows like 2^n * n^2,
so it is only practical for small orders (about 15 items max).
"""


def solve_held_karp(start, items, dist):
    n = len(items)
    if n == 0:
        return []

    # Index 0 = depot, 1..n = items. Precompute all distances once.
    points = [start] + list(items)
    D = [[dist(a, b) for b in points] for a in points]

    # A 'mask' is a set of items stored as bits: bit j = 1 -> item j visited.
    # dp[mask][j] = shortest walk that starts at the depot, visits exactly
    #               the items in `mask`, and ends at item j.
    FULL = 1 << n
    INF = float("inf")
    dp = [[INF] * n for _ in range(FULL)]
    parent = [[-1] * n for _ in range(FULL)]  # previous item, to rebuild the route

    # Base case: depot -> item j directly.
    for j in range(n):
        dp[1 << j][j] = D[0][j + 1]

    # Grow the visited sets one item at a time.
    for mask in range(1, FULL):
        for j in range(n):
            if not (mask & (1 << j)) or dp[mask][j] == INF:
                continue
            for k in range(n):
                if mask & (1 << k):  # k already visited
                    continue
                new_mask = mask | (1 << k)
                cost = dp[mask][j] + D[j + 1][k + 1]
                if cost < dp[new_mask][k]:
                    dp[new_mask][k] = cost
                    parent[new_mask][k] = j

    # All items visited: pick the best last item, including the walk back.
    full = FULL - 1
    last = min(range(n), key=lambda j: dp[full][j] + D[j + 1][0])

    # Walk the parent pointers backwards to rebuild the route.
    order, mask, j = [], full, last
    while j != -1:
        order.append(j)
        prev = parent[mask][j]
        mask ^= 1 << j  # remove j from the set
        j = prev
    order.reverse()
    return [items[j] for j in order]


if __name__ == "__main__":
    import time
    from itertools import permutations

    import numpy as np

    from src.orders import generate_orders
    from src.solvers.heuristics import solve_two_opt
    from src.warehouse import Warehouse

    wh = Warehouse()

    # 1) Correctness: compare with brute force (try every order) on small orders.
    for order in generate_orders(wh, n_orders=20, n_items=6, seed=1):
        items = [wh.locations[pid] for pid in order]
        brute = min(wh.route_length(p) for p in permutations(items))
        assert wh.route_length(solve_held_karp(wh.depot, items, wh.dist)) == brute
    print("Held-Karp matches brute force on 20 orders of 6 items")

    # 2) Quality + speed vs 2-opt.
    for n_items in (8, 12):
        orders = generate_orders(wh, n_orders=50, n_items=n_items)
        for name, solver in [("NN + 2-opt", solve_two_opt),
                             ("Held-Karp", solve_held_karp)]:
            lengths, t0 = [], time.perf_counter()
            for order in orders:
                items = [wh.locations[pid] for pid in order]
                route = solver(wh.depot, items, wh.dist)
                assert sorted(route) == sorted(items)
                lengths.append(wh.route_length(route))
            ms = (time.perf_counter() - t0) / len(orders) * 1000
            print(f"{n_items:2d} items | {name:10s} mean length = "
                  f"{np.mean(lengths):6.1f} | {ms:7.2f} ms/order")

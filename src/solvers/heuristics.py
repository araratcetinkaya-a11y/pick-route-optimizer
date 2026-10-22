"""Classic heuristics: nearest neighbour and 2-opt.

Same interface as every solver:
    solve(start, items, dist) -> route
"""


def solve_nearest_neighbour(start, items, dist):
    """Greedy: always walk to the closest item not picked yet."""
    remaining = list(items)
    route = []
    current = start
    while remaining:
        nearest = min(remaining, key=lambda item: dist(current, item))
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return route


def two_opt(start, route, dist):
    """Improve a route by reversing segments while that makes it shorter.

    Take two edges (a->b) and (c->d). If reconnecting them as (a->c) and
    (b->d) is shorter, reverse the part between b and c. Repeat until no
    swap helps any more (a 'local optimum').
    """
    tour = [start] + list(route) + [start]  # depot fixed at both ends
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[j + 1]
                delta = dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d)
                if delta < 0:  # the swap makes the tour shorter
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour[1:-1]  # strip the depot again


def solve_two_opt(start, items, dist):
    """Nearest neighbour as a starting route, then polish it with 2-opt."""
    return two_opt(start, solve_nearest_neighbour(start, items, dist), dist)


if __name__ == "__main__":
    import numpy as np

    from src.orders import generate_orders
    from src.solvers.baseline import solve_terminal_order
    from src.warehouse import Warehouse

    wh = Warehouse()
    orders = generate_orders(wh, n_orders=200, n_items=8)

    for name, solver in [("Terminal order", solve_terminal_order),
                         ("Nearest neighbour", solve_nearest_neighbour),
                         ("NN + 2-opt", solve_two_opt)]:
        lengths = []
        for order in orders:
            items = [wh.locations[pid] for pid in order]
            route = solver(wh.depot, items, wh.dist)
            assert sorted(route) == sorted(items)  # every item picked exactly once
            lengths.append(wh.route_length(route))
        print(f"{name:18s} mean length = {np.mean(lengths):.1f}")

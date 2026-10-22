"""Tabular Q-learning for one order.

Same interface as every solver:
    solve(start, items, dist) -> route

The agent learns *per order*: for every new order it trains from scratch.
- State  = (where I am, which items I have already picked)
- Action = which unpicked item to walk to next
- Reward = minus the distance walked (shorter route -> higher total reward)
"""
import numpy as np


def train_q_learning(start, items, dist, episodes=10000, alpha=0.5,
                     eps_start=1.0, eps_end=0.05, seed=0):
    n = len(items)
    points = [start] + list(items)            # index 0 = depot, 1..n = items
    D = [[dist(a, b) for b in points] for a in points]
    rng = np.random.default_rng(seed)

    # Q[pos, mask, a] = expected total reward if I am at `pos`, have picked
    # the items in `mask` (same bit trick as Held-Karp), and walk to item a.
    # Starting at 0 is 'optimistic' (real rewards are negative), which
    # pushes the agent to try actions it hasn't tried yet.
    Q = np.zeros((n + 1, 1 << n, n))
    history = []                              # route length of every episode

    for ep in range(episodes):
        # Exploration rate: from mostly random to mostly greedy.
        eps = eps_start + (eps_end - eps_start) * ep / (episodes - 1)
        pos, mask, length = 0, 0, 0

        while mask != (1 << n) - 1:          # until every item is picked
            valid = [a for a in range(n) if not mask & (1 << a)]
            if rng.random() < eps:            # explore: random valid item
                a = rng.choice(valid)
            else:                             # exploit: best known item
                a = max(valid, key=lambda x: Q[pos, mask, x])

            next_pos, next_mask = a + 1, mask | (1 << a)
            reward = -D[pos][next_pos]
            length += D[pos][next_pos]

            if next_mask == (1 << n) - 1:     # last item: walk back to depot
                reward -= D[next_pos][0]
                length += D[next_pos][0]
                future = 0
            else:
                next_valid = [b for b in range(n) if not next_mask & (1 << b)]
                future = max(Q[next_pos, next_mask, b] for b in next_valid)

            # The Q-learning update: nudge the estimate towards
            # (reward now + best value from the next state).
            Q[pos, mask, a] += alpha * (reward + future - Q[pos, mask, a])
            pos, mask = next_pos, next_mask

        history.append(length)
    return Q, history


def greedy_route(Q, items):
    """Follow the learned Q-table: always take the best-valued item."""
    n = len(items)
    pos, mask, route = 0, 0, []
    while mask != (1 << n) - 1:
        valid = [a for a in range(n) if not mask & (1 << a)]
        a = max(valid, key=lambda x: Q[pos, mask, x])
        route.append(items[a])
        pos, mask = a + 1, mask | (1 << a)
    return route


def solve_q_learning(start, items, dist, episodes=10000, seed=0):
    Q, _ = train_q_learning(start, items, dist, episodes=episodes, seed=seed)
    return greedy_route(Q, items)


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    from src.orders import generate_orders
    from src.solvers.exact import solve_held_karp
    from src.solvers.heuristics import solve_two_opt
    from src.warehouse import Warehouse

    wh = Warehouse()

    # 1) Learning curve on one 8-item order.
    items = [wh.locations[pid] for pid in generate_orders(wh, 1, 8)[0]]
    Q, history = train_q_learning(wh.depot, items, wh.dist)
    window = 200
    smooth = np.convolve(history, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4.5))
    plt.plot(history, color="lightgray", lw=0.8, label="Episode route length")
    plt.plot(range(window - 1, len(history)), smooth, color="tab:blue",
             label=f"Moving average ({window} episodes)")
    plt.axhline(wh.route_length(items), color="tab:red", ls="--",
                label="Terminal order")
    plt.axhline(wh.route_length(solve_held_karp(wh.depot, items, wh.dist)),
                color="tab:green", ls="--", label="Optimal (Held-Karp)")
    plt.xlabel("Episode")
    plt.ylabel("Route length")
    plt.title("Q-learning on one 8-item order")
    plt.legend()
    plt.savefig("results/step5_learning_curve.png", dpi=150, bbox_inches="tight")
    print("Saved results/step5_learning_curve.png")

    # 2) Quick comparison on 50 orders of 8 items.
    orders = generate_orders(wh, n_orders=50, n_items=8)
    for name, solver in [("Terminal order", lambda s, i, d: list(i)),
                         ("NN + 2-opt", solve_two_opt),
                         ("Q-learning", solve_q_learning),
                         ("Held-Karp", solve_held_karp)]:
        lengths, t0 = [], time.perf_counter()
        for order in orders:
            items = [wh.locations[pid] for pid in order]
            route = solver(wh.depot, items, wh.dist)
            assert sorted(route) == sorted(items)
            lengths.append(wh.route_length(route))
        ms = (time.perf_counter() - t0) / len(orders) * 1000
        print(f"{name:15s} mean length = {np.mean(lengths):6.1f} | {ms:8.2f} ms/order")
    plt.show()

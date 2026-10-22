"""Before/after picture for the README: one 8-item order,
picked in terminal order vs. the optimal route.

Run from the project root:
    python -m experiments.route_example
"""
import matplotlib.pyplot as plt

from src.orders import generate_orders
from src.solvers.exact import solve_held_karp
from src.visualize import plot_route
from src.warehouse import Warehouse

wh = Warehouse()
items = [wh.locations[pid] for pid in generate_orders(wh, n_orders=1, n_items=8)[0]]

plot_route(wh, items, title="Before: terminal order",
           save_path="results/route_before.png")
plot_route(wh, solve_held_karp(wh.depot, items, wh.dist), title="After: optimal route",
           save_path="results/route_after.png")
print("Saved results/route_before.png and results/route_after.png")
plt.show()

import matplotlib.pyplot as plt


def corridor_path(a, b, H):
    """Points of the walking path from a to b along the corridors."""
    (x1, y1), (x2, y2) = a, b
    if x1 == x2:
        return [a, b]
    y_cross = 0 if (y1 + y2) <= (2 * H - y1 - y2) else H  # bottom or top
    return [a, (x1, y_cross), (x2, y_cross), b]


def plot_route(wh, route, title="", save_path=None):
    """Draw the warehouse and the closed tour depot -> route -> depot."""
    H = wh.aisle_length
    x_max = (wh.n_aisles - 1) * wh.aisle_spacing
    fig, ax = plt.subplots(figsize=(9, 6))

    # walkable corridors: aisles + bottom/top cross aisles
    for a in range(wh.n_aisles):
        x = a * wh.aisle_spacing
        ax.plot([x, x], [0, H], color="lightgray", lw=8, zorder=0)
    for y in (0, H):
        ax.plot([0, x_max], [y, y], color="lightgray", lw=8, zorder=0)

    # the route, drawn along the corridors (no diagonal shortcuts)
    stops = [wh.depot] + list(route) + [wh.depot]
    path = []
    for a, b in zip(stops, stops[1:]):
        path += corridor_path(a, b, H)
    xs, ys = zip(*path)
    ax.plot(xs, ys, color="tab:blue", lw=2, zorder=1)

    # items, numbered by visit order
    for i, (x, y) in enumerate(route, start=1):
        ax.scatter(x, y, color="tab:orange", s=150, zorder=2)
        ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=8, zorder=3)
    ax.scatter(*wh.depot, marker="s", color="tab:red", s=150, zorder=2, label="Depot")

    ax.set_title(f"{title}  (length = {wh.route_length(route)})")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


if __name__ == "__main__":
    from src.orders import generate_orders
    from src.warehouse import Warehouse

    wh = Warehouse()
    order = generate_orders(wh, n_orders=1, n_items=8)[0]
    route = [wh.locations[pid] for pid in order]
    plot_route(wh, route, title="Terminal order", save_path="results/step2_demo.png")
    plt.show()

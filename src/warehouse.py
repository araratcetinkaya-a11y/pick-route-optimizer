from dataclasses import dataclass

import numpy as np


@dataclass
class Warehouse:
    """Rectangular warehouse: parallel vertical aisles + one cross aisle at the
    bottom (y=0) and one at the top (y=aisle_length)."""
    n_aisles: int = 10
    aisle_length: int = 20     # H
    aisle_spacing: int = 3     # walking distance between neighbouring aisles
    seed: int = 0

    def __post_init__(self):
        self.depot = (0, 0)  # packing station: bottom-left corner
        slots = [(a * self.aisle_spacing, y)
                 for a in range(self.n_aisles)
                 for y in range(1, self.aisle_length)]
        # Product IDs are assigned to slots randomly (with a seed),
        # so sorting by ID says nothing about location -> like the terminal.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(slots))
        self.locations = {pid: slots[i] for pid, i in enumerate(perm)}

    @property
    def n_products(self):
        return len(self.locations)

    def dist(self, a, b):
        """Walking distance along the corridors (not straight-line)."""
        (x1, y1), (x2, y2) = a, b
        if x1 == x2:                      # same aisle
            return abs(y1 - y2)
        H = self.aisle_length             # different aisles: via bottom or top
        return abs(x1 - x2) + min(y1 + y2, 2 * H - y1 - y2)

    def route_length(self, route):
        """Length of the closed tour depot -> route... -> depot.
        `route` is a list of (x, y) locations."""
        stops = [self.depot] + list(route) + [self.depot]
        return sum(self.dist(a, b) for a, b in zip(stops, stops[1:]))

    def distance_matrix(self, points):
        """Pairwise distances, used by the solvers later."""
        n = len(points)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = self.dist(points[i], points[j])
        return D

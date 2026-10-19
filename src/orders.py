import numpy as np


def generate_orders(warehouse, n_orders, n_items, seed=42):
    """Reproducible random orders. Each order = list of distinct product IDs,
    in 'terminal order' (the order they arrive in)."""
    rng = np.random.default_rng(seed)
    return [
        rng.choice(warehouse.n_products, size=n_items, replace=False).tolist()
        for _ in range(n_orders)
    ]

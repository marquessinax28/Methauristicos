def init_lhs(n, bounds, dim=2):
    lo, hi = bounds
    pts = np.empty((n, dim))
    for d in range(dim):
        intervals = np.linspace(lo, hi, n + 1)
        rnds = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(rnds)
        pts[:, d] = rnds
    return pts
def init_maxdistance(n, bounds, dim=2, num_candidates=50):
    lo, hi = bounds
    pts = [np.random.uniform(lo, hi, dim)]
    for _ in range(1, n):
        cands = np.random.uniform(lo, hi, (num_candidates, dim))
        dists = np.array([min(np.linalg.norm(c - p) for p in pts) for c in cands])
        pts.append(cands[np.argmax(dists)])
    return np.array(pts)
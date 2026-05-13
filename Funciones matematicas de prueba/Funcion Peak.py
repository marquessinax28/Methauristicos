def peak(x, y):
    return 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x+1)**2 - y**2)
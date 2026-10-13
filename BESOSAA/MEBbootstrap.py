import numpy as np

def meboot(x, reps=999, trim=0.10, expand_sd=True, rng=None):
    """
    Maximum Entropy Bootstrap for a univariate time series.

    Parameters
    ----------
    x : array-like, shape (n,)
        Original time series, ordered in time.
    reps : int
        Number of bootstrap replicates.
    trim : float
        Trimming proportion used to set tail limits.
    expand_sd : bool
        Apply the usual variance-expansion adjustment.
    rng : np.random.Generator, optional

    Returns
    -------
    ensemble : ndarray, shape (n, reps)
        Each column is one bootstrapped time series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = rng or np.random.default_rng()

    order = np.argsort(x)
    x_sorted = x[order]

    # Intermediate points between sorted observations
    z = (x_sorted[:-1] + x_sorted[1:]) / 2.0

    # Tail limits based on trimmed mean of successive differences
    dv = np.diff(x_sorted)
    dv_trim = np.mean(np.sort(dv)[int(trim * (n - 1)): int((1 - trim) * (n - 1))])
    xmin = x_sorted[0] - dv_trim
    xmax = x_sorted[-1] + dv_trim

    # Interval means for the ME density
    m = np.empty(n)
    m[0] = 0.75 * x_sorted[0] + 0.25 * x_sorted[1]
    m[-1] = 0.25 * x_sorted[-2] + 0.75 * x_sorted[-1]
    if n > 2:
        m[1:-1] = (
            0.25 * x_sorted[:-2]
            + 0.50 * x_sorted[1:-1]
            + 0.25 * x_sorted[2:]
        )

    # Class limits: xmin, z_1,...,z_{n-1}, xmax
    limits = np.concatenate(([xmin], z, [xmax]))

    ensemble = np.empty((n, reps))
    for b in range(reps):
        u = rng.uniform(0, 1, size=n)

        # ME density: piecewise uniform on each interval
        widths = np.diff(limits)
        probs = widths / widths.sum()
        cum = np.cumsum(probs)

        # Quantiles of the piecewise-uniform ME density
        q = np.empty(n)
        interval = np.searchsorted(cum, u, side="left")
        interval = np.clip(interval, 0, n - 1)

        left = limits[:-1][interval]
        right = limits[1:][interval]
        cum_left = np.concatenate(([0.0], cum[:-1]))[interval]
        p_int = probs[interval]

        q = left + (u - cum_left) / p_int * (right - left)
        q = np.sort(q)

        # Restore the original temporal rank pattern
        y = np.empty(n)
        y[order] = q
        ensemble[:, b] = y

    if expand_sd:
        original_sd = x.std(ddof=1)
        boot_sd = ensemble.std(axis=0, ddof=1)
        scale = np.divide(
            original_sd, boot_sd,
            out=np.ones_like(boot_sd),
            where=boot_sd > 0,
        )
        ensemble = x.mean() + (ensemble - ensemble.mean(axis=0)) * scale

    return ensemble

if __name__ == "__main__":
   import pandas as pd

   # y: your stationary time series as a 1-D NumPy array or pandas Series
   y = np.loadtxt(f"transpCosts75.csv", delimiter=",")

   rng = np.random.default_rng(123)
   B = meboot(y, reps=125, trim=0.10, rng=rng)

   def statistic(series):
      return series.mean()  # replace with your estimator


   theta_hat = statistic(y)
   theta_star = np.array([statistic(B[:, b]) for b in range(B.shape[1])])

   # Percentile confidence interval
   alpha = 0.05
   ci = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
   print("point estimate:", theta_hat)
   print("95% ME-bootstrap percentile CI:", ci)

   # Bootstrap standard error
   se = theta_star.std(ddof=1)
   print("bootstrap SE:", se)
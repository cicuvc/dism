import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LN2 = np.log(2.0)

# -----------------------------
# Float32 helpers + FTZ
# -----------------------------
def f32(x):
    return np.asarray(x, dtype=np.float32)

def is_subnormal_f32(x_f32):
    x = np.abs(x_f32.astype(np.float32))
    tiny_normal = np.float32(2.0) ** np.float32(-126.0)
    return (x > 0.0) & (x < tiny_normal)

def ftz_f32(x_f32):
    x = x_f32.astype(np.float32)
    x[is_subnormal_f32(x)] = np.float32(0.0)
    return x

def exp2_mufu_model(x_f32, apply_ftz=True):
    """
    Practical MUFU.EX2.approx.ftz.rn-ish model:
      - interpret input as float32
      - compute exp2 in float64 then round to float32
      - apply FTZ on output
    """
    x = f32(x_f32)
    y = np.exp(LN2 * x.astype(np.float64)).astype(np.float32)
    if apply_ftz:
        y = ftz_f32(y)
    return y

# -----------------------------
# Math helpers (float64 "truth")
# -----------------------------
def log2_f64(x):
    return np.log(x) / LN2

def exp2_f64(x):
    return np.exp(LN2 * x)

# -----------------------------
# Exact op and approx op
# -----------------------------
def exact_op(x, y):
    """
    exact: -log2(exp2(-x)+exp2(-y)) in float64, stable form:
      out = min(x,y) - log2(1 + 2^{-abs(x-y)})
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mi = np.minimum(x, y)
    d = np.abs(x - y)
    return mi - log2_f64(1.0 + exp2_f64(-d))

def f_target_f64(t):
    # f(t)=log2(log2(1+2^t)) - t on t<=0
    return log2_f64(log2_f64(1.0 + exp2_f64(t))) - t

def poly_no_c0_eval_f32(coeffs, p_f32):
    """
    poly(p) = sum_{k=1..deg} c_k * p^k, with c0=0.
    Evaluate in float32 using Horner:
      poly = p*(c1 + p*(c2 + p*(... + p*c_deg)))
    coeffs: array-like length=deg, representing [c1,c2,...,c_deg]
    """
    p = f32(p_f32)
    deg = len(coeffs)
    # Start with c_deg
    t = np.full_like(p, np.float32(coeffs[-1]), dtype=np.float32)
    # accumulate downwards: t = p*t + c_k
    for k in range(deg - 2, -1, -1):
        t = f32(p * t + np.float32(coeffs[k]))
    # multiply by p for c0=0 constraint
    t = f32(p * t)
    return t

def approx_op(x, y, LIMIT, coeffs):
    """
    Approx implementation (float32-ish):
      nabs=-abs(x-y), mi=min(x,y), p=max(nabs, LIMIT)
      logs = nabs + poly(p)
      res = ex2.approx.ftz.rn(logs) (modeled)
      out = mi - res
    """
    x_f = f32(x)
    y_f = f32(y)

    nabs = f32(-np.abs((x_f - y_f).astype(np.float32)))
    mi = f32(np.minimum(x_f, y_f))
    p = f32(np.maximum(nabs, np.float32(LIMIT)))

    poly = poly_no_c0_eval_f32(coeffs, p)
    logs = f32(nabs + poly)

    res = exp2_mufu_model(logs, apply_ftz=True)
    out = f32(mi - res)
    return out.astype(np.float64)

# -----------------------------
# Generic Remez for c0=0 polynomial of degree=deg
# -----------------------------
def remez_poly_c0zero(a, b, deg, max_iter=80, grid=30001, tol=1e-12):
    """
    Minimax Remez for polynomial:
      p(t) = sum_{k=1..deg} c_k t^k   (c0 fixed to 0)
    Unknowns: deg coefficients + E => deg+1 unknowns
    Use (deg+1) alternation points.
    """
    m = deg + 1  # number of equations / alternation points
    x = np.linspace(a, b, m)

    gx = np.linspace(a, b, grid)
    fy = f_target_f64(gx)

    for _ in range(max_iter):
        s = np.array([(-1.0) ** i for i in range(m)], dtype=np.float64)

        # Build linear system: sum_{k=1..deg} c_k x^k + s_i*E = f(x_i)
        A = np.zeros((m, m), dtype=np.float64)  # columns: c1..c_deg, E
        for k in range(1, deg + 1):
            A[:, k - 1] = x ** k
        A[:, deg] = s

        rhs = f_target_f64(x)
        sol = np.linalg.solve(A, rhs)
        coeffs = sol[:deg]  # c1..c_deg

        # error on grid
        # evaluate poly in float64 for extremal search stability
        p = np.zeros_like(gx)
        # Horner in float64: p = gx*(c1 + gx*(c2 + ...))
        t = np.full_like(gx, coeffs[-1], dtype=np.float64)
        for k in range(deg - 2, -1, -1):
            t = gx * t + coeffs[k]
        p = gx * t

        err = fy - p
        abs_err = np.abs(err)

        # peak picking (discrete)
        peaks = []
        for i in range(1, grid - 1):
            if abs_err[i] >= abs_err[i - 1] and abs_err[i] >= abs_err[i + 1]:
                peaks.append(i)
        if len(peaks) < m:
            peaks = np.argsort(abs_err)[-m:].tolist()

        # choose m peaks with sign alternation, preferring large error
        peaks = sorted(peaks, key=lambda i: abs_err[i], reverse=True)

        chosen = []
        chosen_signs = []
        for idx in peaks:
            si = np.sign(err[idx]) if err[idx] != 0 else 0.0
            if len(chosen) == 0:
                chosen.append(idx)
                chosen_signs.append(si)
            else:
                if si == 0 or chosen_signs[-1] == 0 or si * chosen_signs[-1] < 0:
                    chosen.append(idx)
                    chosen_signs.append(si)
            if len(chosen) == m:
                break

        if len(chosen) < m:
            chosen = peaks[:m]

        new_x = np.sort(gx[chosen])

        if np.max(np.abs(new_x - x)) < tol:
            return coeffs.astype(np.float64)

        x = new_x

    return coeffs.astype(np.float64)

def max_abs_error_full(LIMIT, coeffs, d_max=50.0, n=250000):
    """
    Evaluate max abs error of the log2(1+2^-d) term over d in [0,d_max]:
      true(d)   = log2(1+2^-d)
      approx(d) = exp2( -d + poly(max(-d,LIMIT)) )
    with poly evaluated in float32 and exp2 modeled with FTZ.
    """
    d = np.linspace(0.0, d_max, n, dtype=np.float64)
    nabs = -d
    p = np.maximum(nabs, LIMIT)

    poly_f32 = poly_no_c0_eval_f32(coeffs, f32(p))
    logs_f32 = f32(f32(nabs) + poly_f32)

    approx = exp2_mufu_model(logs_f32).astype(np.float64)
    true = log2_f64(1.0 + exp2_f64(-d))
    return float(np.max(np.abs(approx - true)))

def optimize_limit(deg, limit_lo=-12.0, limit_hi=-1.0, coarse_step=0.25, refine_rounds=8):
    # coarse scan
    candidates = np.arange(limit_lo, limit_hi + 1e-12, coarse_step)
    best = None

    for L in candidates:
        coeffs = remez_poly_c0zero(L, 0.0, deg=deg)
        e = max_abs_error_full(L, coeffs)
        item = (float(L), float(e), coeffs)
        if best is None or item[1] < best[1]:
            best = item

    bestL, bestE, bestC = best

    # local refinement
    left = max(limit_lo, bestL - coarse_step)
    right = min(limit_hi, bestL + coarse_step)

    for _ in range(refine_rounds):
        xs = np.linspace(left, right, 9)
        local_best = None
        for L in xs:
            coeffs = remez_poly_c0zero(L, 0.0, deg=deg)
            e = max_abs_error_full(L, coeffs)
            item = (float(L), float(e), coeffs)
            if local_best is None or item[1] < local_best[1]:
                local_best = item

        bestL, bestE, bestC = local_best

        span = (right - left) * 0.35
        left = max(limit_lo, bestL - span)
        right = min(limit_hi, bestL + span)

    return bestL, bestE, bestC

# -----------------------------
# Plotting
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=3, help="Polynomial degree (c0 fixed to 0).")
    ap.add_argument("--grid", type=int, default=401, help="Heatmap grid size per axis (e.g., 401).")
    ap.add_argument("--rel_eps", type=float, default=1e-6, help="Denominator floor for relative error.")
    ap.add_argument("--min_abs", type=float, default=1e-9, help="Clip floor for abs error before LogNorm.")
    ap.add_argument("--min_rel", type=float, default=1e-9, help="Clip floor for rel error before LogNorm.")
    args = ap.parse_args()

    deg = args.degree
    if deg < 1:
        raise ValueError("degree must be >= 1")

    LIMIT, err_bound, coeffs = optimize_limit(deg=deg)

    print("=== Optimized parameters ===")
    print(f"degree = {deg}  (poly(p)=sum_k=1..deg c_k p^k, c0=0)")
    print(f"LIMIT = {LIMIT:.12f}")
    print(f"estimated max|abs error| of log2(1+2^-d) term (d in [0,50]) = {err_bound:.3e}")
    for i, c in enumerate(coeffs, start=1):
        print(f"c{i} = {c: .18e}")

    # Heatmap grid
    n = args.grid
    xs = np.linspace(-10.0, 10.0, n, dtype=np.float64)
    ys = np.linspace(-10.0, 10.0, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    exact = exact_op(X, Y)
    approx = approx_op(X, Y, LIMIT, coeffs)

    abs_err = np.abs(approx - exact)
    denom = np.maximum(np.abs(exact), args.rel_eps)
    rel_err = abs_err / denom

    print(f"heatmap max|abs error| on [-10,10]^2 = {abs_err.max():.3e}")
    print(f"heatmap max relative error on [-10,10]^2 (rel_eps={args.rel_eps:g}) = {rel_err.max():.3e}")

    abs_plot = np.clip(abs_err, args.min_abs, None)
    rel_plot = np.clip(rel_err, args.min_rel, None)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)

    im0 = axes[0].imshow(
        abs_plot,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis",
        norm=LogNorm(vmin=abs_plot.min(), vmax=abs_plot.max()),
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title(f"|approx-exact| (log scale)\ndeg={deg}, LIMIT={LIMIT:.4f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label("absolute error")

    im1 = axes[1].imshow(
        rel_plot,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="magma",
        norm=LogNorm(vmin=rel_plot.min(), vmax=rel_plot.max()),
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(f"relative error |err|/max(|exact|,{args.rel_eps:g}) (log scale)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label("relative error")

    plt.show()

if __name__ == "__main__":
    main()
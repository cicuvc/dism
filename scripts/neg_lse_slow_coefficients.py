"""
Author: Copilot GPT5.2

This script searches optimal cut-off LIMIT and fit coefficients for neg_lse(x,y).

__forceinline__ __device__ static float op(float x, float y) {
    // This approximately calculates -log2(exp2(-x)+exp2(-y)) safely.
    // out = -log2(1+exp2(-|x-y|)) + min(x,y)
    // Trivial approach: use MUFU to calculate exp2(-|x-y|), and 3 order polynomial to fit -log2(1+x) in (0,1]
    // Here: let -log2(1+exp2(x)) = -exp2(f(max(LIMIT, x)) + x), so f(x) = log2(log2(1+exp2(max(LIMIT, x)))) - x
    // and fit log2(log2(1+exp2(t))) in (LIMIT, 0]. Fit error of polynomial and cut-off error can be reduced by 
    // the small derivative of exp2 at negative interval.

    // SASS code:
    // FADD DIFF, X, -Y
    // FMNMX.MIN MI, X, Y
    // FMNMX.MAX P, -|DIFF|, -5.f
    // FFMA X1, P, -0.004380030100270f, -0.056839571752722f
    // FFMA X2, P, X1, -0.278691685310180f
    // FFMA LOGS, P, X2, -|DIFF|
    // MUFU.EX2 RES, LOGS
    // FADD OUT, MI, -RES
    float nabs = -abs(x - y), mi = min(x, y); 
    float p = max(nabs, -4.f); 
    float logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n":"=f"(res): "f"(logs));
    return mi - res;
}

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LN2 = np.log(2.0)

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
    # Practical MUFU.EX2.approx.ftz.rn-ish model:
    # exp2 computed in float64 then rounded to float32; FTZ on output.
    x = f32(x_f32)
    y = np.exp(LN2 * x.astype(np.float64)).astype(np.float32)
    if apply_ftz:
        y = ftz_f32(y)
    return y

def log2_f64(x):
    return np.log(x) / LN2

def exp2_f64(x):
    return np.exp(LN2 * x)

def exact_op(x, y):
    """
    exact: -log2(exp2(-x)+exp2(-y)) in float64, stable form
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mi = np.minimum(x, y)
    d = np.abs(x - y)
    return mi - log2_f64(1.0 + exp2_f64(-d))

def poly3_no_c0_eval_f32(c123, p_f32):
    """
    poly(p) = c1*p + c2*p^2 + c3*p^3
    float32 Horner: p*(c1 + p*(c2 + p*c3))
    """
    c1, c2, c3 = [np.float32(v) for v in c123]
    p = f32(p_f32)

    t = np.full_like(p, c3, dtype=np.float32)
    t = f32(p * t + c2)
    t = f32(p * t + c1)
    t = f32(p * t)
    return t

def approx_op(x, y, LIMIT, c123):
    """
    Approx implementation (float32-ish) mimicking your kernel:
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

    poly = poly3_no_c0_eval_f32(c123, p)
    logs = f32(nabs + poly)

    res = exp2_mufu_model(logs, apply_ftz=True)
    out = f32(mi - res)
    return out.astype(np.float64)

# ---------- fit machinery (same idea as before) ----------

def f_target_f64(t):
    # f(t)=log2(log2(1+2^t)) - t
    return log2_f64(log2_f64(1.0 + exp2_f64(t))) - t

def remez_cubic_c0zero(a, b, max_iter=60, grid=20001, tol=1e-12):
    """
    Remez minimax on [a,b] for p(t)=c1*t+c2*t^2+c3*t^3 (c0=0).
    Unknowns: c1,c2,c3,E => 4 alternation points.
    """
    x = np.linspace(a, b, 4)
    gx = np.linspace(a, b, grid)
    fy = f_target_f64(gx)

    for _ in range(max_iter):
        s = np.array([(-1.0) ** i for i in range(4)], dtype=np.float64)

        A = np.zeros((4, 4), dtype=np.float64)
        A[:, 0] = x
        A[:, 1] = x**2
        A[:, 2] = x**3
        A[:, 3] = s

        rhs = f_target_f64(x)
        sol = np.linalg.solve(A, rhs)
        c1, c2, c3, E = sol

        p = c1 * gx + c2 * gx**2 + c3 * gx**3
        err = fy - p
        abs_err = np.abs(err)

        # discrete peak picking
        peaks = []
        for i in range(1, grid - 1):
            if abs_err[i] >= abs_err[i - 1] and abs_err[i] >= abs_err[i + 1]:
                peaks.append(i)
        if len(peaks) < 4:
            peaks = np.argsort(abs_err)[-4:].tolist()

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
            if len(chosen) == 4:
                break

        if len(chosen) < 4:
            chosen = peaks[:4]

        new_x = np.sort(gx[chosen])
        if np.max(np.abs(new_x - x)) < tol:
            return (c1, c2, c3)
        x = new_x

    return (c1, c2, c3)

def max_abs_error_full(LIMIT, c123, d_max=50.0, n=200000):
    d = np.linspace(0.0, d_max, n, dtype=np.float64)
    nabs = -d
    p = np.maximum(nabs, LIMIT)

    poly_f32 = poly3_no_c0_eval_f32(c123, f32(p))
    logs_f32 = f32(f32(nabs) + poly_f32)
    approx = exp2_mufu_model(logs_f32).astype(np.float64)

    true = log2_f64(1.0 + exp2_f64(-d))
    return float(np.max(np.abs(approx - true)))

def optimize_limit(limit_lo=-12.0, limit_hi=-1.0, coarse_step=0.25, refine_rounds=8):
    candidates = np.arange(limit_lo, limit_hi + 1e-12, coarse_step)
    best = None

    for L in candidates:
        c123 = remez_cubic_c0zero(L, 0.0)
        e = max_abs_error_full(L, c123)
        item = (float(L), float(e), c123)
        if best is None or item[1] < best[1]:
            best = item

    bestL, bestE, bestC = best
    left = max(limit_lo, bestL - coarse_step)
    right = min(limit_hi, bestL + coarse_step)

    for _ in range(refine_rounds):
        xs = np.linspace(left, right, 9)
        local_best = None
        for L in xs:
            c123 = remez_cubic_c0zero(L, 0.0)
            e = max_abs_error_full(L, c123)
            item = (float(L), float(e), c123)
            if local_best is None or item[1] < local_best[1]:
                local_best = item

        bestL, bestE, bestC = local_best
        span = (right - left) * 0.35
        left = max(limit_lo, bestL - span)
        right = min(limit_hi, bestL + span)

    return bestL, bestE, bestC

# ---------- plotting ----------

def main():
    LIMIT, err_bound, c123 = optimize_limit()
    c1, c2, c3 = c123

    print("=== Optimized parameters ===")
    print(f"LIMIT = {LIMIT:.12f}")
    print(f"estimated max|abs error| of log2(1+2^-d) term (d in [0,50]) = {err_bound:.3e}")
    print("poly(p)=c1*p+c2*p^2+c3*p^3 (c0=0)")
    print(f"c1 = {c1: .18e}")
    print(f"c2 = {c2: .18e}")
    print(f"c3 = {c3: .18e}")

    # grid
    n = 401  # increase for higher resolution
    xs = np.linspace(-10.0, 10.0, n, dtype=np.float64)
    ys = np.linspace(-10.0, 10.0, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    exact = exact_op(X, Y)
    approx = approx_op(X, Y, LIMIT, c123)

    abs_err = np.abs(approx - exact)

    # relative error: |err| / max(|exact|, rel_eps)
    rel_eps = 1e-6  # avoid blow-up where exact ~ 0; tune to taste
    denom = np.maximum(np.abs(exact), rel_eps)
    rel_err = abs_err / denom

    print(f"heatmap max|abs error| on [-10,10]^2 = {abs_err.max():.3e}")
    print(f"heatmap max relative error on [-10,10]^2 (with rel_eps={rel_eps:g}) = {rel_err.max():.3e}")

    # Log color scale needs strictly positive
    min_abs = 1e-9
    min_rel = 1e-9
    abs_plot = np.clip(abs_err, min_abs, None)
    rel_plot = np.clip(rel_err, min_rel, None)

    abs_vmin, abs_vmax = abs_plot.min(), abs_plot.max()
    rel_vmin, rel_vmax = rel_plot.min(), rel_plot.max()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)

    im0 = axes[0].imshow(
        abs_plot,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis",
        norm=LogNorm(vmin=abs_vmin, vmax=abs_vmax),
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title(f"Absolute error |approx-exact| (log scale)\nLIMIT={LIMIT:.4f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("|error|")

    im1 = axes[1].imshow(
        rel_plot,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="magma",
        norm=LogNorm(vmin=rel_vmin, vmax=rel_vmax),
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(f"Relative error |approx-exact|/max(|exact|,{rel_eps:g}) (log scale)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("relative error")

    plt.show()

if __name__ == "__main__":
    main()
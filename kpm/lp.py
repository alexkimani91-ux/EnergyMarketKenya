"""Light-weight sparse LP/MILP builder on top of SciPy's HiGHS interface.

Variables and constraints are created in vectorised blocks, so models with
hundreds of thousands of variables build in well under a second.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

INF = np.inf


class LPBuilder:
    def __init__(self):
        self.n = 0
        self._lb, self._ub, self._c, self._int = [], [], [], []
        self.m = 0
        self._sense, self._rhs = [], []
        self._r, self._k, self._v = [], [], []
        self.obj_const = 0.0

    # ------------------------------------------------------------------ vars
    def var(self, shape, lb=0.0, ub=INF, cost=0.0, integer=False):
        shape = (shape,) if np.isscalar(shape) else tuple(shape)
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self._lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel().copy())
        self._ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel().copy())
        self._c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel().copy())
        self._int.append(np.full(size, 1 if integer else 0, dtype=np.uint8))
        return idx

    def add_cost(self, idx, cost):
        """Add to the objective coefficient of existing variables."""
        c = np.concatenate(self._c) if len(self._c) > 1 else self._c[0]
        np.add.at(c, np.asarray(idx).ravel(), np.broadcast_to(cost, np.shape(idx)).ravel())
        self._c = [c]

    # ----------------------------------------------------------- constraints
    def rows(self, shape, sense, rhs=0.0):
        """Create a block of rows. sense in {'=', '<', '>'}."""
        shape = (shape,) if np.isscalar(shape) else tuple(shape)
        size = int(np.prod(shape))
        idx = np.arange(self.m, self.m + size).reshape(shape)
        self.m += size
        self._sense.append(np.full(size, {"=": 0, "<": 1, ">": 2}[sense], dtype=np.int8))
        self._rhs.append(np.broadcast_to(np.asarray(rhs, float), shape).ravel().copy())
        return idx

    def terms(self, rows, cols, vals=1.0):
        rows, cols = np.broadcast_arrays(np.asarray(rows), np.asarray(cols))
        vals = np.broadcast_to(np.asarray(vals, float), rows.shape)
        self._r.append(rows.ravel())
        self._k.append(cols.ravel())
        self._v.append(vals.ravel().astype(float))

    # ----------------------------------------------------------------- solve
    def _assemble(self):
        lb, ub = np.concatenate(self._lb), np.concatenate(self._ub)
        c = np.concatenate(self._c)
        integ = np.concatenate(self._int)
        sense = np.concatenate(self._sense) if self._sense else np.zeros(0, np.int8)
        rhs = np.concatenate(self._rhs) if self._rhs else np.zeros(0)
        A = sp.coo_matrix((np.concatenate(self._v), (np.concatenate(self._r), np.concatenate(self._k))),
                          shape=(self.m, self.n)).tocsr()
        return c, lb, ub, integ, sense, rhs, A

    def solve(self, mip=False, time_limit=None, mip_gap=1e-3, verbose=False):
        c, lb, ub, integ, sense, rhs, A = self._assemble()
        if mip and integ.any():
            lo = np.where(sense == 1, -INF, rhs)
            hi = np.where(sense == 2, INF, rhs)
            opts = {"disp": verbose, "mip_rel_gap": mip_gap}
            if time_limit:
                opts["time_limit"] = time_limit
            r = milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lo, hi), options=opts)
            if r.x is None:
                raise RuntimeError(f"MILP failed: {r.message}")
            # fix integers and re-solve as LP to obtain prices (duals)
            xi = np.round(r.x[integ == 1])
            lb, ub = lb.copy(), ub.copy()
            lb[integ == 1] = xi
            ub[integ == 1] = xi
        eq, le, ge = sense == 0, sense == 1, sense == 2
        A_eq, b_eq = A[eq], rhs[eq]
        A_ub = sp.vstack([A[le], -A[ge]]).tocsr()
        b_ub = np.concatenate([rhs[le], -rhs[ge]])
        opts = {"disp": verbose, "presolve": True}
        if time_limit:
            opts["time_limit"] = time_limit
        res = linprog(c, A_ub=A_ub if A_ub.shape[0] else None, b_ub=b_ub if A_ub.shape[0] else None,
                      A_eq=A_eq if A_eq.shape[0] else None, b_eq=b_eq if A_eq.shape[0] else None,
                      bounds=np.column_stack([lb, ub]),
                      method="highs", options=opts)
        if res.status != 0:
            raise RuntimeError(f"LP failed (status {res.status}): {res.message}")
        dual = np.zeros(self.m)
        if eq.any():
            dual[eq] = res.eqlin.marginals
        nle = int(le.sum())
        if A_ub.shape[0]:
            dual[le] = res.ineqlin.marginals[:nle]
            dual[ge] = -res.ineqlin.marginals[nle:]
        return Solution(res.x, dual, res.fun + self.obj_const)


class Solution:
    def __init__(self, x, dual, obj):
        self.x, self.dual, self.obj = x, dual, obj

    def val(self, idx):
        return self.x[np.asarray(idx)]

    def mu(self, rows):
        return self.dual[np.asarray(rows)]

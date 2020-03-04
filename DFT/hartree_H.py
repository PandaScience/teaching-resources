#!/usr/bin/env python3
# coding: utf-8
# vim: set ai ts=3 sw=4 sts=0 noet pi ci

# Copyright (c) 2018-2020 René Wirnata
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without liitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Example implementation of DFT Assignment 5.1: Hartree energy for H-atom GS

This script uses the last assignment's code to determine a solution of the
radial Schrödinger equation for the hydrogen ground state (n=1, l=0). After
normalizing, the Hartree potential energy w(r) = r*vh(r) is computed in a
second "integration" step and numerically integrated to the Hartree energy
(~0.3125 Ha). For hydrogen, the homogeneous solution w_hom(r) = beta * r
is not required in order to match the boundary condition (--> beta = 0).

Note, that the integration limits (tmin, tmax) and step size (h) need to be
identical for solve_rseq() and solve_poisson() or you must use interpolated
versions of the functions w(r) and u(r) when computing the Hartree energy.
Further, tmin for solve_poisson() should not be smaller than tmin for
solve_rseq(), because extrapolating u(r) beyond the computed data points may
result in errors.
"""
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import newton
# when using UnivariateSpline you MUST set s=0; interp1d also does the job
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import matplotlib.pyplot as plt
import numpy as np


# we use some global values
h = 0.02
tmin = 1e-5
tmax = 20
nsteps = round((tmax - tmin) / h)


def rseq(t, y, en):
    """Radial Schrödinger equation with l=0 as system of 1st order ODEs."""
    vext = -1 / t
    y1, y2 = y
    y1d = y2
    f = -2 * (en - vext)
    y2d = f * y1
    return [y1d, y2d]


def solve_rseq(en):
    """Wrapper for integrating the radial SEQ inwards."""
    sol = solve_ivp(
        # function signature must be f(t, y), so use lambda function
        lambda t, y: rseq(t, y, en),
        [tmax, tmin],
        # use values from exact radial function for hydrogen:
        # u(r) = 2r exp(-r), u'(r) = 2(1-r) exp(-r)
        [2 * tmax * np.exp(-tmax), 2 * (1 - tmax) * np.exp(-tmax)],
        t_eval=np.linspace(tmax, tmin, nsteps),
    )
    # reverse output such that t0 < ... < tN with corresponding y0 ... yN
    return sol.y[0, ::-1], sol.t[::-1]


def poisson(t, y, u):
    """Poisson's equation as system of 1st order ODEs."""
    y1, y2 = y
    y1d = y2
    y2d = -u(t) ** 2 / t
    return [y1d, y2d]


def solve_poisson(u):
    """Wrapper for integrating Poisson's equation outwards."""
    # integrate forward from 0+eps to rmax with y(t) = w(r) = r * v_H(r)
    sol = solve_ivp(
        lambda t, y: poisson(t, y, u),
        [tmin, tmax],
        # use initial value w(0)=0 and w'(r) = (2r + 1) * e^(-2r) --> w'(0)=1
        [0, 1],
        t_eval=np.linspace(tmin, tmax, nsteps),
    )
    return sol.y[0, :], sol.t


def main():
    print("\n/-----------------------------------------------\\")
    print("| Ass.5.1: Hartree energy for H-atom GS density |")
    print("\\-----------------------------------------------/\n")

    print("[INFO] using h = {} for numerically solving ODEs".format(h))

    # use secant method for finding the root in u_0(E)
    # integration returns (u(r_i), r_i) --> u(r_0) = solver(E)[0][0]
    root = newton(lambda en0: solve_rseq(en0)[0][0], -2.0)
    print("[INFO] found energy eigenvalue @ E = {:.5f} Ha".format(root))
    # integrate again using the correct energy
    u, r = solve_rseq(root)

    # normalize u**2 to 1
    norm = trapz(u ** 2, r)
    u /= np.sqrt(norm)
    print("[INFO] normalizing |u(r)|^2 from {:.5f} to 1".format(norm))
    # interpolate normalized u(r) using B-splines
    u_spl = Spline(r, u)

    # determine w(r) via Poisson's equation for single-orbital density n_s(r)
    w, r = solve_poisson(u_spl)
    # find hom.sol. w_hom(r) = a*r to match BC w(r_max) = q_tot = 1 for H-atom
    beta = (1 - w[-1]) / r[-1]
    w += beta * r
    print("[INFO] adding w_hom(r) = b * r  -->  b = {:.4f}".format(beta))
    w_spl = Spline(r, w)
    v_spl = Spline(r, w / r)

    # compute Hartree energy (verify integration interval choice in plot later)
    eh = 0.5 * trapz(w_spl(r) / r * u_spl(r) ** 2, r)
    print("[INFO] Hartree energy: {:.5f} Ha".format(eh))

    # compare numerical vs. exact Hartree potential energy function
    w_exact = lambda r: -(r + 1) * np.exp(-2 * r) + 1  # noqa
    v_exact = lambda r: w_exact(r) / r  # noqa
    plt.plot(r, w_spl(r), lw=2, ls="--", label=r"$w(r)$ num")
    plt.plot(r, v_spl(r), lw=2, ls="--", label=r"$v(r)$ num")
    plt.plot(r, w_exact(r), lw=3, alpha=0.5, label=r"$w(r)$ exact")
    plt.plot(r, v_exact(r), lw=3, alpha=0.5, label=r"$v(r)$ exact")
    # integrand from Hartree energy --> integrating up to r=10 is sufficient
    label = r"$v(r) \, |u(r)|^2$"
    plt.plot(r, v_spl(r) * u_spl(r) ** 2, alpha=0.5, lw=3, label=label)
    plt.xlabel("r in Bohr")
    plt.legend(loc="best", fancybox=True, shadow=True)
    plt.show()


if __name__ == "__main__":
    main()

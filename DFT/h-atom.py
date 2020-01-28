#!/usr/bin/env python3
# coding: utf-8
# vim: set ai ts=3 sw=4 sts=0 noet pi ci
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import newton
import matplotlib.pyplot as plt
import numpy as np


def u2ex(r):
    """Normalized exact solution u(r) for the Hydrogen atom with n=1, l=0."""
    u = 2 * r * np.exp(-r)
    return u ** 2


def rseq(t, y, en, l):
    """Radial Schrödinger Equation as system of 1st order ODEs."""
    y1, y2 = y
    y1d = y2
    f = l * (l + 1) / (t ** 2) - 2 * (en + 1 / t)
    y2d = f * y1
    return [y1d, y2d]


def integrate(en, l=0, tmin=1e-3, tmax=20):
    """Wrapper for integrating the radial SEQ."""
    sol = solve_ivp(
        # function signature must be f(t, y), so use lambda function
        lambda t, y: rseq(t, y, en, l),
        [tmax, tmin],
        [2 * tmax * np.exp(-tmax), 2 * (1 - tmax) * np.exp(-tmax)],
        t_eval=np.linspace(tmax, tmin, 500),
        # test integrating from r=0 outwards
        # [tmin, tmax],
        # [0, 1],
        # t_eval=np.linspace(tmin, tmax, 800),
        # test other initial values
        # [0, -1E-2],
    )
    # reverse output such that t0 < ... < tN with corresponding y0 ... yN
    return sol.y[0, ::-1], sol.t[::-1]


def get_u0(en, l=0):
    """Wrapper for finding u_nl(r=0, E_n) values after integrating rseq."""
    u, r = integrate(en, l)
    return u[0]


def plot_u0():
    u0 = []
    ens = np.linspace(-1, 1, 200)
    for en in ens:
        u0.append(get_u0(en))
    # plot raw data for u(0)
    plt.figure()
    plt.plot(ens, u0, lw=2, label="u(r=0,E)")
    plt.axhline(y=0, lw=2, ls="--", c="k", alpha=0.5)
    plt.legend(loc="best", fancybox=True, shadow=True)


def main():
    # show plot of u(r=0, E) for visually determine roots
    plot_u0()

    # use secant method for finding the root in u_0(E)
    root = newton(get_u0, -1.0)
    # alternatively, use
    # root = newton(lambda x: integrate(x)[0][0], -1.0)
    print("\n[INFO] found energy eigenvalue @ E = {:.5f} Hartree".format(root))

    # integrate again using correct energy
    u, r = integrate(root, tmin=1e-3)
    # normalize u**2 to 1
    u2 = u ** 2
    norm = trapz(u2, r)
    u2 /= norm

    # plot squared normalized function with correct energy u(r, E_n)
    plt.figure()
    plt.plot(r, u2, lw=3, alpha=0.6, label=r"$u^2_\mathrm{num}(r)$")
    # add exact solutions
    plt.plot(r, u2ex(r), ls="--", lw=2, label=r"$u^2_\mathrm{exact}(r)$")
    # horizontal and vertical black lines indicating origin
    plt.axhline(y=0, color="k", ls="--")
    plt.axvline(x=0, color="k", ls="--")
    # add legend box with labels
    plt.legend(loc="best", fancybox=True, shadow=True)

    # determine energy spectrum and plot eigenfunctions
    print("--- spectrum of H-atom ---")
    for l in range(6):
        root = newton(lambda en: get_u0(en, l), -1.0)
        n = l + 1
        en = -1 / (2 * n ** 2)
        print("l = {}  -->  n = {} : En = {: .6}\t~ {}".format(l, n, root, en))

    # draw all plots to screen
    plt.show()


if __name__ == "__main__":
    main()

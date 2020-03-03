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

""" Example implementation of DFT Assignment 4: Radial SEQ for the H-atom

This script implements the full radial Schrödinger equation for the H-atom
according to Problem Sheet 4, Eq. (29) + atomic units. First, the 2nd order ODE
is converted to a system of 1st order equations in rseq(). This function is
then integrated for a prescribed range, energy and orbital quantum number using
the integrate() routine. The latter is coupled to the Secant root finding
algorithm from scipy, that should eventually return the lowest energy for the
given setup, provided the numerical parameters are set properly.

For the hydrogen ground state, i.e. 1 electron in the 1s-orbital, this script
yields almost exactly -0.5 Hartree.

In addition, we can find the beginning of the H-atom spectrum using a neat
trick (see assignment sheet):

l  -->  n : En/Hartree  ~  exact/Hartree
----------------------------------------
0  -->  1 : -0.499981   ~ -0.5
1  -->  2 : -0.124997   ~ -0.125
2  -->  3 : -0.0544761  ~ -0.0555556
3  -->  4 : -0.0302866  ~ -0.03125
4  -->  5 : -0.0189096  ~ -0.02
5  -->  6 : -0.0126417  ~ -0.0138889
6  -->  7 : -0.0340679  ~ -0.0102041
7  -->  8 : -0.24337    ~ -0.0078125

Unfortunately, the numerical values are far off for n>6. For small principal
quantum numbers, however, we find quite accurate energies when comparing to the
exact ones, En = -Z^2 / (2 * n^2).
"""
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import newton
import matplotlib.pyplot as plt
import numpy as np


def u2ex(r):
    """Normalized exact solution u(r) for the Hydrogen atom with n=1, l=0."""
    u = 2 * r * np.exp(-r)
    return u ** 2


def rseq(t, y, en, l):
    """Radial Schrödinger equation as system of 1st order ODEs."""
    y1, y2 = y
    y1d = y2
    f = l * (l + 1) / (t ** 2) - 2 * (en + 1 / t)
    y2d = f * y1
    return [y1d, y2d]


def integrate(en, l=0, tmin=1e-5, tmax=20):
    """Wrapper for integrating the radial SEQ."""
    sol = solve_ivp(
        # function signature must be f(t, y), so use lambda function
        lambda t, y: rseq(t, y, en, l),
        [tmax, tmin],
        # use values from exact radial function:
        # u(r) = 2r exp(-r), u'(r) = 2(1-r) exp(-r)
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


def get_u0(en, l=0, rmax=20):
    """Wrapper for finding u_nl(r=0, E_n) values after integrating rseq."""
    u, r = integrate(en, l, tmax=rmax)
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
    print("\n/-------------------------------\\")
    print("|   Ass.4: spectrum of H-atom   |")
    print("\\-------------------------------/\n")
    # show plot of u(r=0, E) for visually determine roots
    plot_u0()

    # use secant method for finding the root in u_0(E)
    root = newton(get_u0, -1.0)
    # alternatively, use
    # root = newton(lambda x: integrate(x)[0][0], -1.0)
    print("[INFO] found energy eigenvalue @ E = {:.5f} Hartree".format(root))
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
    print("\nl  -->  n : En/Hartree  ~  exact/Hartree")
    print("----------------------------------------")
    for l in range(8):
        rmax = max(20, 10*l)
        r = newton(lambda en: get_u0(en, l, rmax=rmax), -1.0, maxiter=200)
        n = l + 1
        en = -1 / (2 * n ** 2)
        print("{}  -->  {} : {: .6}\t~  {:.6}".format(l, n, r, en))

    # draw all plots to screen
    plt.show()


if __name__ == "__main__":
    main()

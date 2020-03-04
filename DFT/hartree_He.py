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

""" Example implementation of DFT Assignment 5.2: Self-consistent loop for He

This is a self-consistent version of the A5.1 script with support for the
He-atom. Instead of the real electron-electron interaction term, we use the
classical Hartree potential and include a Hartree-Fock-like self-interaction
correction (SIC), which amounts to half of the proper Hartree potential. This
hack will be replaced by the LDA exchange-correlation (XC) potential in the
next assignment. Nevertheless, the code structure already follows the general
Kohn-Sham self-consconsistency cycle (see below).

The Kohn-Sham basically contains 6 steps (see "DFT Cheat Sheet" for details):
    0) initialize density; here we use the exact density of the H-atom,
    1) build the effective Kohn-Sham potential; here no XC but SIC-Hartree,
    2) solve the single-particle Schrödinger-like equation,
    3) build the new density from single-particle solutions,
    4) evaluate the total energy density functional E[n] for the new density,
    5) check for self-consistency,
    6) mix new and old density for faster convergence and increased stability.

In the following example implementation, all radial integrals are expressed
in terms of the function

                             den(r) = Z * |u(r)|^2 ,                        (1)

which can be used similar to

                             n_tot(x) = Z * n_s(x)                          (2)

in 3D integrals in case the density is radially symmetric,

                   n_s(x) = n_s(r) = (4 pi r^2)^(-1) |u(r)|^2 .             (3)

This makes it easier to compare code and equations from the exercise sheets.
For example, the integral from the equation for the Hartree energy can be
reformulated into

    /                         /+inf                    /+inf
    | d3x v(r) n_tot(r) = 4pi | dr r^2 v(r) n_tot(r) = | dr v(r) den(r) .   (4)
    /|R^3                     /0                       /0

Note, that the Hartree potential obtained by solving Poisson's equation
corresponds to the density from Eq.(2), where Z is either 1 (H) or 2 (He).
Consequently, we have to use q_tot = Z for the determination of the homogeneous
solution w_hom(r) = beta * r. By contrast, u(r) still is the solution of a
one-electron Schrödinger equation with wave function

             psi(r) = Y_00 * u(r) / r   and   n_s(r) = |psi(r)|^2 ,         (5)

and thus |u(r)|^2 has to be normalized to 1 regardless of the value of Z.
"""
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import newton
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np


# we use some global values
z = 2  # may be 1 for H-atom or 2 for He-atom
h = 0.02
tmin = 1e-5
tmax = 20
nsteps = round((tmax - tmin) / h)


def rseq(t, y, en, veff):
    """Radial SEQ with l=0 and effective potential veff(r)."""
    y1, y2 = y
    y1d = y2
    # compare with Eq.(11) from Problem Sheet 5 and Eq.(3) from Problem Sheet 4
    f = -2 * (en - veff(t))
    y2d = f * y1
    return [y1d, y2d]


def solve_rseq(en, veff):
    """Wrapper for integrating the radial SEQ inwards."""
    sol = solve_ivp(
        # function signature must be f(t, y), so use lambda function
        lambda t, y: rseq(t, y, en, veff),
        [tmax, tmin],
        # use values from exact radial function for hydrogen:
        # u(r) = 2r exp(-r), u'(r) = 2(1-r) exp(-r)
        [2 * tmax * np.exp(-tmax), 2 * (1 - tmax) * np.exp(-tmax)],
        t_eval=np.linspace(tmax, tmin, nsteps),
    )
    # reverse output such that t0 < ... < tN with corresponding y0 ... yN
    return sol.y[0, ::-1], sol.t[::-1]


def poisson(t, y, den):
    """Poisson's equation for radially symmetric density den(r) = |u(r)|^2."""
    y1, y2 = y
    y1d = y2
    y2d = -den(t) / t
    return [y1d, y2d]


def solve_poisson(den):
    """Wrapper for integrating Poisson's equation outwards."""
    # integrate forward from 0+eps to rmax with y(t) = w(r) = r * v_H(r)
    sol = solve_ivp(
        lambda t, y: poisson(t, y, den),
        [tmin, tmax],
        # use initial value w(0)=0 and w'(r) = (2r + 1) * e^(-2r) --> w'(0)=1
        [0, 1],
        t_eval=np.linspace(tmin, tmax, nsteps),
    )
    return sol.y[0, :], sol.t


def energy_functional(den, vh, en):
    """Builds total energy from SP-SEQ eigenvalues and Hartree energy. """
    print("[INFO] building new total energy:")
    # evaluate and integrate functions on a dense grid
    r = np.linspace(tmin, tmax, 3 * nsteps)
    # comapre with Eq.(1) from Assignment Sheet
    e_h = 0.5 * trapz(vh(r) * den(r), r)
    e_tot = z * en - e_h
    print("\t SP-SEQ eigenvalue:  {: .4f} Ha".format(en))
    print("\t SIC-hartree energy: {: .4f} Ha".format(e_h))
    print("\t ------------------------------")
    print("\t total energy:       {: .4f} Ha".format(e_tot))
    return e_tot


def iteration_step(den):
    # -- Step 1: build effective Kohn-Sham potential --
    # determine w(r) via Poisson's equation for initial or mixed density
    w, r = solve_poisson(den)
    # find hom.sol. w_hom(r) = a*r to match BC w(r_max) = q_tot = z
    beta = (z - w[-1]) / r[-1]
    w += beta * r
    print("[INFO] adding hom. solution with beta = {:.4f}".format(beta))
    # interpolate directly Hartree potential since we don't need w(r) anymore
    # and include a Hartree-Fock-like exchange of -1/2 * Hartree potential
    vh = Spline(r, 0.5 * w / r)

    def veff(r):
        vext = -z / r
        # compare with DFT cheat sheet formulae
        return vh(r) + vext

    # -- Step 2: solve single-particle Schrödinger-like equation --
    # integration returns (u(r_i), r_i) --> u(r_0) = solver(E)[0][0]
    en = newton(lambda en0: solve_rseq(en0, veff)[0][0], -2.0, maxiter=50)
    # integrate again using the correct energy
    u, r = solve_rseq(en, veff)
    # normalize u^2 to 1 independent of z-value since psi(r) = Y_00 * u(r) / r
    norm = trapz(u ** 2, r)
    u /= np.sqrt(norm)
    print("[INFO] normalizing |u(r)|^2 from {:.3g} to 1".format(norm))

    # -- Step 3: construct and interpolate new density --
    den = Spline(r, z * u ** 2)

    # -- Step 4: compute total energy
    etot = energy_functional(den, vh, en)

    return etot, den


def main():
    print("\n/---------------------------------------------------\\")
    print("| Ass.5.2: Self-consistent loop for the helium atom |")
    print("\\---------------------------------------------------/\n")

    print("[INFO] using h = {} for numerically solving ODEs".format(h))

    print("\n[ITER] initialize with density of H-atom")
    # -- Step 0: initialize density with exact H-atom-like density --
    den_init = lambda r: z * z ** 3 * r * np.exp(-2 * z * r)  # noqa
    # -- Step 0: alternatively initialize density with random numbers --
    # import random
    # den_init = lambda r: random.randrange(0, 2)
    den_mix = den_old = den_init
    etot_old = 0
    iterstep = 0
    # repeat until convergence is achieved
    while True:
        iterstep += 1
        print("\n\n[ITER] starting {}. iteration step".format(iterstep))
        # -- Steps 1 to 4: solve Kohn-Sham equations and find total energy --
        etot_new, den_new = iteration_step(den_mix)
        ediff = abs(etot_new - etot_old)
        print("[INFO] energy difference dE = {:.5g}".format(ediff))
        # -- Step 5: check for convergence --
        if ediff > 1e-5:
            # -- Step 6: mix densities for faster convergence --
            # sweet spot for He: 0.88
            a = 0.88
            r = np.linspace(tmin, tmax, 3 * nsteps)
            den_mix = Spline(r, a * den_new(r) + (1 - a) * den_old(r))
            den_old = den_new
            etot_old = etot_new
        if iterstep > 20:
            print("\n[STOP] could not achieve convergence in 20 iterations\n")
            break
        else:
            print("\n[STOP] convergence achieved in", iterstep, "steps\n")
            print("/------------------------------------\\")
            print("|     >>> final total energy <<<     |")
            print("|                                    |")
            print("|     E[n] = {} * ev - E_H(SIC)[n]    |".format(z))
            print("|          = {:.4f} Ha              |".format(etot_new))
            print("\\------------------------------------/")
            break


if __name__ == "__main__":
    main()

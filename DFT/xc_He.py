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

""" Example implementation of DFT Assignment 7: Kohn-Sham loop incl. XC for He

In this version of the self-consistent DFT code for the helium atom, we employ
the full Kohn-Sham effective potential, including exchange and correlation.
We use the easy-to-implement Local Density Approximation, where the exchange
(particle) density is given directly in terms of the (local) charge density.
For the correlation density, we implement two versions: Ceperley/Alder's
parametrization (1980) and the much newer and easier (revised) Chachiyo
parametrization (2016).

Note, that we use the proper LDA exchange, not the Slater version!

The iteration_step() function follows again the general Kohn-Sham cycle,
consisting of (see "DFT Cheat Sheet" for details):

    0) initialize density,
    1) build the effective Kohn-Sham potential incl. exchange and correlation,
    2) solve the single-particle Schrödinger-like equation,
    3) build the new density from single-particle solutions,
    4) evaluate the total energy density functional E[n] for the new density,
    5) check for self-consistency,
    6) mix new and old density for faster convergence and increased stability.

Integrals are again expressed in terms of den(r) = Z * |u(r)|^2 (see docstring
in hartree_He.py for details). Further, the energy functional

                                             /+inf
             E[n] = E_s - E_H[n] + E_xc[n] - | dr den(r) v_xc(r)            (1)
                                             /0

(with E_s = Z * RSEQ-eigenvalue) can be re-arranged into

                                   /+inf
             E[n] = E_s - E_H[n] + | dr den(r) (e_xc - v_xc)(r)             (2)
                                   /0

for the LDA, where E_xc[n] is a local functional of the density. By separating
the xc-density according to

                              e_xc = e_x + e_c ,                            (3)

we can split the integral in Eq.(1) into a (total) exchange and correlation
contribution, which are listed in each iteration step separately. From these
data, we can confirm that the correlation contribution is much smaller compared
to the exchange part.

Running the code with different xc-settings results in total energies, which
are in good agreement with the experimental value:

                 exp.        : -2.903  Ha
                 SIC-Hartree : -2.8615 Ha (7 iteration steps)
                 x + c(CA)   : -2.8298 Ha (9 iteration steps)
                 x + c(rCY)  : -2.8283 Ha (9 iteration steps)
                 x + c(CY)   : -2.8274 Ha (9 iteration steps)
                 x only      : -2.7233 Ha (9 iteration steps)
                 no xc       : -1.9515 Ha (75+ iteration steps)

Note, that in some routines we use the same names for arrays and the respective
splines for better "physical" reading, even if this should be considered bad
coding practice in general. Same for global variables, which we put at the very
start of the script for quick parameter adjustments.
"""
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import newton
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np
import random


# we use some global values
z = 2  # may be 1 for H-atom or 2 for He-atom
h = 0.01
tmin = 1e-5
tmax = 20
nsteps = round((tmax - tmin) / h)
refine = 3
nsteps_dense = refine * nsteps

itermax = 50
etol = 1e-5
amix = 0.88

init_density = "H-atom"  # use "H-atom" or "random"
correlation = "ca"  # use "chachiyo" | "rev-chachiyo" | "ca" | "none"
exchange = True


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


def ceperley_alder(rs_arr):
    """Returns correlation density and potential by Ceperley/Alder (1980)."""
    # Phys.Rev.Lett. 45 (7) p. 566-569; doi:10.1103/PhysRevLett.45.566
    if z == 1:
        # spin-polarized
        A, B, C, D = 0.01555, -0.0269, 0.0007, -0.0048
        g, b1, b2 = -0.0843, 1.3981, 0.2611
    else:
        # unpolarized
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        g, b1, b2 = -0.1423, 1.0529, 0.3334
    ec = np.empty_like(rs_arr)
    vc = np.empty_like(rs_arr)
    for idx, rs in enumerate(rs_arr):
        sqrs = np.sqrt(rs)
        lnrs = np.log(rs)
        if rs >= 1:
            # wrong expressions from Thijssen:
            # num = 1 + 7/6 * b1 * np.sqrt(rs) + b2 * rs
            # denom = 1 + b1 * np.sqrt(rs) + b2 * rs
            # vc[idx] = ec[idx] * num / denom
            ec[idx] = g / (1 + b1 * sqrs + b2 * rs)
            vc[idx] = ec[idx] + ec[idx]**2 * (b1/6 * sqrs + b2/3 * rs) / g
        else:
            ec[idx] = A * lnrs + B + C * rs * lnrs + D * rs
            vc[idx] = B + A * (lnrs - 1/3) + (2*C * lnrs + 2*D - C) * rs/3
    return vc, ec


def chachiyo(rs, rev=False):
    """Returns (revised) Chachiyo correlation density and potential (2016)."""
    # J.Chem.Phys. 145, 021101 (2016); doi:10.1063/1.4958669
    # J.Chem.Phys. 145, 157101 (2016); doi:10.1063/1.4964758 (rev. by Karasiev)
    if z == 1:
        # spin-polarized
        a = (np.log(2) - 1) / (4 * np.pi**2)
        b = 27.420360
        b1 = 28.3559732
    else:
        # unpolarized
        a = (np.log(2) - 1) / (2 * np.pi**2)
        b = 20.4562557
        b1 = 21.7392245
    if not rev:
        # go back to original Chachiyo parametrization
        b1 = b
    # use vc(rs) = ec(rs) - rs/3 * ec'(rs)
    ec = a * np.log(1 + b1/rs + b/rs**2)
    vc = ec + a/3 * (2 * b + b1 * rs) / (b + b1 * rs + rs**2)
    return vc, ec


def energy_functional(den, vh, dx, dc, en):
    """Builds total energy from SP-SEQ eigenvalues, Hartree and XC energies."""
    print("[INFO] building new total energy:")
    # evaluate and integrate functions on a dense grid
    r = np.linspace(tmin, tmax, nsteps_dense)
    # calculate Hartree energy and exchange/correlation contributions
    eh = 1/2 * trapz(vh(r) * den(r), r)
    x_contrib = trapz(dx(r) * den(r), r)
    c_contrib = trapz(dc(r) * den(r), r)
    xc = x_contrib + c_contrib
    e_tot = z * en - eh + xc
    print("\t SP-SEQ eigenvalue:  {: .4f} Ha".format(en))
    print("\t Hartree energy:     {: .4f} Ha".format(eh))
    print("\t XC contribution:    {: .4f} Ha".format(xc))
    print("\t    - exchange:      {: .4f} Ha".format(x_contrib))
    print("\t    - correlation:   {: .4f} Ha".format(c_contrib))
    print("\t ------------------------------")
    print("\t total energy:       {: .4f} Ha".format(e_tot))
    return e_tot


def iteration_step(den):
    """Performs one iteration of the Kohn-Sham cycle (see DFT cheat sheet)."""
    # -- Step 1: build effective Kohn-Sham potential --
    # -- Step 1.1: find w(r) via radial Poisson equation for old density --
    w, r = solve_poisson(den)
    # find hom.sol. w_hom(r) = a*r to match BC w(r_max) = q_tot = z
    beta = (z - w[-1]) / r[-1]
    w += beta * r
    print("[INFO] using w_hom(r) with beta = {:.4f}".format(beta))
    # interpolate directly Hartree potential since we don't need w(r) anymore
    vh = Spline(r, w / r)

    # prevent RuntimeWarning for small negative density at r_min ~ 0 a.u.
    # --> https://stackoverflow.com/a/45384691
    den = den(r)
    den[0] += 1e-10

    # -- Step 1.2: build LDA exchange density and corresponding potential --
    if exchange is True:
        pre = 3 / (4 * np.pi**2)
        vx = -(pre * den / r**2)**(1 / 3)
    else:
        vx = np.zeros_like(den)
    dx = Spline(r, -1/4 * vx)
    vx = Spline(r, vx)

    # -- Step 1.3: build correlation density and corresponding potential --
    # convert density into Wigner-Seitz radius r_s(r)
    rs = (3 * r ** 2 / den)**(2/3)
    # use (revised) Chachiyo's parametrization (2016)
    if correlation == "chachiyo":
        vc, ec = chachiyo(rs)
    elif correlation == "rev-chachiyo":
        vc, ec = chachiyo(rs, rev=True)
    # use Ceperley/Alder's parametrization (1980)
    elif correlation == "ca":
        vc, ec = ceperley_alder(rs)
    # do not use any correlation
    elif correlation == "none":
        vc = ec = np.zeros_like(rs)
    else:
        raise ValueError("unknown correlation setting")
    dc = Spline(r, ec - vc)
    vc = Spline(r, vc)

    def veff(r):
        """Effective Kohn-Sham potential --> compare with DFT cheat sheet."""
        vext = -z / r
        return vh(r) + vx(r) + vc(r) + vext

    # -- Step 2: solve single-particle Schrödinger-like equation --
    # integration returns (u(r_i), r_i) --> u(r_0) = solver(E)[0][0]
    en = newton(lambda en0: solve_rseq(en0, veff)[0][0], -2.0, maxiter=50)
    # integrate again using the correct energy
    u, r = solve_rseq(en, veff)
    # normalize u^2 to 1 regardless of z-value since psi(r) = Y_00 * u(r) / r
    norm = trapz(u ** 2, r)
    u /= np.sqrt(norm)
    print("[INFO] normalizing |u(r)|^2 from {:.3g} to 1".format(norm))

    # -- Step 3: construct and interpolate new density --
    den_new = Spline(r, z * u ** 2)

    # -- Step 4: compute total energy
    etot = energy_functional(den_new, vh, dx, dc, en)

    return etot, den_new


def main():
    print("\n/-----------------------------------------------------\\")
    print("| Ass.7: Full Kohn-Sham loop incl. XC for the He atom |")
    print("\\-----------------------------------------------------/\n")

    print("[INIT] parameter settings for this run:")
    print("       electrons:       {}".format(z))
    print("       step size h:     {}".format(h))
    print("       minimum r:       {}".format(tmin))
    print("       maximum r:       {}".format(tmax))
    print("       grid pts:        {}".format(nsteps))
    print("       dense grid pts:  {}".format(nsteps_dense))
    print("       max. iterations: {}".format(itermax))
    print("       convergence dE:  {}".format(etol))
    print("       mixing alpha:    {}".format(amix))
    print("       initial density: {}".format(init_density))
    print("       exchange:        {}".format(exchange))
    print("       correlation:     {}".format(correlation))

    # -- Step 0: initialize density with exact H-atom-like density --
    if init_density == "H-atom":
        den_init = lambda r: z * z ** 3 * r * np.exp(-2 * z * r)  # noqa
    # -- alternatively initialize density with random numbers --
    elif init_density == "random":
        den_init = lambda r: random.randrange(0, 2)  # noqa
    else:
        raise ValueError("unknown initial density setting")

    # init loop variables
    den_mix = den_old = den_init
    etot_old = 0
    iterstep = 0

    # Kohn-Sham cycle: repeat until convergence is achieved
    while True:
        iterstep += 1
        print("\n\n[ITER] starting {}. iteration step".format(iterstep))

        # -- Steps 1 to 4: solve Kohn-Sham equations and find total energy --
        etot_new, den_new = iteration_step(den_mix)
        ediff = abs(etot_new - etot_old)
        print("[INFO] energy difference dE = {:.3e} Ha".format(ediff))

        # -- Step 5: check for convergence --
        if iterstep >= itermax and ediff > etol:
            print("\n[STOP] could not achieve convergence in 20 iterations\n")
            break
        elif ediff > etol:
            # -- Step 6: mix densities for faster convergence --
            r = np.linspace(tmin, tmax, nsteps_dense)
            den_mix = Spline(r, amix * den_new(r) + (1 - amix) * den_old(r))
            den_old = den_new
            etot_old = etot_new
        else:
            print("\n[STOP] convergence achieved in", iterstep, "steps\n")
            print("/------------------------------------\\")
            print("|     >>> final total energy <<<     |")
            print("|                                    |")
            print("|   E[n] = E_s[n] - E_H[n] + XC[n]   |")
            print("|        = {:.4f} Ha                |".format(etot_new))
            print("\\------------------------------------/")
            break


if __name__ == "__main__":
    main()

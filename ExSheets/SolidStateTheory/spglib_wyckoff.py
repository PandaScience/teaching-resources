#!/usr/bin/env python
# vim:set noet ci pi sts=0 sw=4 ts=4 ai sc ic:
#
# usage: ./spglib_wyckoff example.cif example2.cif ...

import argparse
from ase.io import read
from ase.data import chemical_symbols
import spglib
from termcolor import cprint, colored

# parse and evaluate command line arguments
parser = argparse.ArgumentParser(
    description=(
        "Spglib based CIF reader. \n"
        "Prints cell parameters, finds wyckoff symbols and calculates "
        "transformation matrices for cif cell -> ICT standardized cell. "
    )
)
parser.add_argument(
    "cifs",
    type=str,
    metavar="CIF_files",
    nargs="+",
    help="CIF files to read from; at least 1.",
)
args = parser.parse_args()

# start analyzing input
cprint(
    "\n"
    "/----------------------------------------------------------\\\n"
    "|                spglib based CIF analyzer                 |\n"
    "\\----------------------------------------------------------/\n",
    "green",
)
cprint("Following structures were passed to the script:", "green")
print("\n".join(args.cifs))

cprint("\nStarting spacegroup finder...", "green")

for struct in args.cifs:

    cif = read(str(struct))

    print(
        colored("\nCIF file: ", "blue")
        + colored(str(struct), "red")
    )

    cell = (
        cif.get_cell(),
        cif.get_scaled_positions(),
        cif.get_atomic_numbers(),
    )

    data = spglib.get_symmetry_dataset(cell)

    sg = data.international
    print(
        colored("Found spacegroup (intern.): ", "blue")
        + colored(str(sg), "red")
    )

    hall = data.hall
    hallnum = data.hall_number
    print("Hall Number: ", hallnum)
    print("Hall Symbols: ", hall)

    pointgrp = data.pointgroup
    print("Point Group: ", pointgrp)

    cprint("Cell according to input CIF file:", "blue")
    choice = data.choice
    print("Cell Choice: ", choice)
    origin = data.origin_shift
    print("Origin Shift: " + "  ".join(str(n) for n in origin))
    print("Cell Parameters:")
    rcell = cell[0].round(decimals=4)
    print("\n".join("\t".join(str(cell) for cell in row) for row in rcell))

    print("Atomic Positions and their Wyckoff ID:")
    wyckoffs = data.wyckoffs
    for i in range(len(cell[1])):
        pos = cell[1][i].round(decimals=4)
        sym = chemical_symbols[cell[2][i]]
        print(wyckoffs[i], "  ", sym, "\t", "\t".join(map(str, pos)))

    cprint("Standardized cell according to ICT: ", "blue")
    # cell = standardize_cell(cell)
    # data = get_symmetry_dataset(cell)
    wyckoffs = data.wyckoffs
    lattice = data.std_lattice
    pos = data.std_positions
    types = data.std_types
    trafo = data.transformation_matrix

    print("Std. Cell Parameters:")
    print("\n".join("\t".join(str(cell) for cell in row) for row in lattice))

    print("Std. Atomic Positions and their Wyckoff ID:")
    for i in range(len(types)):
        sym = chemical_symbols[types[i]]
        print(wyckoffs[i], "  ", sym, "\t", "\t".join(map(str, pos[i])))

    print("Transformation Matrix:")
    trafo = trafo.round(5)
    print("\n".join("\t".join(str(cell) for cell in row) for row in trafo))

cprint("\nFinished!", "green")

# EOF

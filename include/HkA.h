#include <vector>
#include <complex>

#ifndef TBG_HKA_H
#define TBG_HKA_H

void set_HkA(std::vector<std::complex<double>> &HkA, const std::vector<double> &kvec, const std::vector<double> &lvec, const std::vector<std::vector<double>> &UNIT_CELL, const double g, const std::vector<double> eA);
/**
 *	Set time-dependent Hamiltonian matrix with Peierls field
 *  -HkA: Complex vector[NATOM*NATOM] to store Hamiltonian linearly coupled to light
 *  -kvec: Real vector of the reciprocal space - k-point for which HkA will be calculated
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -g light-matter coupling constant
 *  -eA polarization vector of light
 */

std::vector<double> calcLinABands(const std::vector<std::vector<double>> &firstBZ,
                                  const vector<dvec> &UNIT_CELL,
                                  const dvec &lvec);
/**
 *	Calculate bands of HkA(k) in the 1st BZ
 *  -bands: Vector to store eigenvalues of all k-points
 *  -firstBZ: irreducible BZ
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real Vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 */

#endif //TBG_HKA_H

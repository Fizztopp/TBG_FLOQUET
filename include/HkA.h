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

/**
 * Calculate full Hamiltonian with Complete Peierls Coupling
 * @param HkA Output matrix
 * @param kvec 3D k-point for which matrix element should be calculated
 * @param lvec super-lattice vector
 * @param UNIT_CELL atomic positions in unit-cell
 * @param g light-matter coupling
 * @param eA unit-vector of A-field direction
 */
void set_HkExpCoupling(std::vector<std::complex<double>> &HkA,
                       const std::vector<double> &kvec,
                       const std::vector<double> &lvec,
                       const std::vector<std::vector<double>> &UNIT_CELL,
                       const double g,
                       const std::vector<double> eA);


/**
 * Calculates linear light-coupled hamiltonian in a given basis
 *
 * basisVectors is a matrix of vectors which define the basis in which HkA should be represented
 * kvec defines the k-vector for which HkA should be calcualted
 * lvec are the real super-lattice vectors
 * UNIT_CELL contains the atomic positions in the unit-cell
 */
std::vector<std::complex<double>> HkAInGivenBasis(const std::vector<std::complex<double>>& basisVectors,
                                                  const std::vector<double> &kvec,
                                                  const std::vector<double> &lvec,
                                                  const std::vector<std::vector<double>> &UNIT_CELL);

/**
 *	Set time-dependent Hamiltonian matrix with Peierls field
 *  -HkA: Complex vector[NATOM*NATOM] to store Hamiltonian linearly coupled to light
 *  -kvec: Real vector of the reciprocal space - k-point for which HkA will be calculated
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -g light-matter coupling constant
 *  -eA polarization vector of light
 */
void set_HkAA(std::vector<std::complex<double>> &HkAA,
              const std::vector<double> &kvec,
              const std::vector<double> &lvec,
              const std::vector<std::vector<double>> &UNIT_CELL,
              const double g,
              const std::vector<double> eA);

/**
 * Calculates linear light-coupled hamiltonian in a given basis
 *
 * basisVectors is a matrix of vectors which define the basis in which HkA should be represented
 * kvec defines the k-vector for which HkA should be calcualted
 * lvec are the real super-lattice vectors
 * UNIT_CELL contains the atomic positions in the unit-cell
 */
std::vector<std::complex<double>> HkAAInGivenBasis(const std::vector<std::complex<double>> &basisVectors,
                                                   const std::vector<double> &kvec,
                                                   const std::vector<double> &lvec,
                                                   const std::vector<std::vector<double>> &UNIT_CELL);


/**
 *
 * @param basisVectors basis in which to transform matrix result
 * @param kvec k-vector for which to calculate matrix
 * @param lvec super-lattice vectors
 * @param UNIT_CELL atomic positions in unit-cell
 * @return Matrix linearly coupled to A in the basis in which Hk0 is diagonal
 */
std::vector<std::complex<double>> HkExpCouplingInGivenBasis(const std::vector<std::complex<double>> &basisVectors,
                                                            const std::vector<double> &kvec,
                                                            const std::vector<double> &lvec,
                                                            const std::vector<std::vector<double>> &UNIT_CELL);


/**
 *	Calculate bands of HkA(k) in the 1st BZ
 *  -bands: Vector to store eigenvalues of all k-points
 *  -firstBZ: irreducible BZ
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real Vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 */
std::vector<double> calcLinABands(const std::vector<std::vector<double>> &firstBZ,
                                  const vector<dvec> &UNIT_CELL,
                                  const dvec &lvec);

#endif //TBG_HKA_H

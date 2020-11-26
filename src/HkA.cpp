#include <vector>
#include <complex>
#include <cassert>
#include <fstream>
#include <iostream>

#include "Constants.h"
#include "InlineFunctions.h"
#include "Diagonalization.h"
#include "Hk0.h"
#include "mkl.h"

void set_HkA(std::vector<std::complex<double>> &HkA,
             const std::vector<double> &kvec,
             const std::vector<double> &lvec,
             const std::vector<std::vector<double>> &UNIT_CELL,
             const double g,
             const std::vector<double> eA)
/**
 *	Set time-dependent Hamiltonian matrix with Peierls field
 *  -HkA: Complex vector[NATOM*NATOM] to store Hamiltonian linearly coupled to light
 *  -kvec: Real vector of the reciprocal space - k-point for which HkA will be calculated
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -g light-matter coupling constant
 *  -eA polarization vector of light
 */
{
    //assert that vector sizes are correct
    assert(HkA.size() == NATOM * NATOM);
    assert(kvec.size() == 3);
    assert(lvec.size() == 4);
    assert(UNIT_CELL.size() == NATOM);
    assert(eA.size() == 3);
    //assert that polarization vector has unit length
    assert(abs(eA[0] * eA[0] + eA[1] * eA[1] + eA[2] * eA[2] - 1.0) < 1e-10);

    const double lcell = lconst * sqrt(pow(lvec[0], 2.) + pow(lvec[1], 2.));
    const double qq2 = qq1 * aa2 / aa1;
    const double kx = kvec[0];
    const double ky = kvec[1];

    std::fill(HkA.begin(), HkA.end(), 0.0);

    // Bottom layer
    double d, rx, ry, rz;
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        // Back-gate voltage
        HkA[fq(i, i, NATOM)] = VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM / 2; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
    // Top layer
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = NATOM / 2; i < NATOM; ++i) {
        // Top-gate voltage
        HkA[fq(i, i, NATOM)] = -VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
    // Inter-layer terms
#ifndef NO_IC
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        for (int j = NATOM / 2; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    // Vpp_pi term
                    HkA[fq(i, j, NATOM)] += (1. - pow(aa2 / d, 2.)) * t1 / RG * exp(qq1 * (1. - (d / aa1)))
                                            * exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                    // Vpp_sigma term
                    HkA[fq(i, j, NATOM)] += pow(aa2 / d, 2.) * t2 * exp(qq2 * (1. - (d / aa2))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;

                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
#endif
}

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
              const std::vector<double> eA) {
    //assert that vector sizes are correct
    assert(HkAA.size() == NATOM * NATOM);
    assert(kvec.size() == 3);
    assert(lvec.size() == 4);
    assert(UNIT_CELL.size() == NATOM);
    assert(eA.size() == 3);
    //assert that polarization vector has unit length
    assert(abs(eA[0] * eA[0] + eA[1] * eA[1] + eA[2] * eA[2] - 1.0) < 1e-10);

    const double lcell = lconst * sqrt(pow(lvec[0], 2.) + pow(lvec[1], 2.));
    const double qq2 = qq1 * aa2 / aa1;
    const double kx = kvec[0];
    const double ky = kvec[1];

    std::fill(HkAA.begin(), HkAA.end(), 0.0);

    // Bottom layer
    double d, rx, ry, rz;

#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        // Back-gate voltage
        HkAA[fq(i, i, NATOM)] = VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkAA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkAA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM / 2; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkAA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                             exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                             (-0.5) * g * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst *
                                             (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                }
            }
            HkAA[fq(j, i, NATOM)] = conj(HkAA[fq(i, j, NATOM)]);
        }
    }
    // Top layer
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = NATOM / 2; i < NATOM; ++i) {
        // Top-gate voltage
        HkAA[fq(i, i, NATOM)] = -VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkAA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkAA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkAA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                             exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                             (-0.5) * g * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst *
                                             (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                }
            }
            HkAA[fq(j, i, NATOM)] = conj(HkAA[fq(i, j, NATOM)]);
        }
    }
    // Inter-layer terms
#ifndef NO_IC
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        for (int j = NATOM / 2; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    // Vpp_pi term
                    HkAA[fq(i, j, NATOM)] += (1. - pow(aa2 / d, 2.)) * t1 / RG * exp(qq1 * (1. - (d / aa1)))
                                             * exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                             (-0.5) * g * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst *
                                             (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;
                    // Vpp_sigma term
                    HkAA[fq(i, j, NATOM)] += pow(aa2 / d, 2.) * t2 * exp(qq2 * (1. - (d / aa2))) *
                                             exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                             (-0.5) * g * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst *
                                             II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst;

                }
            }
            HkAA[fq(j, i, NATOM)] = conj(HkAA[fq(i, j, NATOM)]);
        }
    }
#endif

}


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
                       const std::vector<double> eA) {
    //assert that vector sizes are correct
    assert(HkA.size() == NATOM * NATOM);
    assert(kvec.size() == 3);
    assert(lvec.size() == 4);
    assert(UNIT_CELL.size() == NATOM);
    assert(eA.size() == 3);
    //assert that polarization vector has unit length
    assert(abs(eA[0] * eA[0] + eA[1] * eA[1] + eA[2] * eA[2] - 1.0) < 1e-10);

    const double lcell = lconst * sqrt(pow(lvec[0], 2.) + pow(lvec[1], 2.));
    const double qq2 = qq1 * aa2 / aa1;
    const double kx = kvec[0];
    const double ky = kvec[1];

    std::fill(HkA.begin(), HkA.end(), 0.0);

    // Bottom layer
    double d, rx, ry, rz;
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        // Back-gate voltage
        HkA[fq(i, i, NATOM)] = VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM / 2; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            exp(II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst);
                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
    // Top layer
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = NATOM / 2; i < NATOM; ++i) {
        // Top-gate voltage
        HkA[fq(i, i, NATOM)] = -VV / 2.;
        // Sublattice potential
        if (UNIT_CELL[i][3] < 0.9) {
            HkA[fq(i, i, NATOM)] += -dgap / 2.;
        } else {
            HkA[fq(i, i, NATOM)] += dgap / 2.;
        }
        for (int j = i + 1; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    HkA[fq(i, j, NATOM)] += t1 / RG * exp(qq1 * (1. - (d / aa1))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            exp(II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst);
                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
    // Inter-layer terms
#ifndef NO_IC
#pragma omp parallel for private(d, rx, ry, rz)
    for (int i = 0; i < NATOM / 2; ++i) {
        for (int j = NATOM / 2; j < NATOM; ++j) {
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    rx = UNIT_CELL[i][0] - UNIT_CELL[j][0] + double(m - 1) * lvec[0] + double(n - 1) * lvec[2];
                    ry = double(m - 1) * lvec[1] + UNIT_CELL[i][1] - UNIT_CELL[j][1] + double(n - 1) * lvec[3];
                    rz = UNIT_CELL[i][2] - UNIT_CELL[j][2];
                    d = lconst * sqrt(pow(rx, 2.) + pow(ry, 2.) + pow(rz, 2.));
                    // Vpp_pi term
                    HkA[fq(i, j, NATOM)] += (1. - pow(aa2 / d, 2.)) * t1 / RG * exp(qq1 * (1. - (d / aa1)))
                                            * exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            exp(II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst);
                    // Vpp_sigma term
                    HkA[fq(i, j, NATOM)] += pow(aa2 / d, 2.) * t2 * exp(qq2 * (1. - (d / aa2))) *
                                            exp(II * (kx * lconst * rx + ky * lconst * ry)) *
                                            exp(II * g * (eA[0] * rx + eA[1] * ry + eA[2] * rz) * lconst);
                }
            }
            HkA[fq(j, i, NATOM)] = conj(HkA[fq(i, j, NATOM)]);
        }
    }
#endif
}

/**
 * Calculates linear light-coupled hamiltonian in a given basis
 *
 * basisVectors is a matrix of vectors which define the basis in which HkA should be represented
 * kvec defines the k-vector for which HkA should be calcualted
 * lvec are the real super-lattice vectors
 * UNIT_CELL contains the atomic positions in the unit-cell
 */
std::vector<std::complex<double>> HkAInGivenBasis(const std::vector<std::complex<double>> &basisVectors,
                                                  const std::vector<double> &kvec,
                                                  const std::vector<double> &lvec,
                                                  const std::vector<std::vector<double>> &UNIT_CELL) {

    const int N = NATOM;

    std::vector<std::complex<double>> HkA(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP1(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP2(N * N, std::complex<double>(0.0, 0.0));

    set_HkA(HkA, kvec, lvec, UNIT_CELL, cavityConstants::g, cavityConstants::eA);

    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                N, N, N, &alpha,
                &HkA[0], N,
                &basisVectors[0], N,
                &beta, &TEMP1[0], N);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &alpha,
                &basisVectors[0], N,
                &TEMP1[0], N,
                &beta, &TEMP2[0], N);

    return TEMP2;
}

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
                                                   const std::vector<std::vector<double>> &UNIT_CELL) {

    const int N = NATOM;

    std::vector<std::complex<double>> HkAA(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP1(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP2(N * N, std::complex<double>(0.0, 0.0));

    set_HkAA(HkAA, kvec, lvec, UNIT_CELL, cavityConstants::g, cavityConstants::eA);
    //set_Hk0(kvec, HkA, lvec, UNIT_CELL);

    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                N, N, N, &alpha,
                &HkAA[0], N,
                &basisVectors[0], N,
                &beta, &TEMP1[0], N);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &alpha,
                &basisVectors[0], N,
                &TEMP1[0], N,
                &beta, &TEMP2[0], N);

    return TEMP2;
}


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
                                                            const std::vector<std::vector<double>> &UNIT_CELL) {

    const int N = NATOM;

    std::vector<std::complex<double>> HkA(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP1(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP2(N * N, std::complex<double>(0.0, 0.0));

    set_HkExpCoupling(HkA, kvec, lvec, UNIT_CELL, cavityConstants::g, cavityConstants::eA);
    //set_Hk0(kvec, HkA, lvec, UNIT_CELL);

    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                N, N, N, &alpha,
                &HkA[0], N,
                &basisVectors[0], N,
                &beta, &TEMP1[0], N);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &alpha,
                &basisVectors[0], N,
                &TEMP1[0], N,
                &beta, &TEMP2[0], N);

    return TEMP2;
}


std::vector<double> calcLinABands(const std::vector<std::vector<double>> &firstBZ,
                                  const vector<dvec> &UNIT_CELL,
                                  const dvec &lvec)
/**
 *	Calculate bands of HkA(k) in the 1st BZ
 *  -bands: Vector to store eigenvalues of all k-points
 *  -firstBZ: irreducible BZ
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real Vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 */
{

    assert(firstBZ[0].size() == 3);
    assert(lvec.size() == 4);
    assert(UNIT_CELL.size() == NATOM);

    const int num_kpoints_path = firstBZ.size();
    std::vector<double> bands(NATOM * num_kpoints_path);

    std::vector<std::complex<double>> HkA(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<double> evals(NATOM, 0.0);
    for (int k = 0; k < num_kpoints_path; ++k) {
        set_HkA(HkA, firstBZ[k], lvec, UNIT_CELL, cavityConstants::g, cavityConstants::eA);
        diagonalize_eig(HkA, evals);
        for (int m = 0; m < NATOM; m++)
            bands[fq(k, m, NATOM)] = evals[m];
    }
    return bands;
}



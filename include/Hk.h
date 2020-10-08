#ifndef TBG_HK_H
#define TBG_HK_H

#include <vector>

#include "Constants.h"

void set_Hk(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, double time);
/**
 *	Set time-dependent Hamiltonian matrix with Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -time: time variable
 */

void Hk_bands_Floquet(dvec &BANDS_FLOQUET, dvec &OVERLAP_FLOQUET, cvec &Hk_FLOQUET, dvec &evals_FLOQUET, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, int &numprocs, int &myrank);
/**
 *	Calculate Floquet bands by truncated expansion in Floquet eigenfunctions
 *  -BANDS_FLOQUET: Real vector to store Floquet eigenvalues of all k-points
 *  -OVERLAP_FLOQUET: Real vector[num_kpoints_PATHxNATOMx(2*n_max+1)] to store overlap ov Flquet bands with equilibrium bands
 *  -Hk_FLOQUET: Complex vector[(2*m_max+1)x(2*n_max+1)xNATOMxNATOM] to store Flqoeut Hamiltonian matrix
 *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 *  -K_PATH: vector of high-symmetry path vectors
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */

void FloquetEVs(dvec &BANDS, cvec &Hk, dvec &evals, cvec &evals_c, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, double &mu, int &numprocs, int &myrank);
/**
 *	Calculate Floquet bands by diagonalization of the propagator over one period T
 *  -BANDS: Vector to store Floquet eigenvalues of all k-points
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -evals: Vector to store eigenvalues of diagonalization
 *  -evals_c: Vector to store Floquet eigenvalues of diagonalization
 *  -K_PATH: Vector of high-symmetry path vectors
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *	-mu: Chemical potential
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */

void Set_Hk_Floquet(dvec kvec, cvec &Hk_FLOQUET, vector<dvec> &UNIT_CELL, const dvec &lvec);
/**
 *	Set Floquet Hamiltonian in k-orbital basis for use in FLOQUET_BC_LOOP()
 * 	-kvec: Real vector of the reciprocal space
 *  -Hk_FLOQUET: Complex vector[(2*m_max+1)x(2*n_max+1)xNATOMxNATOM] to store Flqoeut Hamiltonian matrix
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 */

#endif //TBG_HK_H

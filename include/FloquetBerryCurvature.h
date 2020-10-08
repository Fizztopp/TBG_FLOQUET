#ifndef TBG_FLOQUETBERRYCURVATURE_H
#define TBG_FLOQUETBERRYCURVATURE_H

#include <vector>
#include <string>

#include "Constants.h"

void FLOQUET_BC_LOOP(dvec kvec, double kmin, double kmax, int Nk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, const string &filename);
/**
 * 	Calculate Berry curvature of expanded Floquet Hamiltonian at kvec by Berry phase over enlosed are in k-space (unit is Angstroem^2)
 *  -kvec: Real vector of the reciprocal space
 * 	-kmin: Double to set loop around kvec
 *  -kmax: Double to set loop around kvec
 * 	-Nk: Number of points pers side to perform loop
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 *  -bands_BCs_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet band Berry curvature
 *	-filename: String to save data
 */

void FLOQUET_BC_LOOP_PATH(double kmin, double kmax, int Nk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, int &numprocs, int &myrank);
/**
 * 	MPI Calculation of Berry curvature of expanded Floquet Hamiltonian along high-symemtry path using FLOQUET_BC_LOOP()
 * 	-kmin: Double to set loop around kvec
 *  -kmax: Double to set loop around kvec
 * 	-Nk: Number of points pers side to perform loop
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -K_PATH: Vector of high-symmetry path vectors
 *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 *  -bands_BCs_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet band Berry curvature
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */

#endif //TBG_FLOQUETBERRYCURVATURE_H

#ifndef TBG_EQBERRYCURVATURE_H
#define TBG_EQBERRYCURVATURE_H

/**
 * 	Calculate local Berry curvature in equlibrium for one k-point by Berry's formula using k-derivative of Hamiltonian
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
void EQ_BC(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals, dvec &bands_BCs, const string& filename);

/**
 * 	Calculate local Berry curvature in equlibrium for single k-point by phase along loop around point divided by area
 *  -kvec: Real vector of the reciprocal space
 * 	-kmin: Double to set loop around kvec
 *  -kmax: Double to set loop around kvec
 * 	-Nk: Number of points pers side to perform loop
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -evals: Real vector[NATOM] to store eigenvalues
 * 	-bands_BCs: Real vector[NATOM] to store band Berry curvature
 *  -filename: String to store data
 */
void EQ_BC_LOOP(dvec kvec, double kmin, double kmax, int Nk, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals, dvec &bands_BCs, const string& filename);

/**
 * 	Calculate Berry curvature in equlibrium for one path of k-points by phase along loop around point divided by area
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 * 	-K_PATH: High-symemtry path through reciprocal cell
 *  -evals: Real vector[NATOM] to store eigenvalues
 * 	-bands_BCs: Real vector[NATOM] to store band Berry curvature
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
void EQ_BC_LOOP_PATH(double kmin, double kmax, int Nk, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals, dvec &bands_BCs, int &numprocs, int &myrank);

/**
 * 	Calculate local Berry curvature in equlibrium for path of k-points by Berry's formula using k-derivative of Hamiltonian
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -K_PATH: High-symemtry path through reciprocal cell
 *  -evals: Real vector[NATOM] to store eigenvalues
 * 	-bands_BCs: Real vector[NATOM] to store band Berry curvature
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
void EQ_BC_LOOP_PATH_CHECK(cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals, dvec &bands_BCs, int &numprocs, int &myrank);


#endif //TBG_EQBERRYCURVATURE_H

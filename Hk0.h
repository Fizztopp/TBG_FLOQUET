#ifndef TBG_HK0_H
#define TBG_HK0_H

/**
 * 	Set eq. Hamiltonian (without external field)
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
void set_Hk0(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL);

/**
 * 	Set derivative along kx of electronic eq. Hamiltonian without Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
void set_dH0dkx(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL);

/**
 * 	Set derivative along ky of electronic eq. Hamiltonian without Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
void set_dH0dky(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL);

/**
 *	Calculate bands of Hk0(k) for path K_PATH through BZ
 *  -BANDS: Vector to store eigenvalues of all k-points
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -evals: Vector to store eigenvalues of diagonalization
 *  -K_PATH: Vector of high-symmetry path vectors
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real Vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 * 	-filename: String to store data
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
void Hk_bands(dvec &BANDS, cvec &Hk, dvec &evals, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, const string& filename, int &numprocs, int &myrank);

void groundstate(cvec &Hk, dvec &evals, vector<dvec> &kweights, vector<dvec> &BZ_IRR, vector<dvec> &UNIT_CELL, const dvec &lvec, double &mu, int &numprocs, int &myrank);
/**
 * 	Calculate initial chemical potential
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -evals: Real vector[NATOM] to store eigenvalues
 *  -kweights: Real vector containing weights of k-points
 *  -BZ_IRR: k-points of irreducable reciprocal cell
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -mu: Chemical potential
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */


#endif //TBG_HK0_H

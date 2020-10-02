#ifndef TBG_FLOQUET_DIAGONALIZATION_H
#define TBG_FLOQUET_DIAGONALIZATION_H

/**
 *  Diagonalization of matrix Hk. Stores eigenvalues in real vector evals and eigenvectors in complex vector Hk
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
void diagonalize(cvec &Hk, dvec &evals);

/**
 *  Diagonalization of matrix Hk. Stores ONLY eigenvalues in real vector evals
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
void diagonalize_eig(cvec &Hk, dvec &evals);


void diagonalize_F(cvec &Hk_FLOQUET, dvec &evals_FLOQUET);
/**
 *  Diagonalization of Floquet matrix H_FLOQUET[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM]
 *  -Hk: Complex vector[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[(2*m_max+1) x NATOM] to store eigenvalues
 */

void diagonalize_GE(cvec &Hk, cvec &TEMP1, cvec &TEMP2, cvec &evals_c);
/**
 *	Diagonalization of NON_Hermitian matrix Hk. Writes eiegenvalues to vector evals_c
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Complex vector[NATOM] to store eigenvalues
 */

#endif //TBG_FLOQUET_DIAGONALIZATION_H

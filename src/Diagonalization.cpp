#include <complex>
#include <vector>
//#include <math.h>
#include <assert.h>
//#define NO_IC                                                         // switch interlayer coupling on/off
using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

#define MKL_Complex16 cdouble
#include "mkl.h"

#include "Constants.h"
#include "Diagonalization.h"

//'N','V':  Compute eigenvalues only, and eigenvectors
char    jobz = 'V';
//'U','L':  Upper, Lower triangle of H is stored
char    uplo = 'U';
// The order of the matrix H.  NATOM >= 0
int     matsize = NATOM;
// The leading dimension of the array H.  lda >= max(1, NATOM)
int     lda = NATOM;
// The length of the array work.  lwork  >= max(1,2* NATOM-1)
int     lwork = 2*NATOM-1;
// dimension (max(1, 3* NATOM-2))
double  rwork[3*NATOM-2];
// dimension (MAX(1,LWORK))
cdouble work[2*NATOM-1];
// Info
int	    info;

void diagonalize(cvec &Hk, dvec &evals)
{
/**
 *  Diagonalization of matrix Hk. Stores eigenvalues in real vector evals and eigenvectors in complex vector Hk
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
    zheev_(&jobz, &uplo, &matsize, &Hk[0], &lda, &evals[0], &work[0], &lwork, &rwork[0], &info);
    assert(!info);
}

char    jobz_eig = 'N';
void diagonalize_eig(cvec &Hk, dvec &evals)
{
/**
 *  Diagonalization of matrix Hk. Stores ONLY eigenvalues in real vector evals
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
    zheev_(&jobz_eig, &uplo, &matsize, &Hk[0], &lda, &evals[0], &work[0], &lwork, &rwork[0], &info);
    assert(!info);
}

int     matsize_F = NATOM*(2*n_max+1);      							// The order of the matrix A.  N >= 0
int     lda_F = NATOM*(2*n_max+1);            							// The leading dimension of the array A.  LDA >= max(1,N)
int     lwork_F = 2*NATOM*(2*n_max+1)-1;      							// The length of the array WORK.  LWORK >= max(1,2*N-1)
double  rwork_F[3*NATOM*(2*n_max+1)-2];       							// dimension (max(1, 3*N-2))
cdouble work_F[2*NATOM*(2*n_max+1)-1];


void diagonalize_F(cvec &Hk_FLOQUET, dvec &evals_FLOQUET)
{
/**
 *  Diagonalization of Floquet matrix H_FLOQUET[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM]
 *  -Hk: Complex vector[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[(2*m_max+1) x NATOM] to store eigenvalues
 */
    zheev_(&jobz, &uplo, &matsize_F, &Hk_FLOQUET[0], &lda_F, &evals_FLOQUET[0], &work_F[0], &lwork_F, &rwork_F[0], &info);
    assert(!info);
}

char JOBVL = 'N';                                                       // don't compute left eigenvectors!
char JOBVR = 'V';                                                       // compute right eigenvectors!
int LDVL = NATOM;
int LDVR = NATOM;
int     lwork_GE = 2*NATOM;
double  rwork_GE[2*NATOM];


void diagonalize_GE(cvec &Hk, cvec &TEMP1, cvec &TEMP2, cvec &evals_c)
{
/**
 *	Diagonalization of NON_Hermitian matrix Hk. Writes eiegenvalues to vector evals_c
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Complex vector[NATOM] to store eigenvalues
 */
    zgeev_(&JOBVL, &JOBVL, &matsize, &Hk[0], &lda, &evals_c[0], &TEMP1[0], &LDVL, &TEMP2[0], &LDVR, &work[0], &lwork_GE, &rwork_GE[0], &info);
    assert(!info);
}
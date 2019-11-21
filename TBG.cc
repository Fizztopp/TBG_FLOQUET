/**
 *	TIGHT-BINDING MODEL FOR TWISTED BILAYER GRAPHENE (TBG)
 *  Copyright (C) 2019, Gabriel E. Topp
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 *  02111-1307, USA.
 * 	
 * 	This code is based on a full unit cell tight-binding model for twisted bilayer graphene. For commensurate angles the follwing objects can be calculated
 *  -filling-dependent chemical potential
 *  -equilibrium bands
 *  -equilibrium Berry curvature (2 different methods)
 *  -Floquet spectrum (2 different methods)
 *  -Berry curvature of Floquet states
 * 
 *  Necessary input:
 *  -Unit_Cell.dat: contains atomic positions, and sublattice index
 *  -k_path.dat: list of k-points along high symmetry path
 *  -k_BZ: List of k-points of Brilluoin zone (for reduce also weights are necessary!)  
 */
 

#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <math.h>
#include <assert.h>
#include <iterator>
#include <sstream>
#include <string>
#include <algorithm>


// PARAMETERS ##########################################################

// intrinsic parameters
// electronic
#define SC        4                                                     // defines super cell (m+1,n) and thus commensurate twist angle
#define NATOM     244                     						        // # atoms (dimension of Hamiltonian)
#define lconst    2.445                                                 // lattice constant (Angstroem)                                        
#define	qq1       3.15													// hopping renormalization 
#define	aa1       1.411621												// intralayer nearest-neigbour distance	
#define	aa2       3.364                                                 // interlayer distance (Angstroem)
#define	t1        -3.24                                                 // hopping parameter of pz-pi (eV)
#define	t2        0.55													// hopping parameter of pz-sigma (eV)
#define BETA      10.0                       					     	// inverse temperature (1/eV)
// additional options
#define RG        1.0                                                   // Fermi renormalization (1. off) <-- magic angle ~1.05 <->  Natom ~13468 <-> v_fermi ~0.0
#define VV        0.0001                                                // symmetric top-gate/back-gate potential (eV)
#define dgap      0.0001                                                // sublattice potential a la Haldane (eV)

// numerical paramters
#define mu_init   0.80											     	// initial guess for chemical potenital -> not necessarily needed for Floquet (eV)
#define dev       1e-9                   					        	// exit deviation for while loop in groundstate() 
#define DELTA     1e-5												    // correction prefactor for chemical potential in groundstate() 

// Peierls driving
#define w_peierls      2.226                                            // Frequency of Applied Field (eV)
#define Ax_peierls     0.05                                             // Amplitude of Applied Field in x-direction
#define Ay_peierls     0.05                                             // Amplitude of Applied Field in y-direction
#define Az_peierls     0.0                                              // Amplitude of Applied Field in z-direction
 
// FLOQUET
#define m_max 2                                                         // order of truncation: m in {-m,...,0,...+m} 
#define n_max 2                                                         // order of truncation: n in {-n,...,0,...+n} (m_max == n_max!) 
#define timesteps_F 2e2                                                 // # of steps to perform integration over one period T=2pi/Omega                             

#define PI 3.14159265359

#define COUNT  0

// CALCULATION OPTIONS #################################################

#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node                                             
    #include <mpi.h>
#endif

#ifndef NO_OMP                                                          // BOTTLENECK: Diagonalization -> can't be parallelized by OpenMP
    #include <omp.h>                                                    // REMEMBER: Shared memory only on same node!
#endif

//#define NO_IC                                                         // switch interlayer coupling on/off 

using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name 
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

cdouble II(0,1);


// DEFINITION OF FUNCTIONS #############################################

//LAPACK (Fortran 90) functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//routine to find eigensystem of Hk
extern "C" {
/** 
 *  Computes the eigenvalues and, optionally, the eigenvectors for a Hermitian matrices H
 */
    void zheev_(char* jobz, char* uplo, int* N, cdouble* H, int* LDA, double* W, cdouble* work, int* lwork, double* rwork, int *info);
}
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


extern "C" {
/**
 *  Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE (non Hermitian) matrix --> used in FloquetEVs
 */ void zgeev_(char* JOBVL, char* JOBVR, int* N, cdouble* A, int* LDA, cdouble* W, cdouble* VL, int* LDVL, cdouble* VR, int* LDVR, cdouble* work, int* lwork, double* rwork, int *info);
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


//INLINE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inline int fq(int i, int j, int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
}


inline int f_FL(int m, int n, int i, int j)
/**
 *	Wrapper for Floquet matrix MAT[m, n, i, j], (2*m_max+1)x(2*n_max+1)xNATOM*NATOM block matrix element where i,j in {0,..,NATOM-1}, m in {-m,...,0,...+m}, n in {-n,...,0,...+n}
 */
{
	return (2*n_max+1)*NATOM*NATOM*m + NATOM*n + (2*n_max+1)*NATOM*i + j;
}


inline double delta(int a, int b)
/**
 *  Delta function
 */
{
	if (a==b)
		return 1.;
	else
		return 0.;
}


template <class Vec>
inline void print(Vec vec)
/**
 *	Print out vector
 */
{
	for(int i=0; i<vec.size(); i++)
		{
	    	cout << vec[i] << " ";
	    }	
	cout << endl;
}


inline double Ax_t(double time)
{
/**
 *	Peierls field for electrons in x-direction:
 *  -time: Real time coordinate
 */
    return Ax_peierls*sin(w_peierls*time);
}


inline double Ay_t(double time)
{
/**
 *	Peierls field for electrons in y-direction:
 *  -time: Real time coordinate
 */
    return Ay_peierls*cos(w_peierls*time);
}


inline double Az_t(double time)
{
/**
 *	Peierls field for electrons in z-direction:
 *  -time: real time coordinate
 */
    return Az_peierls*sin(w_peierls*time);
}


inline double fermi(double energy, double mu)
{
/**
 *	Fermi distribution:
 *	-energy: Energy eigenvalue
 *	-mu: Chemical potential
 */
    return 1./(exp((energy-mu)*BETA) + 1.);
}


// VOID FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void ReadIn(vector<dvec> &MAT, const string& filename)
{
/**
 *	Read in real valued matrix
 */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		dvec row((istream_iterator<double>(is)),
		istream_iterator<double>());
		MAT.push_back(row);
	}
	in.close();
}


template <class Vec>
void times(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product of quadratic matrices: $C = A \cdot B$
 */
{
    int dim = sqrt(A.size());
	Vec TEMP(dim*dim);
    // Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
	    for(int j=0; j<dim; j++) {
		    TEMP[fq(j,i,dim)] = B[fq(i,j,dim)];
		   }
    }
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += A[fq(i,k,dim)]*TEMP[fq(j,k,dim)]; 
			}
		}
	}	
}


template <class Vec>
void times_dn(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of first factor: $C = A^\dagger \cdot B$
 */
{
	int dim = sqrt(A.size());
	Vec TEMP1(dim*dim);
	Vec TEMP2(dim*dim);
	// Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
		for(int j=0; j<dim; j++) {
			TEMP1[fq(j,i,dim)] = A[fq(i,j,dim)];
			TEMP2[fq(j,i,dim)] = B[fq(i,j,dim)];
		}
	}		
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += conj(TEMP1[fq(i,k,dim)])*TEMP2[fq(j,k,dim)];
			}
		}
	}		
}


template <class Vec>
void times_nd(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of second factor: $C = A \cdot B^\dagger$
 */
{
	int dim = sqrt(A.size());	
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
					C[fq(i,j,dim)] += A[fq(i,k,dim)]*conj(B[fq(j,k,dim)]);
			}
		}
	}	
}


void set_Hk0(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL)
/**
 * 	Set eq. Hamiltonian (without external field)
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	const double lcell = lconst*sqrt(pow(lvec[0],2.)+pow(lvec[1],2.));
	const double qq2 = qq1*aa2/aa1 ;
	const double kx = kvec[0];                                          // private to each Thread
	const double ky = kvec[1];                                 
		
    // Bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												// all computed values of private variables are lost after parrallel region!
	{
#endif		
	double d, rx, ry, rz;                                               // declares variable (allocates memory), private (not shared) for each thread
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		Hk[m] = 0.0;
	}	
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		Hk[fq(i,i,NATOM)] = VV/2.;                                      // NO mass term (V/2*sigma_3), just decreases top layer energy by const. electric potential -V/2 (V/2*sigma_0)!
		if (UNIT_CELL[i][3] < 0.9){
                Hk[fq(i,i,NATOM)] += -dgap/2.;
        }
        else{
                Hk[fq(i,i,NATOM)] += dgap/2.;
        }     
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				          
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		Hk[fq(i,i,NATOM)] = -VV/2.;                                     // NO mass term (V/2*sigma_3), just decreases top layer energy by const. electric potential -V/2 (V/2*sigma_0)!
		if (UNIT_CELL[i][3] < 0.9){
			Hk[fq(i,i,NATOM)] += -dgap/2.;
        }
        else{
			Hk[fq(i,i,NATOM)] += dgap/2.;
		}
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				  
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
	// Inter-layer terms
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += (1.-pow(aa2/d,2.))*t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry));       // Vpp_pi term
					Hk[fq(i,j,NATOM)] += pow(aa2/d,2.)*t2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*lconst*rx+ky*lconst*ry));               // Vpp_sigma term
					//  ((1.-pow(aa2/d,2.)) == 0 for vertical hopping (interlayer hopping in AA regions) --> purely governed by Vpp_sigma term  
           
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif						
}


void set_dH0dkx(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL)
/**
 * 	Set derivative along kx of electronic eq. Hamiltonian without Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	const double lcell = lconst*sqrt(pow(lvec[0],2.)+pow(lvec[1],2.));
	const double qq2 = qq1*aa2/aa1 ;
	const double kx = kvec[0];                                          // private to each Thread
	const double ky = kvec[1];                                 
	
    // Bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												// all computed values of private variables are lost after parrallel region!
	{
#endif		
	double d, rx, ry, rz;                                               // declares variable (allocates memory), private (not shared) for each thread
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		Hk[m] = 0.0;
	}		
#ifndef NO_OMP    	
	#pragma omp for                                  					// workload per thread is dynamic (continue in for loop)
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*(II*rx*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				          
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*(II*rx*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				         
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
	// inter-layer terms
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{	
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += ((1.-pow(aa2/d,2.))*t1/RG*exp(qq1*(1.-(d/aa1)))+pow(aa2/d,2.)*t2*exp(qq2*(1.-(d/aa2))))*(II*rx*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));     // [k] = 1/AA				           
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif			
}


void set_dH0dky(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL)
/**
 * 	Set derivative along ky of electronic eq. Hamiltonian without Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	const double lcell = lconst*sqrt(pow(lvec[0],2.)+pow(lvec[1],2.));
	const double qq2 = qq1*aa2/aa1 ;
	const double kx = kvec[0];                                          // private to each Thread
	const double ky = kvec[1];                                 
	
    // Bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												// all computed values of private variables are lost after parrallel region!
	{
#endif		
	double d, rx, ry, rz;                                               // declares variable (allocates memory), private (not shared) for each thread
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		Hk[m] = 0.0;
	}		
#ifndef NO_OMP    	
	#pragma omp for                                  					// workload per thread is dynamic (continue in for loop)
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*(II*ry*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				         
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*(II*ry*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));      // [k] = 1/AA				
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
	// Inter layer 
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{	
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = lconst*AA --> [rx*lconst] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = lconst*AA --> [ry*lconst] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					Hk[fq(i,j,NATOM)] += ((1.-pow(aa2/d,2.))*t1/RG*exp(qq1*(1.-(d/aa1)))+pow(aa2/d,2.)*t2*exp(qq2*(1.-(d/aa2))))*(II*ry*lconst)*exp(II*(kx*lconst*rx+ky*lconst*ry));     // [k] = 1/AA				           
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif		
}


void EQ_BC(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals, dvec &bands_BCs, const string& filename)
/**
 * 	Calculate local Berry curvature in equlibrium for one k-point by Berry's formula using k-derivative of Hamiltonian
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	cvec *S = new cvec(NATOM*NATOM);
	cvec *SCT = new cvec(NATOM*NATOM); 
	cvec *Hdx = new cvec(NATOM*NATOM);
	cvec *Hdy = new cvec(NATOM*NATOM); 
	cdouble temp1, temp2, temp3, temp4;
	cvec vec1(NATOM);
	cvec vec2(NATOM);
	cvec vec3(NATOM);
	cvec vec4(NATOM);
	
	set_Hk0(kvec, S[0], lvec, UNIT_CELL);
	diagonalize(S[0], evals);   

	for(int i=0; i<NATOM; i++)
	{
		for (int j=0; j<NATOM; j++) 
		{
			(*SCT)[fq(i,j,NATOM)] = conj((*S)[fq(j,i,NATOM)]);
        }
	}
	
	set_dH0dkx(kvec, Hdx[0], lvec, UNIT_CELL);
	set_dH0dky(kvec, Hdy[0], lvec, UNIT_CELL);
	
	// Berry curvature of n-th band     
	for(int n=0; n<NATOM; ++n)                                       
	{
		bands_BCs[n] = 0.0;
		for(int m=0; m<NATOM; ++m)
		{
			if(sqrt(pow(evals[n]-evals[m],2.))<1.e-8) continue;
			temp1 = 0.;
			temp2 = 0.; 
			temp3 = 0.;
			temp4 = 0.; 
			if(m==n){ 
				continue;
			}		
			for(int a=0; a<NATOM; ++a)	
			{
				vec1[a] = 0.;
				vec2[a] = 0.;
				vec3[a] = 0.;
				vec4[a] = 0.;
				for(int b=0; b<NATOM; ++b)	
				{
					vec1[a] += (*Hdx)[fq(a,b,NATOM)]*(*SCT)[fq(b,m,NATOM)]; 
					vec2[a] += (*Hdy)[fq(a,b,NATOM)]*(*SCT)[fq(b,n,NATOM)]; 
					vec3[a] += (*Hdy)[fq(a,b,NATOM)]*(*SCT)[fq(b,m,NATOM)]; 
					vec4[a] += (*Hdx)[fq(a,b,NATOM)]*(*SCT)[fq(b,n,NATOM)]; 
				}
			}
			for(int c=0; c<NATOM; ++c)
			{
				temp1 += vec1[c]*(*S)[fq(n,c,NATOM)];
				temp2 += vec2[c]*(*S)[fq(m,c,NATOM)];
				temp3 += vec3[c]*(*S)[fq(n,c,NATOM)];
				temp4 += vec4[c]*(*S)[fq(m,c,NATOM)];
			}		
			bands_BCs[n] += -imag((temp1*temp2 - temp3*temp4)/pow((evals[n]-evals[m]),2.)); // [BC] = AA*AA
		}	
	}
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank==0)
	{
		if(filename!="no_file")
		{
			ofstream myfile (filename);
			if (myfile.is_open())
			{
				for(int n=0; n<NATOM; ++n) 
				{
					myfile << bands_BCs[n] << endl;
				}	
				myfile.close();
			}
			else cout << "Unable to open file" << endl;	
		}
	}	
	delete S;
	delete SCT;  
	delete Hdx;
	delete Hdy;   
}	


void EQ_BC_LOOP(dvec kvec, double kmin, double kmax, int Nk, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals, dvec &bands_BCs, const string& filename)
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
{
	double dk = (kmax-kmin)/double(Nk-1);
	cdouble temp1, temp2, temp3, temp4;
	dvec k0(2);
	vector<cvec*> S_ARRAY(Nk*Nk);                                       
	for(int n=0; n<Nk*Nk; n++)
		S_ARRAY[n] = new cvec(NATOM*NATOM);	
	
	// Set k-point of lower right corner of loop
	k0[0] = kvec[0]-0.5*(kmax-kmin);
	k0[1] = kvec[1]-0.5*(kmax-kmin);
	
	// Calculate eigenvectors of gridpoints along loop
	for(int i=0; i<Nk; i++)
	{
		kvec[0] = k0[0]+i*dk;
		for(int j=0; j<Nk; j++)
		{					
			kvec[1] = k0[1]+j*dk;
			set_Hk0(kvec, S_ARRAY[fq(i,j,Nk)][0], lvec, UNIT_CELL);
			diagonalize(S_ARRAY[fq(i,j,Nk)][0], evals);
		}
	}
	// Calculate Phase around loop
	for(int n=0; n<NATOM; n++)	
	{
		bands_BCs[n] = 0.;
		for(int i=0; i<Nk-1; i++)
		{
			for(int j=0; j<Nk-1; j++)
			{		
				temp1 = 0.;
				temp2 = 0.; 
				temp3 = 0.;
				temp4 = 0.; 
				for(int a=0; a<NATOM; ++a)	
				{
					temp1 += conj((*S_ARRAY[fq(i,j,Nk)])[fq(n,a,NATOM)])*(*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,NATOM)];
					temp2 += conj((*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,NATOM)])*(*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,NATOM)];
					temp3 += conj((*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,NATOM)])*(*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,NATOM)];
					temp4 += conj((*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,NATOM)])*(*S_ARRAY[fq(i,j,Nk)])[fq(n,a,NATOM)];
				}
				bands_BCs[n] += imag(log(temp1*temp2*temp3*temp4))/pow(kmax-kmin,2.);
			}		
		}		
	}	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank==0)
	{	
		if(filename!="no_file")
		{	
			ofstream myfile (filename);
			if (myfile.is_open())
			{
				for(int n=0; n<NATOM; ++n) 
				{
					// Berry curvature equal to phase diveded by area of loop
					myfile << bands_BCs[n] << endl;
				}	
				myfile.close();
			}
			else cout << "Unable to open file" << endl;	
		}
	}	
	for(int n=0; n<Nk*Nk; n++)
	{                            
		delete S_ARRAY[n];
	}	
}	


void EQ_BC_LOOP_PATH(double kmin, double kmax, int Nk, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals, dvec &bands_BCs, int &numprocs, int &myrank)
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
{
	int num_kpoints = K_PATH.size();
	dvec BC_ARRAY(num_kpoints*NATOM);                                       
	
	for(int k=myrank; k<num_kpoints; k+=numprocs)
	{
		EQ_BC_LOOP(K_PATH[k], kmin, kmax, Nk, Hk, lvec, UNIT_CELL, evals, bands_BCs, "no_file");
		for(int n=0; n<NATOM; ++n) 
		{
			BC_ARRAY[fq(k,n,NATOM)] = bands_BCs[n];
		}	
	}	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BC_ARRAY[0], NATOM*num_kpoints, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{	
		ofstream myfile ("EQ_BC_LOOP_PATH.dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints; ++k)
			{
				for(int n=0; n<NATOM; ++n) 
				{
					myfile << BC_ARRAY[fq(k,n,NATOM)]/pow(kmax-kmin,2.) << " ";
				}	
				myfile << endl;
			}
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	}
}	


void EQ_BC_LOOP_PATH_CHECK(cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals, dvec &bands_BCs, int &numprocs, int &myrank)
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
{
	int num_kpoints = K_PATH.size();
	dvec BC_ARRAY(num_kpoints*NATOM);                                       
	
	for(int k=myrank; k<num_kpoints; k+=numprocs)
	{
		EQ_BC(K_PATH[k], Hk, lvec, UNIT_CELL, evals, bands_BCs, "no_file");
		for(int n=0; n<NATOM; ++n) 
		{
			BC_ARRAY[fq(k,n,NATOM)] = bands_BCs[n];
		}	
	}	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BC_ARRAY[0], NATOM*num_kpoints, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{	
		ofstream myfile ("EQ_BC_LOOP_PATH_CHECK.dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints; ++k)
			{
				for(int n=0; n<NATOM; ++n) 
				{
					myfile << BC_ARRAY[fq(k,n,NATOM)] << " ";
				}	
				myfile << endl;
			}
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	}
}	


void set_Hk(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL, double time)
/**
 *	Set time-dependent Hamiltonian matrix with Peierls field
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM*NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -time: time variable
 */
{
	const double lcell = lconst*sqrt(pow(lvec[0],2.)+pow(lvec[1],2.));
	const double qq2 = qq1*aa2/aa1 ;
	const double kx = kvec[0];                                          
	const double ky = kvec[1];                                 
		
    // Bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												
	{
#endif		
	double d, rx, ry, rz;                                               
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		 Hk[m] = 0.0;
	 }	
#ifndef NO_OMP    	
	#pragma omp for                                  					
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		// Back-gate voltage
		Hk[fq(i,i,NATOM)] = VV/2.;  
		// Sublattice potential                                    
		if (UNIT_CELL[i][3] < 0.9){
			Hk[fq(i,i,NATOM)] += -dgap/2.;
        }
        else{
            Hk[fq(i,i,NATOM)] += dgap/2.;
        }   
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					//[rx] = lconst*Angstroem --> [rx*lconst] = Angstroem 
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 	
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];
					//[dd] = Angstroem  	                                            
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry))*exp(-II*(Ax_t(time)*rx+Ay_t(time)*ry));      			       
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		// Top-gate voltage
		Hk[fq(i,i,NATOM)] = -VV/2.;
		// Sublattice potential                                     
		if (UNIT_CELL[i][3] < 0.9){
			Hk[fq(i,i,NATOM)] += -dgap/2.;
        }
        else{
            Hk[fq(i,i,NATOM)] += dgap/2.;
        }   		
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 	
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                      
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    
					Hk[fq(i,j,NATOM)] += t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry))*exp(-II*(Ax_t(time)*rx+Ay_t(time)*ry));    		         
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
	// Inter-layer terms
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                           
	                d = lconst*sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));   
					// Vpp_pi term
					Hk[fq(i,j,NATOM)] += (1.-pow(aa2/d,2.))*t1/RG*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*lconst*rx+ky*lconst*ry))*exp(-II*(Ax_t(time)*rx+Ay_t(time)*ry+Az_t(time)*rz));
					// Vpp_sigma term       
					Hk[fq(i,j,NATOM)] += pow(aa2/d,2.)*t2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*lconst*rx+ky*lconst*ry))*exp(-II*(Ax_t(time)*rx+Ay_t(time)*ry+Az_t(time)*rz));           
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif					
}


void groundstate(cvec &Hk, dvec &evals, vector<dvec> &kweights, vector<dvec> &BZ_IRR, vector<dvec> &UNIT_CELL, const dvec &lvec, double &mu, int &numprocs, int &myrank)
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
{	
	const int num_kpoints_BZ = BZ_IRR.size();  
	
	int count = 0;                                                      
	double N_tot;	
	double mu_old;    
	double deviation = 1.0;
	mu = mu_init;
	double num_kpoints_BZ_full = 0.;
	dvec EVALS(num_kpoints_BZ*NATOM);                                   
	
	for(int k=0; k<kweights.size(); k++)
		num_kpoints_BZ_full += kweights[k][0];							
	
	for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
	{		
		set_Hk0(BZ_IRR[k], Hk, lvec, UNIT_CELL);
		diagonalize_eig(Hk, evals);                                 	
		for(int i=0; i<NATOM; i++)
		{			
			EVALS[fq(k,i,NATOM)] = evals[i];	
		}
	}
	
	while(deviation > dev)
	{
		count++;
					
		mu_old = mu;	
	    
		N_tot = 0.;
		for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
		{		
			for(int i=0; i<NATOM; i++)
			{			
				N_tot +=  fermi(EVALS[fq(k,i,NATOM)], mu)*kweights[k][0];	
			}
		}

#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &N_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif		
		mu += -DELTA*(N_tot-NATOM/2.*double(num_kpoints_BZ_full));	                       
		
		deviation = abs(mu-mu_old);
		if(myrank==0){
			cout << "loop #" << count << ": deviation = " << deviation << endl;
			cout << "chemical potential mu = " << mu << endl;
		}	
	}
	if(myrank==0)
	{
		ofstream myfile ("mu.dat");
		if (myfile.is_open())
		{
			myfile << mu;
			myfile.close();
		}	
	else cout << "Unable to open file" << endl;	
	}
}

	
void Hk_bands(dvec &BANDS, cvec &Hk, dvec &evals, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, const string& filename, int &numprocs, int &myrank)
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
{
	const int num_kpoints_path = K_PATH.size();

	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{
		set_Hk0(K_PATH[k], Hk, lvec, UNIT_CELL);
		diagonalize_eig(Hk, evals);
		for(int m=0; m<NATOM; m++)
			BANDS[fq(k, m, NATOM)] = evals[m];
	}
#ifndef NO_MPI	
	MPI_Allreduce(MPI_IN_PLACE, &BANDS[0], num_kpoints_path*NATOM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{
		ofstream myfile (filename);
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<NATOM; m++)
				{
					myfile << BANDS[fq(k, m, NATOM)] << " " ;
				}
				myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}	
}


void Hk_bands_Floquet(dvec &BANDS_FLOQUET, dvec &OVERLAP_FLOQUET, cvec &Hk_FLOQUET, dvec &evals_FLOQUET, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, int &numprocs, int &myrank)
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
{
	const int num_kpoints_path = K_PATH.size();
	const double T = 2.*M_PI/w_peierls;
	const double dt = T/double(timesteps_F-1);
	
	cvec *TEMP1 = new cvec(NATOM*NATOM);
	cvec *TEMP2 = new cvec(NATOM*NATOM); 
	double temp; 
	
	cvec H0(NATOM*NATOM);
	dvec evals(NATOM);
	cdouble tempc; 
	
	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{
#ifndef NO_OMP    	
	#pragma omp parallel for collapse(4)                                
#endif						                
		for(int m=-m_max; m<m_max+1; m++)
		{
			for(int n=-n_max; n<n_max+1; n++)
			{					
				for(int i=0; i<NATOM; i++)
				{
					for(int j=0; j<NATOM; j++)
					{
						Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] = 0.0;
					}
				}
			}
		}												
		if(myrank==0) cout << endl; 
		if(myrank==0) cout << "k = " << k << endl;	
		// Perform integration over one period T
		for(double t=0; t<T-dt/2.; t+=dt)
		{	
			if(myrank==0) cout << "time step: " << t/dt <<  endl;
			
			set_Hk(K_PATH[k], TEMP1[0], lvec, UNIT_CELL, t);
			set_Hk(K_PATH[k], TEMP2[0], lvec, UNIT_CELL, t+dt);								
			for(int m=-m_max; m<m_max+1; m++)
			{
				for(int n=-n_max; n<n_max+1; n++)
				{		
#ifndef NO_OMP    	
			#pragma omp parallel for                   
#endif								
					for(int i=0; i<NATOM; i++)
					{
						for(int j=0; j<NATOM; j++)
						{
							Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] += 0.5/T*(exp(II*w_peierls*double(m-n)*t)*(*TEMP1)[fq(i,j,NATOM)] + exp(II*w_peierls*double(m-n)*(t+dt))*(*TEMP2)[fq(i,j,NATOM)])*dt + double(m)*w_peierls*delta(i,j)*delta(m,n)/double(timesteps_F-1);
						}
					}				
				}
			}
		}
		// Diagonalize Floquet Hamiltonian in order to get eigenvalues and eigenvectors		
		diagonalize_F(Hk_FLOQUET, evals_FLOQUET);  		
		for(int jj=0; jj<NATOM*(2*n_max+1); jj++)
		{
			BANDS_FLOQUET[fq(k,jj,NATOM*(2*n_max+1))] = evals_FLOQUET[jj];
		}	
		// Calculate squared overlap of Floquet eigenstates with eigenstates of eq. Hamiltonian
		set_Hk0(K_PATH[k], H0, lvec, UNIT_CELL);
		diagonalize(H0, evals);
		for(int i=0; i<NATOM*(2*n_max+1); ++i)
		{
			temp = 0.;
			for(int w=0; w<NATOM; ++w)
			{
				tempc = 0.;
				for(int j=0; j<NATOM; ++j)
				{
					tempc += Hk_FLOQUET[fq(i,NATOM*n_max-1+j,NATOM*(2*n_max+1))]*conj(H0[fq(w,j,NATOM)]);
				}
				temp += real(tempc*conj(tempc));
			}	
			OVERLAP_FLOQUET[fq(k,i,NATOM*(2*n_max+1))] = temp; 
		}	  				
	}
	delete TEMP1, TEMP2;	
	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BANDS_FLOQUET[0], NATOM*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &OVERLAP_FLOQUET[0], NATOM*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	// Store data
	if(myrank==0)
	{
		ofstream myfile ("bands_floquet.dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<NATOM*(2*n_max+1); m++)
				{
					myfile << BANDS_FLOQUET[fq(k,m,NATOM*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
	if(myrank==0)
	{
		ofstream myfile ("overlap_floquet.dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<NATOM*(2*n_max+1); m++)
				{
					myfile << OVERLAP_FLOQUET[fq(k,m,NATOM*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
}


void FloquetEVs(dvec &BANDS, cvec &Hk, dvec &evals, cvec &evals_c, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, double &mu, int &numprocs, int &myrank)
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
{
	const int num_kpoints_path = K_PATH.size();
	const double T = 2.*M_PI/w_peierls;
	const double dt = T/double(timesteps_F);
	
	cvec *TEMP1 = new cvec(NATOM*NATOM);
	cvec *TEMP2 = new cvec(NATOM*NATOM); 
	cvec *TEMP3 = new cvec(NATOM*NATOM); 
	cvec *TEMP4 = new cvec(NATOM*NATOM); 
	
	cvec *temp3, *temp4;
	
	for(int k=0; k<num_kpoints_path; k++)
		for(int m=0; m<NATOM; m++)
		{
			BANDS[fq(k, m, NATOM)] = 0.0;
		}
	
	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{		
		for(int i=0; i<NATOM; i++)
		{
			for(int j=0; j<NATOM; j++)
			{
				(*TEMP4)[fq(i,j,NATOM)] = delta(i,j);                   // Identity matrix
			}
		}		                												
		if(myrank==0) cout << endl << "k = " << k << endl;	
		
		// Calculate time ordered Floqeut evolution operator
		for(double t=0; t<T-dt/2; t+=dt)
		{	
			if(myrank==0) cout << "time step: " << t/dt <<  endl;
			
			set_Hk(K_PATH[k], TEMP1[0], lvec, UNIT_CELL, t);
			set_Hk(K_PATH[k], TEMP2[0], lvec, UNIT_CELL, t+dt);			
				
			for(int i=0; i<NATOM; i++)
			{
				for(int j=0; j<NATOM; j++)
				{
					Hk[fq(i,j,NATOM)] = 0.5*((*TEMP1)[fq(i,j,NATOM)] + (*TEMP2)[fq(i,j,NATOM)]);
				}
			}
			
			// Calculate Floqeut operater U[dt]=Exp[...] by diagonalization and back transformation
			diagonalize(Hk, evals);                                     
			
			for(int i=0; i<NATOM; i++)
			{
				for(int j=0; j<NATOM; j++)
				{
					(*TEMP1)[fq(i,j,NATOM)] = exp(-II*evals[i]*dt)*delta(i,j);
				}
			}		
			
			times(TEMP1[0], Hk, TEMP2[0]);                              
			times_dn(Hk, TEMP2[0], TEMP1[0]);	

			times(TEMP1[0], TEMP4[0], TEMP3[0]);                        // |Psi(t)> = U(t-dt)....U(dt)|Psi(0)> --> small times to the right!  
			temp3 = TEMP3;
			temp4 = TEMP4;
			TEMP4 = temp3;		
			TEMP3 = temp4;
		}	
		// Calculate eigenvalues of effective {Floqeut} Hamiltonaian
		cout << "Start Final Diagonalization ---------------------------------------------------------------------------------------------" << endl;
		diagonalize_GE(TEMP4[0], TEMP1[0], TEMP2[0], evals_c);                               // evals (should be) purley imaginary!!!
		for(int m=0; m<NATOM; m++)
		{
			BANDS[fq(k, m, NATOM)] = real(II*log(evals_c[m])/T);
			if(imag(II*log(evals_c[m]))>1e-10)
			{
				cout << "Nonvanishing imaginary part!!! ---------------------------------------------------------------------------------------------" << endl;
				cout << "Im(epsilon) = " << imag(II*log(evals_c[m])/T) << endl;
			}	
		}	
	}
	delete TEMP1, TEMP2, TEMP3, TEMP4;	
	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BANDS[0], num_kpoints_path*NATOM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{
		ofstream myfile ("bands_strob.dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<NATOM; m++)
				{
					myfile << BANDS[fq(k, m, NATOM)] << " " ;
				}
				myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}	
}


void Set_Hk_Floquet(dvec kvec, cvec &Hk_FLOQUET, vector<dvec> &UNIT_CELL, const dvec &lvec)
/**
 *	Set Floquet Hamiltonian in k-orbital basis for use in FLOQUET_BC_LOOP()
 * 	-kvec: Real vector of the reciprocal space
 *  -Hk_FLOQUET: Complex vector[(2*m_max+1)x(2*n_max+1)xNATOMxNATOM] to store Flqoeut Hamiltonian matrix
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 */
{
	const double T = 2.*M_PI/w_peierls;
	const double dt = T/double(timesteps_F);
	
	cvec *TEMP1 = new cvec(NATOM*NATOM);
	cvec *TEMP2 = new cvec(NATOM*NATOM); 
	double temp; 
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
#ifndef NO_OMP    	
	#pragma omp parallel for collapse(4)                                // PERFEKTLY nested loops are collapsed into one loop
#endif						                
		for(int m=-m_max; m<m_max+1; m++)
		{
			for(int n=-n_max; n<n_max+1; n++)
			{					
				for(int i=0; i<NATOM; i++)
				{
					for(int j=0; j<NATOM; j++)
					{
						Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] = 0.0;
					}
				}
			}
		}												
		for(double t=0; t<T-dt/2.; t+=dt)
		{	
			if(rank==0) cout << "time step: " << t/dt <<  endl;
			set_Hk(kvec, TEMP1[0], lvec, UNIT_CELL, t);
			set_Hk(kvec, TEMP2[0], lvec, UNIT_CELL, t+dt);								
			for(int m=-m_max; m<m_max+1; m++)
			{
				for(int n=-n_max; n<n_max+1; n++)
				{		
#ifndef NO_OMP    	
			#pragma omp parallel for                   
#endif								
					for(int i=0; i<NATOM; i++)
					{
						for(int j=0; j<NATOM; j++)
						{
							Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] += 0.5/T*(exp(II*w_peierls*double(m-n)*t)*(*TEMP1)[fq(i,j,NATOM)] + exp(II*w_peierls*double(m-n)*(t+dt))*(*TEMP2)[fq(i,j,NATOM)])*dt + double(m)*w_peierls*delta(i,j)*delta(m,n)/double(timesteps_F);
						}
					}				
				}
			}
		}		 				
	delete TEMP1, TEMP2;	
}


void FLOQUET_BC_LOOP(dvec kvec, double kmin, double kmax, int Nk, const dvec &lvec, vector<dvec> &UNIT_CELL, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, const string &filename)
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
{
	double dk = (kmax-kmin)/double(Nk-1);
	double temp;
	cdouble temp1, temp2, temp3, temp4;
	dvec k0(2);	 
	vector<cvec*> S_ARRAY(Nk*Nk);                                       
	for(int n=0; n<Nk*Nk; n++)
		S_ARRAY[n] = new cvec((2*m_max+1)*(2*n_max+1)*NATOM*NATOM);	
	
	// Set k-point of lower right corner of loop
	k0[0] = kvec[0]-0.5*(kmax-kmin);
	k0[1] = kvec[1]-0.5*(kmax-kmin);
	
	// Calculate eigenvectors of gridpoints along loop
	for(int i=0; i<Nk; i++)
	{
		kvec[0] = k0[0]+i*dk;
		for(int j=0; j<Nk; j++)
		{					
			kvec[1] = k0[1]+j*dk;
			Set_Hk_Floquet(kvec, S_ARRAY[fq(i,j,Nk)][0], UNIT_CELL, lvec);
			diagonalize_F(S_ARRAY[fq(i,j,Nk)][0], evals_FLOQUET);	
		}
	}
	// Calculate Phase around loop
	for(int n=0; n<(2*n_max+1)*NATOM; n++)	
	{
		bands_BCs_FLOQUET[n] = 0.;
		for(int i=0; i<Nk-1; i++)
		{
			for(int j=0; j<Nk-1; j++)
			{		
				temp1 = 0.;
				temp2 = 0.; 
				temp3 = 0.;
				temp4 = 0.; 
				for(int a=0; a<(2*n_max+1)*NATOM; ++a)	
				{
					temp1 += conj((*S_ARRAY[fq(i,j,Nk)])[fq(n,a,(2*n_max+1)*NATOM)])*(*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,(2*n_max+1)*NATOM)];
					temp2 += conj((*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,(2*n_max+1)*NATOM)])*(*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,(2*n_max+1)*NATOM)];
					temp3 += conj((*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,(2*n_max+1)*NATOM)])*(*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,(2*n_max+1)*NATOM)];
					temp4 += conj((*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,(2*n_max+1)*NATOM)])*(*S_ARRAY[fq(i,j,Nk)])[fq(n,a,(2*n_max+1)*NATOM)];
				}
				bands_BCs_FLOQUET[n] += imag(log(temp1*temp2*temp3*temp4));
			}		
		}	
	}		
	if(filename!="no_file")
	{
		ofstream myfile1 (filename);
		if (myfile1.is_open())
		{
			for(int n=0; n<NATOM*(2*n_max+1); ++n) 
			{
				//  Berry curvature equal to phase diveded by area of loop
				myfile1 << bands_BCs_FLOQUET[n]/pow(kmax-kmin,2.) << endl;
			}	
			myfile1.close();
		}
		else cout << "Unable to open file" << endl;	
	}
	for(int n=0; n<Nk*Nk; n++)
	{                            
		delete S_ARRAY[n];
	}	
}	


void FLOQUET_BC_LOOP_PATH(double kmin, double kmax, int Nk, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<dvec> &K_PATH, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, int &numprocs, int &myrank)
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
{
	int num_kpoints = K_PATH.size();
	dvec BC_ARRAY(num_kpoints*NATOM*(2*m_max+1));                                       
	
	for(int k=myrank; k<num_kpoints; k+=numprocs)
	{
		FLOQUET_BC_LOOP(K_PATH[k], kmin, kmax, Nk, lvec, UNIT_CELL, evals_FLOQUET, bands_BCs_FLOQUET, "no_file");
		for(int n=0; n<NATOM*(2*m_max+1); ++n) 
		{
			BC_ARRAY[fq(k,n,NATOM*(2*m_max+1))] = bands_BCs_FLOQUET[n];
		}	
	}	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BC_ARRAY[0], NATOM*num_kpoints*(2*m_max+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{	
		ofstream myfile1 ("FLOQUET_BC_LOOP_PATH.dat");
		if (myfile1.is_open())
		{
			for(int k=0; k<num_kpoints; ++k)
			{
				for(int n=0; n<NATOM*(2*m_max+1); ++n) 
				{
					myfile1 << BC_ARRAY[fq(k,n,NATOM*(2*m_max+1))]/pow(kmax-kmin,2.) << " ";
				}	
				myfile1 << endl;
			}
			myfile1.close();
		}
		else cout << "Unable to open file" << endl;	
	}
}	


//  main() function #####################################################

int main(int argc, char * argv[])
{
    //************** MPI INIT ***************************
  	int numprocs=1, myrank=0, namelen;
    
#ifndef NO_MPI
  	char processor_name[MPI_MAX_PROCESSOR_NAME];
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	MPI_Get_processor_name(processor_name, &namelen);
    
	cout << "Process " << myrank << " on " << processor_name << " out of " << numprocs << " says hello." << endl;
	MPI_Barrier(MPI_COMM_WORLD);
    
#endif
	if(myrank==0) cout << "\n\tProgram running on " << numprocs << " processors." << endl;

	//************** OPEN_MP INIT **************************************
#ifndef NO_OMP 	  
	cout << "# of processes " << omp_get_num_procs() << endl;
#pragma omp parallel 
	cout << "Thread " << omp_get_thread_num() << " out of " << omp_get_num_threads() << " says hello!" << endl;     
#endif
	//******************************************************************
   
	// DECLARATIONS AND INTITALIZATIONS
	const int a = SC+1;
	const int b = SC;

	if(NATOM != 4*(SC*SC+(SC+1)*SC+(SC+1)*(SC+1)))
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
	// 1st angle   
	const double angle1 = atan2(double(b)*sqrt(3.)/2.,double(a)+double(b)/2.) ;
	if(myrank==0) cout << "agle1: " << angle1 << endl;
	// 2nd angle       
	const double angle2 = angle1 + PI/3. ;                                          
	if(myrank==0) cout << "agle2: " << angle2 << endl;
	
	// side length of super cell
	const double d = sqrt(double(b*b)*3./4.+pow(double(a)+double(b)/2.,2.));
	if(myrank==0) cout << "d: " << d << endl;
	
	// superlattice bravis translational vectors
	const dvec lvec = {d*cos(angle1),  d*sin(angle1), d*sin(PI/6.-angle1), d*cos(PI/6.-angle1)};
	
	// chemical potential
	double mu = 0.0;
	
	//Read in atomic positions
	vector<dvec> UNIT_CELL;
	ReadIn(UNIT_CELL, "Unit_Cell.dat");
	if(myrank==0) cout << "Unit_Cell.dat --> " <<  UNIT_CELL.size() << " points" << endl;
	if(NATOM != UNIT_CELL.size())
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
	//Read in vector of k-points
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "k_path.dat");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	// irr. BZ
	//vector of weights
	vector<dvec> kweights_irr;
	ReadIn(kweights_irr, "k_weights_irr.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "k_BZ_irr.dat");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();
	
    // full BZ
	//vector of weights
	vector<dvec> kweights_full;
	ReadIn(kweights_full, "k_weights_full.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_FULL;
	ReadIn(BZ_FULL, "k_BZ_full.dat");
	if(myrank==0) cout << "full BZ --> " << BZ_FULL.size() << " points" << endl;
	int num_kpoints_BZ_full = BZ_FULL.size();
	
	// vector for eigenvalues
	dvec evals(NATOM);
	cvec evals_c(NATOM);
	
	// bands 
	dvec BANDS(num_kpoints_PATH*NATOM);

	// vector for Hamiltonian Hk
	cvec *Hk = new cvec(NATOM*NATOM);
		
	// Berry Curvature
	dvec bands_BCs(NATOM);	
	dvec bands_BCs_FLOQUET(NATOM*(2*m_max+1));	
	 
 	// vector to store Floquet matrix
    cvec *Hk_FLOQUET = new cvec((2*m_max+1)*(2*n_max+1)*NATOM*NATOM);    
	
	// vector for eigenvalues
	dvec *evals_FLOQUET = new dvec(NATOM*(2*n_max+1));
	
	// bands 
	dvec *BANDS_FLOQUET = new dvec(num_kpoints_PATH*NATOM*(2*n_max+1));
	dvec *OVERLAP_FLOQUET = new dvec(num_kpoints_PATH*NATOM*(2*n_max+1));

	// CALCULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	const clock_t begin_time = clock();                                 // time summed over all threads
#ifndef NO_OMP	 
	double dtime = omp_get_wtime();	                                    // time per core
#endif

	
	if(myrank==0){cout << "Start caluclation of chemical potential" << endl;}
	groundstate(Hk[0], evals, kweights_full, BZ_FULL, UNIT_CELL, lvec, mu, numprocs, myrank);

	if(myrank==0){cout << "Start caluclation of equilibrium bands" << endl;}
	Hk_bands(BANDS, Hk[0], evals, K_PATH, UNIT_CELL, lvec, "bands.dat", numprocs, myrank);

	if(myrank==0){cout << "Start caluclation of equilibrium Berry curvature along k-path" << endl;}	
	EQ_BC_LOOP_PATH(-1e-5, +1e-5, 2, Hk[0], lvec, UNIT_CELL, K_PATH, evals, bands_BCs, numprocs, myrank);
	
	if(myrank==0){cout << "Start caluclation of equilibrium Berry curvature along k-path (CHECK)" << endl;}		
	EQ_BC_LOOP_PATH_CHECK(Hk[0], lvec, UNIT_CELL, K_PATH, evals, bands_BCs, numprocs, myrank);
	
	if(myrank==0){cout << "Start caluclation of Floqeut bands along k-path" << endl;}		
	Hk_bands_Floquet(BANDS_FLOQUET[0], OVERLAP_FLOQUET[0], Hk_FLOQUET[0], evals_FLOQUET[0], K_PATH, UNIT_CELL, lvec, numprocs, myrank);
	
	if(myrank==0){cout << "Start caluclation of Floqeut Berry curvature along k-path" << endl;}	
	FLOQUET_BC_LOOP_PATH( -1e-5, +1e-5, 2, lvec, UNIT_CELL, K_PATH, evals_FLOQUET[0], bands_BCs_FLOQUET, numprocs, myrank);
	
	if(myrank==0){cout << "Start caluclation of Floqeut bands along k-path (CHECK)" << endl;}		
	FloquetEVs(BANDS, Hk[0], evals, evals_c, K_PATH, UNIT_CELL, lvec, mu, numprocs, myrank);

	if(myrank==0)
	{ 
		cout << "Total caluclation time (MPI): " << float(clock() - begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
#ifndef NO_OMP	
		dtime = omp_get_wtime() - dtime;
		cout << "Total caluclation time (OMP): " << dtime << " seconds" << endl; 
#endif	
	}
	
#ifndef NO_MPI
	MPI_Finalize();
#endif	

// free memory	
	delete Hk;
	delete Hk_FLOQUET;
	delete evals_FLOQUET;
	delete BANDS_FLOQUET;
	delete OVERLAP_FLOQUET;
}



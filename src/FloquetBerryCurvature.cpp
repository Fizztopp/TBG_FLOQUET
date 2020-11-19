#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <complex>
#include <fstream>

#include "Constants.h"
#include "InlineFunctions.h"
#include "Diagonalization.h"
#include "Hk.h"


#ifndef NO_MPI                                 //REMEMBER: Each Process has its own copy of all allocated memory! --> node
#include <mpi.h>
#endif

using namespace std;

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


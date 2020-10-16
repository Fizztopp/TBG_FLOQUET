#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include <string>


#include "Constants.h"
#include "InlineFunctions.h"
#include "Diagonalization.h"
#include "Hk0.h"

#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node
#include <mpi.h>
#endif

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
    int rank = 0;
#ifndef NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

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
    int rank=0;

#ifndef NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

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



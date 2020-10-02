#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>

#include "Constants.h"
#include "InlineFunctions.h"
#include "Diagonalization.h"

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


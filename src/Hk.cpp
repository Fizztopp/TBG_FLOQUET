#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>


#include "Constants.h"
#include "InlineFunctions.h"
#include "Diagonalization.h"
#include "Hk0.h"
#include "MatrixMultiplication.h"

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
    int rank=0;
#ifndef NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

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

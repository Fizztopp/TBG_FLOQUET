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
 *  -Data/Unit_Cell.dat: contains atomic positions, and sublattice index
 *  -Data/k_path.dat: list of k-points along high symmetry path
 *  -k_BZ: List of k-points of Brilluoin zone (for reduce also weights are necessary!)  
 */
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>
#include <iterator>
#include <string>

// PARAMETERS ##########################################################
#include "Constants.h"

// CALCULATION OPTIONS #################################################
#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node
    #include <mpi.h>
#endif

#ifndef NO_OMP                                                          // BOTTLENECK: Diagonalization -> can't be parallelized by OpenMP
    #include <omp.h>                                                    // REMEMBER: Shared memory only on same node!
#endif

#include "InlineFunctions.h"
#include "MatrixMultiplication.h"
#include "Diagonalization.h"
#include "Hk0.h"
#include "FloquetBerryCurvature.h"
#include "FileHandling.h"
#include "Hk.h"


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
	ReadIn(UNIT_CELL, "Data/Unit_Cell.dat");
	if(myrank==0) cout << "Data/Unit_Cell.dat --> " <<  UNIT_CELL.size() << " points" << endl;
	if(NATOM != UNIT_CELL.size())
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
	//Read in vector of k-points
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "Data/k_path.dat");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	// irr. BZ
	//vector of weights
	vector<dvec> kweights_irr;
	ReadIn(kweights_irr, "Data/k_weights_irr.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "Data/k_BZ_irr.dat");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();
	
    // full BZ
	//vector of weights
	vector<dvec> kweights_full;
	ReadIn(kweights_full, "Data/k_weights_full.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_FULL;
	ReadIn(BZ_FULL, "Data/k_BZ_full.dat");
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

	
//	if(myrank==0){cout << "Start caluclation of chemical potential" << endl;}
//	groundstate(Hk[0], evals, kweights_full, BZ_FULL, UNIT_CELL, lvec, mu, numprocs, myrank);

	if(myrank==0){cout << "Start caluclation of equilibrium bands" << endl;}
	Hk_bands(BANDS, Hk[0], evals, K_PATH, UNIT_CELL, lvec, "Data/bands.dat", numprocs, myrank);

//	if(myrank==0){cout << "Start caluclation of equilibrium Berry curvature along k-path" << endl;}
//	EQ_BC_LOOP_PATH(-1e-5, +1e-5, 2, Hk[0], lvec, UNIT_CELL, K_PATH, evals, bands_BCs, numprocs, myrank);
	
//	if(myrank==0){cout << "Start caluclation of equilibrium Berry curvature along k-path (CHECK)" << endl;}
//	EQ_BC_LOOP_PATH_CHECK(Hk[0], lvec, UNIT_CELL, K_PATH, evals, bands_BCs, numprocs, myrank);
//
//	if(myrank==0){cout << "Start caluclation of Floqeut bands along k-path" << endl;}
//	Hk_bands_Floquet(BANDS_FLOQUET[0], OVERLAP_FLOQUET[0], Hk_FLOQUET[0], evals_FLOQUET[0], K_PATH, UNIT_CELL, lvec, numprocs, myrank);
//
//	if(myrank==0){cout << "Start caluclation of Floqeut Berry curvature along k-path" << endl;}
//	FLOQUET_BC_LOOP_PATH( -1e-5, +1e-5, 2, lvec, UNIT_CELL, K_PATH, evals_FLOQUET[0], bands_BCs_FLOQUET, numprocs, myrank);
//
//	if(myrank==0){cout << "Start caluclation of Floqeut bands along k-path (CHECK)" << endl;}
//	FloquetEVs(BANDS, Hk[0], evals, evals_c, K_PATH, UNIT_CELL, lvec, mu, numprocs, myrank);

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



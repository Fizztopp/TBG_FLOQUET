#ifndef TBG_FLOQUET_CONSTANTS_H
#define TBG_FLOQUET_CONSTANTS_H

#include <complex>
#include <cmath>

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

//#define NO_IC                                                         // switch interlayer coupling on/off

using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

#define MKL_Complex16 cdouble

constexpr cdouble II(0,1);

namespace cavityConstants{

    constexpr double g(0.05);
    const std::vector<double> eA{1./sqrt(2.), 1./sqrt(2.), 0.0};
}


#endif //TBG_FLOQUET_CONSTANTS_H

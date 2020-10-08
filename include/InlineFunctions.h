#ifndef TBG_INLINEFUNCTIONS_H
#define TBG_INLINEFUNCTIONS_H

#include "Constants.h"

inline int fq(const int i,const int j,const int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
}


inline int f_FL(const int m,const int n,const int i,const int j)
/**
 *	Wrapper for Floquet matrix MAT[m, n, i, j], (2*m_max+1)x(2*n_max+1)xNATOM*NATOM block matrix element where i,j in {0,..,NATOM-1}, m in {-m,...,0,...+m}, n in {-n,...,0,...+n}
 */
{
    return (2*n_max+1)*NATOM*NATOM*m + NATOM*n + (2*n_max+1)*NATOM*i + j;
}


inline double delta(const int a,const int b)
/**
 *  Delta function
 */
{
    return double(not((a-b)*(a-b)));
}


template <class Vec>
inline void print(const Vec vec)
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


inline double Ax_t(const double time)
{
/**
 *	Peierls field for electrons in x-direction:
 *  -time: Real time coordinate
 */
    return Ax_peierls*sin(w_peierls*time);
}


inline double Ay_t(const double time)
{
/**
 *	Peierls field for electrons in y-direction:
 *  -time: Real time coordinate
 */
    return Ay_peierls*cos(w_peierls*time);
}


inline double Az_t(const double time)
{
/**
 *	Peierls field for electrons in z-direction:
 *  -time: real time coordinate
 */
    return Az_peierls*sin(w_peierls*time);
}


inline double fermi(const double energy, const double mu)
{
/**
 *	Fermi distribution:
 *	-energy: Energy eigenvalue
 *	-mu: Chemical potential
 */
    return 1./(exp((energy-mu)*BETA) + 1.);
}

#endif //TBG_INLINEFUNCTIONS_H

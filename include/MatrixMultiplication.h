#ifndef TBG_MATRIXMULTIPLICATION_H
#define TBG_MATRIXMULTIPLICATION_H

#include "InlineFunctions.h"

template <class Vec>
void times(const Vec &A, const Vec &B, Vec &C)
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
void times_dn(const Vec &A, const Vec &B, Vec &C)
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
void times_nd(const Vec &A, const Vec &B, Vec &C)
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

#endif //TBG_MATRIXMULTIPLICATION_H

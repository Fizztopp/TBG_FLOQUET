#include <vector>
#include <complex>

#include "gtest/gtest.h"
#include "Constants.h"
#include "HkA.h"
#include "Hk0.h"
#include "InlineFunctions.h"
#include "mkl.h"

#include "TestInitialization.h"

TEST(LinATest, diagonalizationOfHk0) {

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    std::vector<std::complex<double>> Hk0(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> eVecs(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<double> evalsHk0(NATOM, 0.0);

    const std::vector<double> kVec1{PI / 3., PI / 3., 0.0};

    set_Hk0(kVec1, Hk0, lvec, UNIT_CELL);
    set_Hk0(kVec1, eVecs, lvec, UNIT_CELL);

    const int N = NATOM;
    const char JOBZ('V');
    const char UPLO('U');
    const int W = 2ul * N;
    int info = 0;

    std::vector<std::complex<double>> WORK(2 * N, std::complex<double>(0.0, 0.0));
    std::vector<double> RWORK(3 * N - 2, 0.0);

    zheev(&JOBZ, &UPLO, &N, &eVecs[0], &N, &evalsHk0[0], &WORK[0], &W, &RWORK[0], &info);

    assert(info == 0);

    std::vector<std::complex<double>> TEMP1(N * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP2(N * N, std::complex<double>(0.0, 0.0));

    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                N, N, N, &alpha,
                &Hk0[0], N,
                &eVecs[0], N,
                &beta, &TEMP1[0], N);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &alpha,
                &eVecs[0], N,
                &TEMP1[0], N,
                &beta, &TEMP2[0], N);

    //check that diagonal is correct ...
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(TEMP2[fq(i, i, NATOM)].real(), evalsHk0[i], 1.0);
        EXPECT_NEAR(TEMP2[fq(i, i, NATOM)].imag(), 0.0, 1e-12);
    }
    //... and that matrix is indeed diagonal
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                EXPECT_NEAR(TEMP2[fq(i, j, N)].real(), 0.0, 1e-12);
                EXPECT_NEAR(TEMP2[fq(i, j, N)].imag(), 0.0, 1e-12);
            }
        }
    }
}

TEST(LinATest, Hk0EqualsHkAExpCouplingForGEq0) {

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    std::vector<std::complex<double>> Hk0(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> HkA(NATOM * NATOM, std::complex<double>(0.0, 0.0));

    const std::vector<double> kVec1{PI / 17., PI / 7., 0.0};

    set_Hk0(kVec1, Hk0, lvec, UNIT_CELL);
    set_HkExpCoupling(HkA, kVec1, lvec, UNIT_CELL, 0.0, cavityConstants::eA);

    for (auto k = 0ul; k < Hk0.size(); ++k) {
        EXPECT_NEAR(Hk0[k].real(), HkA[k].real(), 1e-15);
        EXPECT_NEAR(Hk0[k].imag(), HkA[k].imag(), 1e-15);
    }
}


TEST(LinATest, HkAEquals0ForGEq0) {

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    std::vector<std::complex<double>> Hk0(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> HkA(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> HkAA(NATOM * NATOM, std::complex<double>(0.0, 0.0));

    const std::vector<double> kVec1{PI / 17., PI / 7., 0.0};

    set_Hk0(kVec1, Hk0, lvec, UNIT_CELL);
    set_HkA(HkA, kVec1, lvec, UNIT_CELL, 0.0, cavityConstants::eA);
    set_HkAA(HkAA, kVec1, lvec, UNIT_CELL, 0.0, cavityConstants::eA);

    for (auto k = 0ul; k < Hk0.size(); ++k) {
        EXPECT_NEAR(HkA[k].real(), 0.0, 1e-15);
        EXPECT_NEAR(HkA[k].imag(), 0.0, 1e-15);
        EXPECT_NEAR(HkAA[k].real(), 0.0, 1e-15);
        EXPECT_NEAR(HkAA[k].imag(), 0.0, 1e-15);
    }
}
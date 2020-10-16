#include <vector>
#include <complex>
#include <cmath>

#include "Constants.h"
#include "mkl.h"
#include "gtest/gtest.h"
#include "Diagonalization.h"
#include "MatrixMultiplication.h"
#include "InlineFunctions.h"

TEST(Diagonalization, diagonalizationGivesCorrectEvals) {

    const int N(2);

    std::vector<std::complex<double>> A{std::complex<double>(0.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(0.0, 0.0)};

    std::vector<double> evals(N, 2.0);

    const char JOBZ('V');
    const char UPLO('U');
    const int LDA = N;
    const int W = 2ul * N;
    int info = 0;

    std::vector<std::complex<double>> WORK(2 * N, std::complex<double>(0.0, 0.0));
    std::vector<double> RWORK(3 * N - 2, 0.0);

    zheev(&JOBZ, &UPLO, &N, &A[0], &LDA, &evals[0], &WORK[0], &W, &RWORK[0], &info);

    EXPECT_DOUBLE_EQ(evals[0], -1.);
    EXPECT_DOUBLE_EQ(evals[1], 1.);

}


TEST(Diagonalization, eigenvectorsAreCorrect) {

    const int N(2);

    std::vector<std::complex<double>> A{std::complex<double>(0.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(0.0, 0.0)};

    std::vector<double> evals(N, 2.0);

    const char JOBZ('V');
    const char UPLO('U');
    const int LDA = N;
    const int W = 2ul * N;
    int info = 0;

    std::vector<std::complex<double>> WORK(2 * N, std::complex<double>(0.0, 0.0));
    std::vector<double> RWORK(3 * N - 2, 0.0);

    zheev(&JOBZ, &UPLO, &N, &A[0], &LDA, &evals[0], &WORK[0], &W, &RWORK[0], &info);

    EXPECT_DOUBLE_EQ(A[0].real(), -1. / sqrt(2.));
    EXPECT_DOUBLE_EQ(A[1].real(), 1. / sqrt(2.));
    EXPECT_DOUBLE_EQ(A[2].real(), 1. / sqrt(2.));
    EXPECT_DOUBLE_EQ(A[3].real(), 1. / sqrt(2.));

}

TEST(Diagonalization, diagonalizationEqualsTransformation) {

    const int N(2);

    std::vector<std::complex<double>> A{std::complex<double>(0.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(1.0, 0.0),
                                        std::complex<double>(0.0, 0.0)};

    std::vector<std::complex<double>> AA{std::complex<double>(0.0, 0.0),
                                         std::complex<double>(1.0, 0.0),
                                         std::complex<double>(1.0, 0.0),
                                         std::complex<double>(0.0, 0.0)};


    std::vector<double> evals(N, 2.0);

    const char JOBZ('V');
    const char UPLO('U');
    const int LDA = N;
    const int W = 2ul * N;
    int info = 0;

    std::vector<std::complex<double>> WORK(2 * N, std::complex<double>(0.0, 0.0));
    std::vector<double> RWORK(3 * N - 2, 0.0);

    zheev(&JOBZ, &UPLO, &N, &A[0], &LDA, &evals[0], &WORK[0], &W, &RWORK[0], &info);

    std::vector<std::complex<double>> TEMP1(LDA * N, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> TEMP2(LDA * N, std::complex<double>(0.0, 0.0));

    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
                N, N, N, &alpha,
                &AA[0], LDA,
                &A[0], N, &beta,
                &TEMP1[0], N);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &alpha,
                &A[0], LDA,
                &TEMP1[0], N,
                &beta, &TEMP2[0], N);

    //check that matrix is now diagonal ...
    EXPECT_NEAR(TEMP2[1].real(), 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(TEMP2[1].imag(), 0.0);
    EXPECT_NEAR(TEMP2[2].real(), 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(TEMP2[2].imag(), 0.0);
    //... and that the eigenvalues are correct
    EXPECT_DOUBLE_EQ(TEMP2[0].real(), -1.0);
    EXPECT_DOUBLE_EQ(TEMP2[0].imag(), 0.0);
    EXPECT_DOUBLE_EQ(TEMP2[3].real(), 1.0);
    EXPECT_DOUBLE_EQ(TEMP2[3].imag(), 0.0);
}

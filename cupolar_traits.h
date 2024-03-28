/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

template <typename T>
struct cupolar_traits;

template <>
struct cupolar_traits<double> {
    typedef double S;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr double one = 1.0;
    static constexpr double half = 0.5;
    static constexpr double zero = 0.0;

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_R_64F;
    static constexpr cudaDataType computeType = CUDA_R_64F;
};

template <>
struct cupolar_traits<float> {
    typedef float S;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr float one = 1.0f;
    static constexpr float half = 0.5f;
    static constexpr float zero = 0.0f;

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
        return cublasSscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_R_32F;
    static constexpr cudaDataType computeType = CUDA_R_32F;
};

template <>
struct cupolar_traits<cuDoubleComplex> {
    typedef double S;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr cuDoubleComplex one = cuDoubleComplex{1.0, 0.0};
    static constexpr cuDoubleComplex half = cuDoubleComplex{0.5, 0.0};
    static constexpr cuDoubleComplex zero = cuDoubleComplex{0.0, 0.0};

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
        return cublasZdscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
        return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_C_64F;
    static constexpr cudaDataType computeType = CUDA_C_64F;
};

template <>
struct cupolar_traits<cuComplex> {
    typedef float S;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr cuComplex one = cuComplex{1.0f, 0.0f};
    static constexpr cuComplex half = cuComplex{0.5f, 0.0f};
    static constexpr cuComplex zero = cuComplex{0.0f, 0.0f};

    /*-----------------------------------------------------------------------------
     * scaling of a matrix, matrix addition and mulitplication using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasXdscal(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
        return cublasCsscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
        return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }

    inline static cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
        return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    /*-----------------------------------------------------------------------------
     * computeType and dataType for cuSolver
     *-----------------------------------------------------------------------------*/
    static constexpr cudaDataType dataType = CUDA_C_32F;
    static constexpr cudaDataType computeType = CUDA_C_32F;
};

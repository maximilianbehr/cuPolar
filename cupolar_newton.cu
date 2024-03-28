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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include "checkcuda.h"
#include "cupolar.h"
#include "cupolar_frobenius.h"
#include "cupolar_traits.h"

const static cusolverAlgMode_t CUSOLVER_ALG = CUSOLVER_ALG_0;

template <typename T>
static int cupolar_NewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    /*-----------------------------------------------------------------------------
     * initialize with zero
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = 0;
    *h_bufferSize = 0;

    /*-----------------------------------------------------------------------------
     * get device and host workspace size for LU factorization
     *-----------------------------------------------------------------------------*/
    // create cusolver handle
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // create cusolver params
    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    // compute workspace size
    CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cupolar_traits<T>::dataType, nullptr, n, cupolar_traits<T>::computeType, d_bufferSize, h_bufferSize));

    // free workspace
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));

    /*-----------------------------------------------------------------------------
     * compute final workspace size
     *-----------------------------------------------------------------------------*/
    *d_bufferSize += sizeof(T) * n * n * 3 + sizeof(int64_t) * n + sizeof(int);

    return 0;
}

int cupolar_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_NewtonBufferSize<float>(n, d_bufferSize, h_bufferSize);
}

int cupolar_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_NewtonBufferSize<double>(n, d_bufferSize, h_bufferSize);
}

int cupolar_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_NewtonBufferSize<cuComplex>(n, d_bufferSize, h_bufferSize);
}

int cupolar_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_NewtonBufferSize<cuDoubleComplex>(n, d_bufferSize, h_bufferSize);
}

template <typename T>
__global__ void identity(const int n, T *A, const int lda) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    for (int j = j0; j < n; j += gridDim.y * blockDim.y) {
        for (int i = i0; i < n; i += gridDim.x * blockDim.x) {
            A[i + j * lda] = (i == j) ? cupolar_traits<T>::one : cupolar_traits<T>::zero;
        }
    }
}

template <typename T>
static int cupolar_Newton(const int n, const T *A, void *d_buffer, void *h_buffer, T *Q, T *H) {
    /*-----------------------------------------------------------------------------
     * derived types
     *-----------------------------------------------------------------------------*/
    using S = typename cupolar_traits<T>::S;  // real type: double for cuDoubleComplex, float for cuComplex

    /*-----------------------------------------------------------------------------
     * constants and variables
     *-----------------------------------------------------------------------------*/
    int ret = 0, iter = 1;
    constexpr int maxiter = 100;
    const S tol = std::sqrt(std::numeric_limits<S>::epsilon());  // square root of machine epsilon - newton iteration converges quadratically
    S alpha, beta, zeta;

    /*-----------------------------------------------------------------------------
     * create cuBlas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * create cusolver handle and params structure
     *-----------------------------------------------------------------------------*/
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    cusolverDnParams_t params;
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    CHECK_CUSOLVER(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG));

    /*-----------------------------------------------------------------------------
     * split memory buffer
     * memory layout: |Qold, Qtmp, QoldInv, ipiv, info, d_work|
     *-----------------------------------------------------------------------------*/
    T *Qold = reinterpret_cast<T *>(d_buffer);
    T *Qtmp = reinterpret_cast<T *>(Qold + n * n);                   // put Qtmp after Qold
    T *QoldInv = reinterpret_cast<T *>(Qtmp + n * n);                // put QoldInv after Qold
    int64_t *d_ipiv = reinterpret_cast<int64_t *>(QoldInv + n * n);  // put d_ipiv after QoldInv
    int *d_info = reinterpret_cast<int *>(d_ipiv + n);               // put d_info after d_ipiv
    void *d_work = reinterpret_cast<int *>(d_info + 1);              // put d_work after d_info
    void *h_work = reinterpret_cast<void *>(h_buffer);
    std::swap(Q, Qold);

    /*-----------------------------------------------------------------------------
     * copy A to Qold
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaMemcpy(Qold, A, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));

    /*-----------------------------------------------------------------------------
     * newton iteration
     *-----------------------------------------------------------------------------*/
    iter = 1;
    static_assert(maxiter >= 1, "maxiter >= 1");
    while (true) {
        /*-----------------------------------------------------------------------------
         * copy Qold to Qtmp
         *-----------------------------------------------------------------------------*/
        CHECK_CUDA(cudaMemcpy(Qtmp, Qold, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));

        /*-----------------------------------------------------------------------------
         * compute inv(Q)^H
         *-----------------------------------------------------------------------------*/
        // workspace query for LU factorization
        size_t lworkdevice = 0, lworkhost = 0;
        CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cupolar_traits<T>::dataType, Qtmp, n, cupolar_traits<T>::computeType, &lworkdevice, &lworkhost));

        // compute LU factorization and set right side to identity on different streams
        cudaStream_t streamLU, streamIdentity;
        CHECK_CUDA(cudaStreamCreate(&streamLU));
        CHECK_CUDA(cudaStreamCreate(&streamIdentity));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, streamLU));
        CHECK_CUSOLVER(cusolverDnXgetrf(cusolverH, params, n, n, cupolar_traits<T>::dataType, Qtmp, n, d_ipiv, cupolar_traits<T>::computeType, d_work, lworkdevice, h_work, lworkhost, d_info));

        // set right-hand side to identity
        {
            dim3 grid((n + 15) / 16, (n + 15) / 16);
            dim3 block(16, 16);
            identity<<<grid, block, 0, streamIdentity>>>(n, QoldInv, n);
            CHECK_CUDA(cudaPeekAtLastError());
        }

        // synchronize and destroy streams
        CHECK_CUDA(cudaStreamSynchronize(streamLU));
        CHECK_CUDA(cudaStreamSynchronize(streamIdentity));
        CHECK_CUDA(cudaStreamDestroy(streamLU));
        CHECK_CUDA(cudaStreamDestroy(streamIdentity));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, 0));

        // solve the linear system to compute the hermitian/transposed inverse of Qold
        CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cupolar_traits<T>::dataType, Qtmp, n, d_ipiv, cupolar_traits<T>::computeType, QoldInv, n, d_info));

        /*-----------------------------------------------------------------------------
         * compute alpha and beta to compute zeta for the first iteration
         *-----------------------------------------------------------------------------*/
        if (iter == 1) {
            CHECK_CUPOLAR(cupolar_normFro(n, n, A, &alpha));
            CHECK_CUPOLAR(cupolar_normFro(n, n, QoldInv, &beta));
            beta = S{1.0} / beta;
            zeta = S{1.0} / std::sqrt(alpha * beta);
        }

        /*-----------------------------------------------------------------------------
         * update Q as 0.5 * (zeta * Q + 1/zeta * inv(Q)^H)
         *-----------------------------------------------------------------------------*/
        {
            T a, b;
            if constexpr (std::is_same<T, cuComplex>::value || std::is_same<T, cuDoubleComplex>::value) {
                a = T{S{0.5} * zeta, S{0.0}}, b = T{S{0.5} / zeta, S{0.0}};
            } else {
                a = S{0.5} * zeta, b = S{0.5} / zeta;
            }
            CHECK_CUBLAS(cupolar_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, n, n, &a, Qold, n, &b, QoldInv, n, Q, n));
        }

        /*-----------------------------------------------------------------------------
         * update zeta for the next iteration
         *-----------------------------------------------------------------------------*/
        if (iter == 1) {
            zeta = std::sqrt(S{2.0} * std::sqrt(alpha * beta) / (alpha + beta));
        } else {
            zeta = std::sqrt(S{2.0} / (zeta + S{1.0} / zeta));
        }

        /*-----------------------------------------------------------------------------
         *  compute relative change of Q and Qold
         *-----------------------------------------------------------------------------*/
        S diffQQold, nrmQ;
        CHECK_CUPOLAR(cupolar_diffnormFro(n, n, Q, Qold, &diffQQold));
        CHECK_CUPOLAR(cupolar_normFro(n, n, Q, &nrmQ));
        // printf("iter=%d, diffQQold=%e, nrmQ=%e, rel. change=%e\n", iter, diffQQold, nrmQ, diffQQold / nrmQ);

        /*-----------------------------------------------------------------------------
         * stopping criteria
         *-----------------------------------------------------------------------------*/
        // relative change of Q and Qold is smaller than tolerance
        if (diffQQold < nrmQ * tol) {
            break;
        }

        // maximum number of iterations reached
        if (iter == maxiter) {
            fprintf(stderr, "%s-%s:%d no convergence - maximum number of iterations reached\n", __func__, __FILE__, __LINE__);
            fflush(stderr);
            ret = -1;
            break;
        }

        /*-----------------------------------------------------------------------------
         * swap Q and Qold for the next iteration
         *-----------------------------------------------------------------------------*/
        std::swap(Q, Qold);
        iter++;
    }

    /*-----------------------------------------------------------------------------
     * copy Q and Qold if necessary
     *-----------------------------------------------------------------------------*/
    if (iter % 2 == 1) {
        CHECK_CUDA(cudaMemcpy(Qold, Q, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));
    }

    /*-----------------------------------------------------------------------------
     * compute H if wanted
     *-----------------------------------------------------------------------------*/
    if (H) {
        // compute H = Qold^H * A
        CHECK_CUBLAS(cupolar_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, &cupolar_traits<T>::one, Q, n, A, n, &cupolar_traits<T>::zero, Qtmp, n));
        // correct symmetry/hermitianess of H
        CHECK_CUBLAS(cupolar_traits<T>::cublasXgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, n, n, &cupolar_traits<T>::half, Qtmp, n, &cupolar_traits<T>::half, Qtmp, n, H, n));
    }

    /*-----------------------------------------------------------------------------
     * destroy cuBlas and cuSolver handle and params structure
     *-----------------------------------------------------------------------------*/
    CHECK_CUSOLVER(cusolverDnDestroyParams(params));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return ret;
}

int cupolar_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H) {
    return cupolar_Newton(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H) {
    return cupolar_Newton(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H) {
    return cupolar_Newton(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H) {
    return cupolar_Newton(n, A, d_buffer, h_buffer, Q, H);
}

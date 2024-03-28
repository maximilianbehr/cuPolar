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
static int cupolar_HayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
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

int cupolar_sHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_HayleyBufferSize<float>(n, d_bufferSize, h_bufferSize);
}

int cupolar_dHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_HayleyBufferSize<double>(n, d_bufferSize, h_bufferSize);
}

int cupolar_cHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_HayleyBufferSize<cuComplex>(n, d_bufferSize, h_bufferSize);
}

int cupolar_zHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    return cupolar_HayleyBufferSize<cuDoubleComplex>(n, d_bufferSize, h_bufferSize);
}

__device__ inline static cuComplex operator*(float a, const cuComplex &b) {
    return make_cuComplex(a * b.x, a * b.y);
}

__device__ inline static cuComplex operator+(float a, const cuComplex &b) {
    return make_cuComplex(a + b.x, b.y);
}

__device__ inline static cuDoubleComplex operator*(double a, const cuDoubleComplex &b) {
    return make_cuDoubleComplex(a * b.x, a * b.y);
}

__device__ inline static cuDoubleComplex operator+(double a, const cuDoubleComplex &b) {
    return make_cuDoubleComplex(a + b.x, b.y);
}

template <typename S, typename T>
__global__ void prepare_Hayley(const int n, const S a, const S b, const S c, T *QHQ, T *aIbQHQ) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    for (int j = j0; j < n; j += gridDim.y * blockDim.y) {
        for (int i = i0; i < n; i += gridDim.x * blockDim.x) {
            T tmp = QHQ[i + j * n];
            if (i == j) {
                QHQ[i + j * n] = S{1.0} + c * tmp;
                aIbQHQ[i + j * n] = a + b * tmp;
            } else {
                QHQ[i + j * n] = c * tmp;
                aIbQHQ[i + j * n] = b * tmp;
            }
        }
    }
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
static int cupolar_Hayley(const int n, const T *A, void *d_buffer, void *h_buffer, T *Q, T *H) {
    /*-----------------------------------------------------------------------------
     * derived types
     *-----------------------------------------------------------------------------*/
    using S = typename cupolar_traits<T>::S;  // real type: double for cuDoubleComplex, float for cuComplex

    /*-----------------------------------------------------------------------------
     * constants and variables
     *-----------------------------------------------------------------------------*/
    int ret = 0, iter = 1;
    constexpr int maxiter = 100;
    const S tol = std::cbrt(std::numeric_limits<S>::epsilon());  // square root of machine epsilon - newton iteration converges quadratically
    S alpha, beta, l, a, b, c;
    size_t lworkdevice = 0, lworkhost = 0;

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
     * memory layout: |Qold, Qtmp, Qtmp2, ipiv, info, d_work|
     *-----------------------------------------------------------------------------*/
    T *Qold = reinterpret_cast<T *>(d_buffer);
    T *Qtmp = reinterpret_cast<T *>(Qold + n * n);                 // put Qtmp after Qold
    T *Qtmp2 = reinterpret_cast<T *>(Qtmp + n * n);                // put Qtmp2 after Qold
    int64_t *d_ipiv = reinterpret_cast<int64_t *>(Qtmp2 + n * n);  // put d_ipiv after Qtmp2
    int *d_info = reinterpret_cast<int *>(d_ipiv + n);             // put d_info after d_ipiv
    void *d_work = reinterpret_cast<int *>(d_info + 1);            // put d_work after d_info
    void *h_work = reinterpret_cast<void *>(h_buffer);
    std::swap(Q, Qold);

    /*-----------------------------------------------------------------------------
     * compute alpha = || A ||_F
     *-----------------------------------------------------------------------------*/
    CHECK_CUPOLAR(cupolar_normFro(n, n, A, &alpha));

    /*-----------------------------------------------------------------------------
     * compute A^-1
     *-----------------------------------------------------------------------------*/
    // copy A to Qtmp
    CHECK_CUDA(cudaMemcpy(Qtmp, A, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));

    // workspace query for LU factorization
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
        identity<<<grid, block>>>(n, Qtmp2, n);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    // synchronize and destroy streams
    CHECK_CUDA(cudaStreamSynchronize(streamLU));
    CHECK_CUDA(cudaStreamSynchronize(streamIdentity));
    CHECK_CUDA(cudaStreamDestroy(streamLU));
    CHECK_CUDA(cudaStreamDestroy(streamIdentity));
    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, 0));

    // solve the linear system to compute the hermitian/transposed inverse of Qold
    CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cupolar_traits<T>::dataType, Qtmp, n, d_ipiv, cupolar_traits<T>::computeType, Qtmp2, n, d_info));

    /*-----------------------------------------------------------------------------
     * compute beta = 1 / || A^-1 ||_F
     *-----------------------------------------------------------------------------*/
    CHECK_CUPOLAR(cupolar_normFro(n, n, Qtmp2, &beta));
    beta = S{1.0} / beta;

    /*-----------------------------------------------------------------------------
     * compute l for the first iteration
     *-----------------------------------------------------------------------------*/
    l = beta / alpha;

    /*-----------------------------------------------------------------------------
     * copy A to Qold and scale A/alpha = Qold / alpha
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaMemcpy(Qold, A, sizeof(T) * n * n, cudaMemcpyDeviceToDevice));
    {
        S alphainv = S{1.0} / alpha;
        CHECK_CUBLAS(cupolar_traits<T>::cublasXdscal(cublasH, n * n, &alphainv, Qold, 1));
    }

    /*-----------------------------------------------------------------------------
     * hayley iteration
     *-----------------------------------------------------------------------------*/
    iter = 1;
    static_assert(maxiter >= 1, "maxiter >= 1");
    while (true) {
        /*-----------------------------------------------------------------------------
         * update scaling parameters
         *-----------------------------------------------------------------------------*/
        {
            S d = std::pow(S{4.0} * (S{1.0} - l * l) / (l * l * l * l), S{1.0} / S{3.0});
            a = std::sqrt(S{1.0} + d) + S{0.5} * std::sqrt(S{8.0} - S{4.0} * d + S{8.0} * (S{2.0} - l * l) / (l * l * std::sqrt(S{1.0} + d)));
            b = (a - S{1.0}) * (a - S{1.0}) / S{4.0};
            c = a + b - S{1.0};
            l = l * (a + b * l * l) / (S{1.0} + c * l * l);
        }

        /*-----------------------------------------------------------------------------
         * compute Qold^H Qold -> Qtmp
         *-----------------------------------------------------------------------------*/
        CHECK_CUBLAS(cupolar_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, &cupolar_traits<T>::one, Qold, n, Qold, n, &cupolar_traits<T>::zero, Qtmp, n));

        /*-----------------------------------------------------------------------------
         * compute I + c * Qtmp -> Qtmp and a * I + b * Qtmp -> Qtmp2
         *-----------------------------------------------------------------------------*/
        {
            dim3 grid((n + 15) / 16, (n + 15) / 16);
            dim3 block(16, 16);
            prepare_Hayley<<<grid, block>>>(n, a, b, c, Qtmp, Qtmp2);
            CHECK_CUDA(cudaPeekAtLastError());
        }

        /*-----------------------------------------------------------------------------
         * solve the system (I + c * Qold^H*Qold)\(a * I + b * Qold^H*Qold)
         *-----------------------------------------------------------------------------*/
        // workspace query for LU factorization
        CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, cupolar_traits<T>::dataType, Qtmp, n, cupolar_traits<T>::computeType, &lworkdevice, &lworkhost));
        // compute LU factorization
        CHECK_CUSOLVER(cusolverDnXgetrf(cusolverH, params, n, n, cupolar_traits<T>::dataType, Qtmp, n, d_ipiv, cupolar_traits<T>::computeType, d_work, lworkdevice, h_work, lworkhost, d_info));
        // solve the linear system
        CHECK_CUSOLVER(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, n, cupolar_traits<T>::dataType, Qtmp, n, d_ipiv, cupolar_traits<T>::computeType, Qtmp2, n, d_info));

        /*-----------------------------------------------------------------------------
         * update Q as Q <- Qold*(I + c * Qold^H*Qold)\(a * I + b * Qold^H*Qold)
         *-----------------------------------------------------------------------------*/
        CHECK_CUBLAS(cupolar_traits<T>::cublasXgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &cupolar_traits<T>::one, Qold, n, Qtmp2, n, &cupolar_traits<T>::zero, Q, n));

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

int cupolar_sHayley(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H) {
    return cupolar_Hayley(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_dHayley(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H) {
    return cupolar_Hayley(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_cHayley(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H) {
    return cupolar_Hayley(n, A, d_buffer, h_buffer, Q, H);
}

int cupolar_zHayley(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H) {
    return cupolar_Hayley(n, A, d_buffer, h_buffer, Q, H);
}

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

#include <cublas_v2.h>
#include <cusolverDn.h>

#define CHECK_CUPOLAR(err)                                                       \
    do {                                                                         \
        int error_code = (err);                                                  \
        if (error_code) {                                                        \
            fprintf(stderr, "cupolar Error %d in %s:%d\n", error_code, __FILE__, \
                    __LINE__);                                                   \
            fflush(stderr);                                                      \
            return -1;                                                           \
        }                                                                        \
    } while (false)

#define CHECK_CUDA(err)                                                  \
    do {                                                                 \
        cudaError_t error_code = (err);                                  \
        if (error_code != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA Error %d-%s in %s:%d\n", error_code,   \
                    cudaGetErrorString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                              \
            return -2;                                                   \
        }                                                                \
    } while (false)

#define CHECK_CUBLAS(err)                                                   \
    do {                                                                    \
        cublasStatus_t error_code = (err);                                  \
        if (error_code != CUBLAS_STATUS_SUCCESS) {                          \
            fprintf(stderr, "CUBLAS Error %d-%s in %s:%d\n", error_code,    \
                    cublasGetStatusString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                 \
            return -3;                                                      \
        }                                                                   \
    } while (false)

static inline const char *cupolar_cusolverGetErrorEnum(cusolverStatus_t error) {
    switch (error) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "unknown";
    }
}

#define CHECK_CUSOLVER(err)                                                        \
    do {                                                                           \
        cusolverStatus_t error_code = (err);                                       \
        if (error_code != CUSOLVER_STATUS_SUCCESS) {                               \
            fprintf(stderr, "CUSOLVER Error %d-%s in %s:%d\n", error_code,         \
                    cupolar_cusolverGetErrorEnum(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                        \
            return -4;                                                             \
        }                                                                          \
    } while (false)

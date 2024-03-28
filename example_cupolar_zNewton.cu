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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>

#include "cupolar.h"

int main(void) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;                 // return value
    const int n = 100;           // size of the input matrix A n-by-n
    cuDoubleComplex *A, *Q, *H;  // input matrix and factor matrices
    void *d_buffer = NULL;       // device buffer
    void *h_buffer = NULL;       // host buffer

    /*-----------------------------------------------------------------------------
     * allocate A, Q and H
     *-----------------------------------------------------------------------------*/
    cudaMallocManaged((void **)&A, sizeof(*A) * n * n);
    cudaMallocManaged((void **)&Q, sizeof(*Q) * n * n);
    cudaMallocManaged((void **)&H, sizeof(*H) * n * n);

    /*-----------------------------------------------------------------------------
     * create a random matrix A
     *-----------------------------------------------------------------------------*/
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + j * n] = cuDoubleComplex{(double)rand() / RAND_MAX, (double)rand() / RAND_MAX};
        }
    }

    /*-----------------------------------------------------------------------------
     * perform a workspace query and allocate memory buffer on the host and device
     *-----------------------------------------------------------------------------*/
    size_t d_bufferSize = 0, h_bufferSize = 0;
    cupolar_zNewtonBufferSize(n, &d_bufferSize, &h_bufferSize);

    if (d_bufferSize > 0) {
        cudaMalloc((void **)&d_buffer, d_bufferSize);
    }

    if (h_bufferSize > 0) {
        h_buffer = malloc(h_bufferSize);
    }

    /*-----------------------------------------------------------------------------
     * call cupolar_zNewton to compute Q and H
     *-----------------------------------------------------------------------------*/
    cupolar_zNewton(n, A, d_buffer, h_buffer, Q, H);

    /*-----------------------------------------------------------------------------
     * check polar decomposition A = Q*H
     *-----------------------------------------------------------------------------*/
    double fronrmA = 0.0, fronrmdiff = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cuDoubleComplex sum = cuDoubleComplex{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                sum = cuCadd(sum, cuCmul(Q[i + k * n], H[k + j * n]));
            }
            double diff = cuCabs((cuCsub(A[i + j * n], sum)));
            fronrmdiff += diff * diff;
            fronrmA += cuCreal(cuCmul(cuConj(A[i + j * n]), A[i + j * n]));
        }
    }
    double error = sqrt(fronrmdiff / fronrmA);
    printf("rel. error ||A-Q*H||_F / ||A||_F = %e\n", error);
    if (error < 1e-8) {
        printf("Polar Decomposition successful\n");
    } else {
        printf("Polar Decomposition failed\n");
        ret = 1;
    }

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cudaFree(A);
    cudaFree(Q);
    cudaFree(H);
    cudaFree(d_buffer);
    free(h_buffer);
    return ret;
}

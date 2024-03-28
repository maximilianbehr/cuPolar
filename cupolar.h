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

#ifdef __cplusplus
extern "C" {
#endif

int cupolar_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);

int cupolar_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H);
int cupolar_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H);
int cupolar_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H);
int cupolar_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H);

int cupolar_sHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_dHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_cHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_zHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);

int cupolar_sHayley(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H);
int cupolar_dHayley(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H);
int cupolar_cHayley(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H);
int cupolar_zHayley(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H);

#ifdef __cplusplus
}
#endif

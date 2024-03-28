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
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

template <typename S, typename T>
struct square : public thrust::unary_function<S, T> {
    __host__ __device__ S operator()(const T &x) const {
        if constexpr (std::is_same<T, cuComplex>::value) {
            return cuCrealf(cuCmulf(cuConjf(x), x));
        } else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
            return cuCreal(cuCmul(cuConj(x), x));
        } else {
            return x * x;
        }
        return S{0.0};  // unreachable - fixed compiler warning
    }
};

template <typename S, typename T>
static int cupolar_normFro(int m, int n, const T *A, S *nrmA) {
    /*-----------------------------------------------------------------------------
     * compute || A ||_F = sqrt( sum( A(:).^2 ) )
     *-----------------------------------------------------------------------------*/
    S zero{0.0};
    *nrmA = thrust::transform_reduce(thrust::device_pointer_cast(A), thrust::device_pointer_cast(A + m * n), square<S, T>(), zero, thrust::plus<S>());
    *nrmA = std::sqrt(*nrmA);
    return 0;
}

template <typename S, typename T>
struct diffsquare : public thrust::binary_function<S, T, T> {
    __host__ __device__ S operator()(const T &x, const T &y) const {
        if constexpr (std::is_same<T, cuComplex>::value) {
            T diff = cuCsubf(x, y);
            return cuCrealf(cuCmulf(cuConjf(diff), diff));
        } else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
            T diff = cuCsub(x, y);
            return cuCreal(cuCmul(cuConj(diff), diff));
        } else {
            T diff = x - y;
            return diff * diff;
        }
        return S{0.0};  // unreachable - fixed compiler warning
    }
};

template <typename S, typename T>
static int cupolar_diffnormFro(int m, int n, const T *A, const T *B, S *diff) {
    /*-----------------------------------------------------------------------------
     * compute || A - B ||_F = sqrt( sum( (A(:) - B(:)).^2 ) )
     *-----------------------------------------------------------------------------*/
    S zero{0.0};
    *diff = thrust::inner_product(thrust::device_pointer_cast(A), thrust::device_pointer_cast(A + m * n), thrust::device_pointer_cast(B), zero, thrust::plus<S>(), diffsquare<S, T>());
    *diff = std::sqrt(*diff);
    return 0;
}

# cuPolar - Matrix Polar Decomposition using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuPolar)
 [![DOI](https://zenodo.org/badge/777737236.svg)](https://zenodo.org/doi/10.5281/zenodo.10892350)

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuPolar` is a `CUDA` library implementing Newton and Hayley's method for the Polar Decomposition of a nonsingular square matrix $A=QH$, where $Q$ is unitary and $H$ is hermitian positive semidefinite.

`cuNMF` supports real and complex, single and double precision matrices.

## Available Functions


### Single Precision Functions
```C
int cupolar_sNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_sNewton(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H);

int cupolar_sHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_sHayley(const int n, const float *A, void *d_buffer, void *h_buffer, float *Q, float *H);
```

### Double Precision Functions
```C
int cupolar_dNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_dNewton(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H);

int cupolar_dHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_dHayley(const int n, const double *A, void *d_buffer, void *h_buffer, double *Q, double *H);
```

### Complex Single Precision Functions
```C
int cupolar_cNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_cNewton(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H);

int cupolar_cHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_cHayley(const int n, const cuComplex *A, void *d_buffer, void *h_buffer, cuComplex *Q, cuComplex *H);
```

### Complex Double Precision Functions
```C
int cupolar_zNewtonBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_zNewton(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H);

int cupolar_zHayleyBufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cupolar_zHayley(const int n, const cuDoubleComplex *A, void *d_buffer, void *h_buffer, cuDoubleComplex *Q, cuDoubleComplex *H);
```


## Algorithm

`cuPolar` implements the Newton's method with scaling as well as Hayley's method with scaling for the approximation of the polar decomposition.
The matrix $A$ must be square and nonsingular.

See also Algorithm 3.1

> Byers, R., & Xu, H. (2008). A new scaling for Newton's iteration for the polar decomposition and its backward stability. _SIAM Journal on Matrix Analysis and Applications_, 30(2), 822-843.

and Equations 3.14-3.16

> Nakatsukasa, Y., Bai, Z., & Gygi, F. (2010). Optimizing Halley's iteration for computing the matrix polar decomposition. _SIAM Journal on Matrix Analysis and Applications_, 31(5), 2700-2720.


## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `CUDA >= 11.4.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

We provide examples for all supported matrix formats:
  
| File                                                       | Data                                |
| -----------------------------------------------------------|-------------------------------------|
| [`example_cupolar_sNewton.cu`](example_cupolar_sNewton.cu) | real, single precision matrix       |
| [`example_cupolar_dNewton.cu`](example_cupolar_dNewton.cu) | real, double precision matrix       |
| [`example_cupolar_cNewton.cu`](example_cupolar_cNewton.cu) | complex, single precision matrix    |
| [`example_cupolar_zNewton.cu`](example_cupolar_zNewton.cu) | complex, double precision matrix    |
| [`example_cupolar_sHayley.cu`](example_cupolar_sHayley.cu) | real, single precision matrix       |
| [`example_cupolar_dHayley.cu`](example_cupolar_dHayley.cu) | real, double precision matrix       |
| [`example_cupolar_cHayley.cu`](example_cupolar_cHayley.cu) | complex, single precision matrix    |
| [`example_cupolar_zHayley.cu`](example_cupolar_zHayley.cu) | complex, double precision matrix    |

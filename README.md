# cuPolar - Matrix Polar Decomposition using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuPolar)

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuPolar` is a `CUDA` library implementing Newton and Hayley's method for the Polar Decomposition of a nonsingular square matrix $A=QH$, where $Q$ is unitary and $H$ is hermitian positive semidefinite.

`cuNMF` supports real and complex, single and double precision matrices.

## Available Functions

### General Functions
```C
```

### Single Precision Functions
```C
```

### Double Precision Functions
```C
```

### Complex Single Precision Functions
```C
```

### Complex Double Precision Functions
```C
```


## Algorithm

`cuPolar` implements the Newton's method with scaling as well as Hayley's method with scaling for the approximation of the polar decomposition.
The matrix $A$ must be square and nonsingular.

See also

> Byers, R., & Xu, H. (2008). A new scaling for Newton's iteration for the polar decomposition and its backward stability. _SIAM Journal on Matrix Analysis and Applications_, 30(2), 822-843.

and

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

<!--## Usage and Examples-->

<!--See [`example_cunmf_MUbeta.cu`](example_cunmf_MUbeta.cu) for an example using double precision data.-->



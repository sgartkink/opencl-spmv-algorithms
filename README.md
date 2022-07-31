# opencl-spmv-algorithms

## Implemented algorithms

- Coordinate Format (COO)

- Compressed Sparse Row Format (CSR)

- ELL Format (ELL)

- SELL-C-sigma

- Compressed Multi-Row Sparse Format (CMRS)

## Build

Run `make coo`, `make csr`, `make ell`, `make sigma_c`, `make cmrs` or `make` (all) in the root directory to build a specific algorithm.

### Debug

To build with the debug option, add `DEBUG=yes` to the build command, e.g. `make DEBUG=yes coo`.

## Run

- `./bin/coo`

- `./bin/csr`

- `./bin/ell`

- `./bin/sigma_c`

- `./bin/cmrs`

## Requirements

OpenCL >= 2.2

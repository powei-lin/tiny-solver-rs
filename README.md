# tiny-solver-rs
[![PyPI - Version](https://img.shields.io/pypi/v/tiny-solver.svg)](https://pypi.org/project/tiny-solver)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tiny-solver.svg)](https://pypi.org/project/tiny-solver)

Inspired by [ceres-solver](https://github.com/ceres-solver/ceres-solver), [tiny-solver](https://github.com/keir/tinysolver), and [minisam](https://github.com/dongjing3309/minisam).

This is a general optimizer written in Rust, including bindings for Python. If you're familiar with ceres-solver or factor-graph optimizers, you'll find it very easy to use.

## Installation
### python
The python package can be installed directly from PyPI:
```sh
pip install tiny-solver
```
### rust
```sh
cargo add tiny-solver
```

## Current Features

- [x] Automatic Derivatives using [num-dual](https://github.com/itt-ustutt/num-dual)
- [x] Sparse QR, Sparse Cholesky using [faer](https://github.com/sarah-ek/faer-rs)
- [x] GaussNewtonOptimizer
- [ ] Multithreading jacobian

#### TODO
- [ ] levenberg_marquardt_optimizer
- [ ] information matrix
- [ ] loss function
- [ ] factor in python

## Usage
Under development.

## Example

### python
```sh
git clone https://github.com/powei-lin/tiny-solver-rs.git
cd tiny-solver-rs
pip install tiny-solver matplotlib
python3 examples/python/m3500.py
```
### rust
```sh
git clone https://github.com/powei-lin/tiny-solver-rs.git
cd tiny-solver-rs
cargo run -r --example m3500_benchmar
```
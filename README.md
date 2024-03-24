# tiny-solver-rs
> Warning! This project is still under development.

[![crate](https://img.shields.io/crates/v/tiny-solver.svg)](https://crates.io/crates/tiny-solver)
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
- [x] ~~Sparse QR~~, Sparse Cholesky using [faer](https://github.com/sarah-ek/faer-rs)
- [x] GaussNewtonOptimizer
- [x] Multithreading jacobian
- [x] loss function (Huber)
- [x] Define factor in python

#### TODO
- [ ] LevenbergMarquardtOptimizer
- [ ] information matrix

## Benchmark
| dataset | tiny-solver | gtsam   | minisam  |
|---------|-------------|---------|----------|
| m3500   | 161.1ms     | 130.7ms | 123.6 ms |
|         |             |         |          |

## Usage
```rust
// define your own Cost/Factor struct
// impl residual function
// and the jacobian will be auto generated
struct CustomFactor {}
impl tiny_solver::factors::Factor for CustomFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let x = &params[0][0];
        let y = &params[1][0];
        let z = &params[1][1];

        na::dvector![x + y.clone().mul(2.0) + z.clone().mul(4.0), y.mul(z)]
    }
}

fn main() {
    // init logger, `export RUST_LOG=trace` to see more log
    env_logger::init();

    // init problem (factor graph)
    let mut problem = tiny_solver::Problem::new();

    // add residual blocks (factors)
    // add residual x needs to be close to 3.0
    problem.add_residual_block(
        1,
        vec![("x".to_string(), 1)],
        Box::new(tiny_solver::factors::PriorFactor {
            v: na::dvector![3.0],
        }),
        None,
    );
    // add custom residual for x and yz
    problem.add_residual_block(
        2,
        vec![("x".to_string(), 1), ("yz".to_string(), 2)],
        Box::new(CustomFactor {}),
        None,
    );

    // the initial values for x is 0.7 and yz is [-30.2, 123.4]
    let initial_values = HashMap::<String, na::DVector<f64>>::from([
        ("x".to_string(), na::dvector![0.7]),
        ("yz".to_string(), na::dvector![-30.2, 123.4]),
    ]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);

    // result
    for (k, v) in result {
        println!("{}: {}", k, v);
    }
}
```

## Example
### Basic example
```sh
cargo run -r --example small_problem
```
### M3500 dataset
<img src="docs/m3500_rs.png" width="300" alt="m3500 dataset rust result.">

```sh
git clone https://github.com/powei-lin/tiny-solver-rs.git
cd tiny-solver-rs

# run rust version
cargo run -r --example m3500_benchmar

# run python version
pip install tiny-solver matplotlib
python3 examples/python/m3500.py
```
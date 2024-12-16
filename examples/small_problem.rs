use std::collections::HashMap;
use std::ops::Mul;

use nalgebra as na;
use tiny_solver::{self, Optimizer};

struct CustomFactor {}
// define your own residual function and the jacobian will be auto generated
impl<T: na::RealField> tiny_solver::factors::Factor<T> for CustomFactor {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        let x = &params[0][0];
        let y = &params[1][0];
        let z = &params[1][1];

        na::dvector![
            x.clone()
                + y.clone() * T::from_f64(2.0).unwrap()
                + z.clone() * T::from_f64(4.0).unwrap(),
            y.clone() * z.clone()
        ]
    }
}
// impl tiny_solver::factors::FactorImpl for CustomFactor {}

fn main() {
    // init logger, `export RUST_LOG=trace` to see more log
    env_logger::init();

    // init problem (factor graph)
    let mut problem = tiny_solver::Problem::new();

    // add residual blocks (factors)
    // add residual x needs to be close to 3.0
    problem.add_residual_block(
        1,
        &[("x", 1)],
        Box::new(tiny_solver::factors::PriorFactor {
            v: na::dvector![3.0],
        }),
        None,
    );
    // add custom residual for x and yz
    problem.add_residual_block(2, &[("x", 1), ("yz", 2)], Box::new(CustomFactor {}), None);

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
    for (k, v) in result.unwrap() {
        println!("{}: {}", k, v);
    }
}

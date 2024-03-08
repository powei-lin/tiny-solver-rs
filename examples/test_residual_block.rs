use std::ops::Mul;
use std::vec;

extern crate nalgebra as na;
use na::{Const, Dyn};
use tiny_solver::problem;

fn cost_function_dyn(
    params: &Vec<na::DVector<num_dual::DualDVec64>>,
) -> na::DVector<num_dual::DualDVec64> {
    let x = &params[0][0];
    let y = &params[0][1];
    let z = &params[0][2];
    return na::dvector![x + y.clone().mul(2.0) + z.clone().mul(4.0), y * z];
}

fn main() {
    println!("rs block");
    let rsb = problem::ResidualBlock {
        dim_residual: 2,
        residual_row_start_idx: 0,
        variable_key_list: vec!["aa".to_string()],
        residual_func: Box::new(cost_function_dyn),
    };
    let x0 = num_dual::DualDVec64::new(
        1.2,
        num_dual::Derivative::some(na::dvector![1.0, 0.0, 0.0, 1.0]),
    );
    let y0 = num_dual::DualDVec64::new(
        1.2,
        num_dual::Derivative::some(na::dvector![0.0, 1.0, 0.0, 0.0]),
    );
    let z0 = num_dual::DualDVec64::new(
        1.2,
        num_dual::Derivative::some(na::dvector![0.0, 0.0, 1.0, 0.0]),
    );
    let param = vec![na::dvector![x0, y0, z0]];
    let a = (rsb.residual_func)(&param);
    println!("{}", a[1].eps.clone().unwrap_generic(Dyn(3), Const::<1>));
}

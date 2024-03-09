use std::ops::Mul;
use std::vec;

extern crate nalgebra as na;
use na::{Const, Dyn};
use num_dual::{DualDVec64, DualVec};
use tiny_solver::{problem, residual_block};

fn cost_function_dyn(
    params: &Vec<na::DVector<num_dual::DualDVec64>>,
) -> na::DVector<num_dual::DualDVec64> {
    let x = &params[0][0];
    let y = &params[1][0];
    let z = &params[1][1];
    return na::dvector![x + y.clone().mul(2.0) + z.clone().mul(4.0), y * z];
}

fn rows(aa: &Vec<i32>) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    let mut current = 0;
    for &num in aa {
        let next = current + num;
        let range = (current..next).collect::<Vec<_>>();
        result.push(range);
        current = next;
    }
    result
}

fn main() {
    println!("rs block");
    let rsb = residual_block::ResidualBlock {
        dim_residual: 2,
        residual_row_start_idx: 0,
        variable_key_list: vec!["aa".to_string()],
        residual_func: Box::new(cost_function_dyn),
    };
    // let (r, j) = rsb.jacobian(&vec![na::dvector![1.0], na::dvector![-2.0, 3.0]]);
    // println!("{},{}", r, j);
    let mut problem = problem::Problem::new();
    problem.add_residual_block(2, vec![("abc".to_string(), 1)], Box::new(cost_function_dyn));
    println!("{}", problem.total_residual_dimension);
}

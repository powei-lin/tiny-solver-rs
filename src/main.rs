mod example_cost;
// use std::mem::MaybeUninit;

use example_cost::cost_trait::CostFunc;
use example_cost::ExampleStatic;
use faer_core::Mat;
// use faer_lu::partial_pivoting;
// use std::ops;

// pub struct SliceVec<T, const N: usize> {
//     // pub v: [T, N],
//     pub buff: [MaybeUninit<T>; N],
// }

// pub struct TinySolver<T> {
//     error_: Vec<T>,
//     f_x_new_: Vec<T>,
//     jacobian_: Vec<T>,
//     jtj_: Vec<T>,
//     jtj_augmented_: Vec<T>,
// }

fn main() {
    // let v = vec![1.0, 2.0, 3.0];
    let x0 = [0.76026643, -30.01799744, 0.55192142];
    let mut residual = Mat::new();
    let mut jac: Mat<f64> = Mat::new();
    ExampleStatic::apply(&x0, &mut residual, Some(&mut jac));
    let H = jac.transpose() * jac.clone();
    let g_ = jac.transpose() * residual.clone();

    println!("residual \n{:?}\n", residual);
    println!("jac \n{:?}\n", &jac);
    println!("H:\n{:?}", H);
    println!("gradient:\n{:?}", g_);


    // const NUM_PARAMETERS: u32 = 3;
    // const NUM_RESIDUALS: u32 = 2;
}

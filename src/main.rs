mod cost_function;
mod example_cost;
extern crate nalgebra as na;

use cost_function::CostFunc;
use example_cost::{ExampleStatic, Matrix2x1d, Matrix2x3d, Matrix3x1d};

fn main() {
    // let v = vec![1.0, 2.0, 3.0];

    let mut x0 = Matrix3x1d::new(0.76026643, -30.01799744, 0.55192142);
    let gradient_threshold = 1e-16;
    let mut relative_step_threshold = 1e-16;
    let mut error_threshold = 1e-16;
    let mut initial_scale_factor = 1e-3;
    let max_iterations = 100;

    let mut u: f64;
    let mut v = 2;

    for i in 0..max_iterations {
        let mut residual = Matrix2x1d::default();
        let mut jac = Matrix2x3d::default();
        ExampleStatic::apply(&x0, &mut residual, Some(&mut jac));
        let g_ = jac.transpose() * -residual.clone();
        let H = jac.transpose() * jac.clone();
        println!("residual \n{}\n", residual);
        println!("jac \n{}\n", &jac);
        println!("H:\n{}", H);
        println!("gradient:\n{}", g_);

        let max_gradient = g_.abs().max();
        println!("mg{}", max_gradient);
        if max_gradient < gradient_threshold {
            println!("gradient too small. {}", max_gradient);
            break;
        }

        u = initial_scale_factor * H.diagonal().max();
        v = 2;
        println!("u: {}", u);
        let mut jtj_augmented_ = H.clone();
        let mut mm = jtj_augmented_.diagonal();
        mm.add_scalar_mut(u);
        jtj_augmented_.set_diagonal(&mm);
        println!("jtj {}", jtj_augmented_);
        println!("mm {}", mm);

        let dx = jtj_augmented_.lu().solve(&g_).unwrap();
        x0 += dx;
        // u = H.diagonal().max();
    }
    println!("x0 {}", x0);

    // const NUM_PARAMETERS: u32 = 3;
    // const NUM_RESIDUALS: u32 = 2;
}

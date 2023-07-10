use std::cell::Cell;

use crate::cost_function;
use num_traits;
extern crate nalgebra as na;

use na::{SMatrix, Scalar};

struct TinySolver<
    T: Scalar + num_traits::identities::Zero,
    const NUM_PARAMETERS: usize,
    const NUM_RESIDUALS: usize,
    // F: cost_function::CostFunc<T, NUM_PARAMETERS, NUM_RESIDUALS>,
> {
    // cost_function: F,
    // jtj: SMatrix<Scalar, NUM_RESIDUALS, >,
    _params: SMatrix<T, NUM_PARAMETERS, 1>,
    _residual: SMatrix<T, NUM_RESIDUALS, 1>,
    _jacobian: SMatrix<T, NUM_RESIDUALS, NUM_PARAMETERS>,
    gradient_threshold: f64,
    relative_step_threshold: f64,
    error_threshold: f64,
    initial_scale_factor: f64,
    max_iterations: usize,
}

impl<
        T: Scalar + num_traits::identities::Zero,
        const NUM_PARAMETERS: usize,
        const NUM_RESIDUALS: usize,
    > TinySolver<T, NUM_PARAMETERS, NUM_RESIDUALS>
{
    pub fn init() -> Self {
        Self {
            _params: SMatrix::zeros(),
            _residual: SMatrix::zeros(),
            _jacobian: SMatrix::zeros(),
            gradient_threshold: 1e-16,
            relative_step_threshold: 1e-16,
            error_threshold: 1e-16,
            initial_scale_factor: 1e-3,
            max_iterations: 100,
        }
    }

    pub fn solve(self) -> bool {
        true
    }
}

trait TS<Scalar, const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    type F: cost_function::CostFunc<NUM_PARAMETERS, NUM_RESIDUALS>;
}

// pub trait TinySolverT<
//     Scalar,
//     const NUM_PARAMETERS: usize,
//     const NUM_RESIDUALS: usize,
//     F: cost_function::CostFunc<Scalar, NUM_PARAMETERS, NUM_RESIDUALS>,
// >
// {
//     fn solve(&self, x: &mut SMatrix<Scalar, NUM_PARAMETERS, 1>) {
//         // let mut x0 = Matrix3x1d::new(0.76026643, -30.01799744, 0.55192142);

//         let mut u: f64;
//         let mut v = 2;
//         for i in 0..max_iterations {
//             // _residual: &mut ,
//             // _jacobian: Option<&mut SMatrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS>>,
//             let mut residual: SMatrix<f64, 10000, 1> = SMatrix::zeros();
//             let mut jac = Matrix2x3d::default();
//             F::apply(x, &mut residual, Some(&mut jac));
//             let g_ = jac.transpose() * -residual.clone();
//             let H = jac.transpose() * jac.clone();
//             println!("residual \n{}\n", residual);
//             println!("jac \n{}\n", &jac);
//             println!("H:\n{}", H);
//             println!("gradient:\n{}", g_);

//             let max_gradient = g_.abs().max();
//             println!("mg{}", max_gradient);
//             if max_gradient < gradient_threshold {
//                 println!("gradient too small. {}", max_gradient);
//                 break;
//             }

//             u = initial_scale_factor * H.diagonal().max();
//             v = 2;
//             println!("u: {}", u);
//             let mut jtj_augmented_ = H.clone();
//             let mut mm = jtj_augmented_.diagonal();
//             mm.add_scalar_mut(u);
//             jtj_augmented_.set_diagonal(&mm);
//             println!("jtj {}", jtj_augmented_);
//             println!("mm {}", mm);

//             let dx = jtj_augmented_.lu().solve(&g_).unwrap();
//             x += dx;
//             u = H.diagonal().max();
//         }
//         // println!("x0 {}", x);
//     }
// }

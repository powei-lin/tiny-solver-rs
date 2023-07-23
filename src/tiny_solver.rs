extern crate nalgebra as na;
use std::ops::Mul;

use crate::status::Status;
use na::Matrix;
use num_dual;

pub trait TinySolver<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    fn cost_function(
        _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
    ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>;

    fn solve(params: &mut na::SVector<f64, NUM_PARAMETERS>) -> bool {
        let mut status = Status::defualt();
        let mut u: f64;
        let mut v = 2;

        for i in 0..status.max_iterations {
            let (residual, jac) = num_dual::jacobian(Self::cost_function, params.clone());
            let gradient = jac.transpose().mul(-residual);
            let jtj = jac.transpose().mul(jac);
            // println!("residual \n{}\n", residual);
            // println!("jac \n{}\n", &jac);
            // println!("H:\n{}", jtj);
            // println!("gradient:\n{}", step);

            let max_gradient = gradient.abs().max();
            if max_gradient < status.gradient_threshold {
                println!("gradient too small. {}", max_gradient);
                break;
            }

            u = status.initial_scale_factor * jtj.diagonal().max();
            v = 2;
            println!("u: {}", u);
            let mut jtj_augmented  = na::DMatrix::<f64>::zeros(NUM_PARAMETERS, NUM_PARAMETERS);
            jtj_augmented.copy_from(&jtj);
            jtj_augmented.set_diagonal(&jtj_augmented.diagonal().add_scalar(u));

            println!("jtj {}", jtj_augmented);
            let dx = na::linalg::LU::new(jtj_augmented.clone()).solve(&gradient).unwrap();
            let s0: na::SMatrix<f64, NUM_PARAMETERS, 1> = jtj_augmented.fixed_view(0, 0) * dx;
            let succ = (s0 - gradient).abs().min() < status.error_threshold;
            if succ{
                println!("success!");
                *params += dx;
            }
            else{
                println!("fail {}", s0 - gradient);
            }
            
        }
        println!("x0 {}", params);

        true
    }
}

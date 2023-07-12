extern crate nalgebra as na;
use crate::status::Status;

pub trait TinySolverF64<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize>{
    const NUM_PARAMETERS: usize = NUM_PARAMETERS;
    const NUM_RESIDUALS: usize = NUM_RESIDUALS;
    fn cost_function(
        _params: &mut na::SMatrix<f64, NUM_PARAMETERS, 1>,
        _residual: &mut na::SMatrix<f64, NUM_RESIDUALS, 1>,
        _jacobian: Option<&mut na::SMatrix<f64, NUM_RESIDUALS, NUM_PARAMETERS>>,
    ) {
        unimplemented!()
    }
    fn solve(params: &mut na::SMatrix<f64, NUM_PARAMETERS, 1>) -> bool {
        let mut status = Status::init();
        let mut u: f64;
        let mut v = 2;
        let mut residual = na::SMatrix::<f64, NUM_RESIDUALS, 1>::zeros();
        let mut jac = na::SMatrix::<f64, NUM_RESIDUALS, NUM_PARAMETERS>::zeros();
        for i in 0..status.max_iterations {
            Self::cost_function(params, &mut residual, Some(&mut jac));
            let g_ = jac.transpose() * -residual.clone();
            let H = jac.transpose() * jac.clone();
            println!("residual \n{}\n", residual);
            println!("jac \n{}\n", &jac);
            println!("H:\n{}", H);
            println!("gradient:\n{}", g_);

            let max_gradient = g_.abs().max();
            println!("mg{}", max_gradient);
            if max_gradient < status.gradient_threshold {
                println!("gradient too small. {}", max_gradient);
                break;
            }

            u = status.initial_scale_factor * H.diagonal().max();
            v = 2;
            println!("u: {}", u);
            let mut jtj = na::DMatrix::<f64>::zeros(NUM_PARAMETERS, NUM_PARAMETERS);
            for i in 0..NUM_PARAMETERS {
                for j in 0..NUM_PARAMETERS {
                    if i == j{
                        jtj[(i, j)] = H[(i, j)]+u;
                    }
                    else{
                        jtj[(i, j)] = H[(i, j)];
                    }
                }
            }

            println!("jtj {}", jtj);
            let dx = na::linalg::LU::new(jtj).solve(&g_).unwrap();
            *params += dx;
        }
        println!("x0 {}", params);

        true
    }
    
}
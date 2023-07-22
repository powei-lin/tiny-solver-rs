extern crate nalgebra as na;
use crate::status::Status;
use num_dual;

pub trait TinySolverF64<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
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
        let mut status = Status::defualt();
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
                    if i == j {
                        jtj[(i, j)] = H[(i, j)] + u;
                    } else {
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

pub trait TinySolver<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    const NUM_PARAMETERS: usize = NUM_PARAMETERS;
    const NUM_RESIDUALS: usize = NUM_RESIDUALS;
    fn cost_function<D: num_dual::DualNum<f64>>(
        _params: &mut na::SVector<D, NUM_PARAMETERS>,
        _residual: &mut na::SMatrix<D, NUM_RESIDUALS, 1>,
    ) {
        unimplemented!()
    }
    fn solve(params: &mut na::SMatrix<f64, NUM_PARAMETERS, 1>) -> bool {
        let mut status = Status::defualt();
        let mut u: f64;
        let mut v = 2;
        let mut residual = na::SMatrix::<f64, NUM_RESIDUALS, 1>::zeros();
        let mut jac = na::SMatrix::<f64, NUM_RESIDUALS, NUM_PARAMETERS>::zeros();
        for i in 0..status.max_iterations {
            Self::cost_function(params, &mut residual);
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
                    if i == j {
                        jtj[(i, j)] = H[(i, j)] + u;
                    } else {
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

//     let (x, y) = (5.0, 4.0);
//     // Calculate a simple derivative using dual numbers
//     let x_dual = Dual64::from(x).derivative();
//     let y_dual = Dual64::from(y);
//     println!("{}", f(x_dual, y_dual)); // 2000 + [1200]ε

//     // or use the provided function instead
//     let (_, df) = first_derivative(|x| f(x, y.into()), x);
//     println!("{df}"); // 1200

//     // Calculate a gradient
//     let (value, grad) = gradient(|v| f(v[0], v[1]), SMatrix::from([x, y]));
//     println!("{value} {grad}"); // 2000 [1200, 1000]

//     // Calculate a Hessian
//     let (_, _, hess) = hessian(|v| f(v[0], v[1]), SMatrix::from([x, y]));
//     println!("{hess}"); // [[480, 600], [600, 250]]

//     // for x=cos(t) and y=sin(t) calculate the third derivative w.r.t. t
//     let (_, _, _, d3f) = third_derivative(|t| f(t.cos(), t.sin()), 1.0);
//     println!("{d3f}"); // 7.358639755305733
// }

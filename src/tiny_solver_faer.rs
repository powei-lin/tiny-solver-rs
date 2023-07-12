use crate::status::Status;
use faer_core::Mat;
use faer_core::MatMut;
use faer_lu::partial_pivoting::compute::lu_in_place;
use faer_lu::partial_pivoting::solve::solve;
use faer_lu::partial_pivoting::solve::solve_in_place;

// pub trait TinySolverFAER<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize>{
//     const NUM_PARAMETERS: usize = NUM_PARAMETERS;
//     const NUM_RESIDUALS: usize = NUM_RESIDUALS;
//     fn cost_function(
//         _params: &mut Mat<f64>,
//         _residual: &mut Mat<f64>,
//         _jacobian: Option<&mut Mat<f64>>,
//     ) {
//         unimplemented!()
//     }
//     fn solve(params: &mut Mat<f64>) -> bool {
//         let mut status = Status::init();
//         let mut u: f64;
//         let mut v = 2;
//         let mut residual = Mat::<f64>::zeros(NUM_RESIDUALS, 1);
//         let mut jac = Mat::<f64>::zeros(NUM_RESIDUALS, NUM_PARAMETERS);
//         for i in 0..status.max_iterations {
//             Self::cost_function(params, &mut residual, Some(&mut jac));
//             // faer_core::
//             let g_:MatMut<f64> = (jac.transpose() * residual).as_mut();
//             g_.cwise().for_each(|mut x|x.write(-x.read()));
//             let H = jac.transpose() * jac.clone();
//             // println!("residual \n{}\n", residual);
//             // println!("jac \n{}\n", &jac);
//             // println!("H:\n{}", H);
//             // println!("gradient:\n{}", g_);

//             let mut max_gradient = 0.0;
//             g_.cwise().for_each(|x| {
//                 let absx = x.read().abs();
//                 if absx > max_gradient{
//                     max_gradient = absx;
//                 }
//             });
//             println!("mg{}", max_gradient);
//             if max_gradient < status.gradient_threshold {
//                 println!("gradient too small. {}", max_gradient);
//                 break;
//             }

//             let mut max_d = 0.0;
//             H.as_mut().diagonal().cwise().for_each(|x|{
//                 if x.read() > max_d{
//                     max_d = x.read();
//                 }
//             });
//             u = status.initial_scale_factor * max_d;
//             v = 2;
//             println!("u: {}", u);
//             let mut jtj = Mat::<f64>::zeros(NUM_PARAMETERS, NUM_PARAMETERS).as_mut();
//             for i in 0..NUM_PARAMETERS {
//                 for j in 0..NUM_PARAMETERS {
//                     if i == j{
//                         jtj.write(i, j, H.read_unchecked(i, j) + u);
//                     }
//                     else{
//                         jtj.write(i, j, H.read_unchecked(i, j));
//                     }
//                 }
//             }

//             // // println!("jtj {}", jtj);
//             // let a = Mat::with_dims(n, n, |_, _| gen());
//             // let mut lu = a.clone();
//             // let a = a.as_ref();
//             // let mut lu = lu.as_mut();

//             // let k = 32;
//             // let rhs = Mat::with_dims(n, k, |_, _| gen());
//             // let rhs = rhs.as_ref();
//             // let mut sol = Mat::<f64>::zeros(NUM_PARAMETERS, 1).as_mut();

//             // let mut row_perm = vec![0_usize; n];
//             // let mut row_perm_inv = vec![0_usize; n];

//             // let parallelism = Parallelism::Rayon(0);

//             // let (_, row_perm) = lu_in_place(
//             //     lu.rb_mut(),
//             //     &mut row_perm,
//             //     &mut row_perm_inv,
//             //     parallelism,
//             //     make_stack!(lu_in_place_req::<E>(n, n, parallelism, Default::default())),
//             //     Default::default(),
//             // );

//             // solve_in_place(
//             //     sol.rb_mut(),
//             //     lu.rb(),
//             //     conj_lhs,
//             //     row_perm.rb(),
//             //     rhs,
//             //     parallelism,
//             //     make_stack!(solve_req::<E>(n, n, k, parallelism)),
//             // );
//             // let dx = na::linalg::LU::new(jtj).solve(&g_).unwrap();
//             // *params += dx;
//         }
//         println!("x0 {:#?}", params);

//         true
//     }

// }

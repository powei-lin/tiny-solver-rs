// use num_dual::DualNum;
// use tiny_solver::TinySolver;
// extern crate nalgebra as na;
// use std::ops::Mul;

// struct TestProblem;

// impl TinySolver<3, 2> for TestProblem {
//     fn cost_function(
//         params: nalgebra::SVector<num_dual::DualSVec64<3>, 3>,
//     ) -> nalgebra::SVector<num_dual::DualSVec64<3>, 2> {
//         let x = params[0];
//         let y = params[1];
//         let z = params[2];
//         return nalgebra::SVector::from([x + y.mul(3.0) + z.powf(1.1), y * z]);
//     }
// }

// #[test]
// fn test_residual() {
//     let xvec = na::SVector::from([5.0, 3.0, 2.0]);
//     let residual = TestProblem::cost_function(xvec.map(num_dual::DualVec::from_re));
//     assert_eq!(residual[0].re, 16.143546925072584);
//     assert_eq!(residual[1].re, 6.0);
// }

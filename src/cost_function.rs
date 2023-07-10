extern crate nalgebra as na;

use num_traits;
pub trait CostFunc<const NUM_PARAMETERS: usize, const NUM_RESIDUALS: usize> {
    type T: na::Scalar + num_traits::identities::Zero;
    const NUM_PARAMETERS: usize = NUM_PARAMETERS;
    const NUM_RESIDUALS: usize = NUM_RESIDUALS;
    fn apply(
        _params: &mut na::SMatrix<Self::T, NUM_PARAMETERS, 1>,
        _residual: &mut na::SMatrix<Self::T, NUM_RESIDUALS, 1>,
        _jacobian: Option<&mut na::SMatrix<Self::T, NUM_RESIDUALS, NUM_PARAMETERS>>,
    ) {
        unimplemented!()
    }
}

struct CC {
    NUM_PARAMETERS: usize,
    NUM_RESIDUALS: usize,
}

// impl CostFunc<3, 2> for CC {
//     type T = f64;
//     fn apply(
//         params: &mut na::SMatrix<Self::T, 3, 1>,
//         residual: &mut na::SMatrix<Self::T, 2, 1>,
//         jacobian: Option<&mut na::SMatrix<Self::T, 2, 3>>,
//     ) {
//         let x = params[0];
//         let y = params[1];
//         let z = params[2];
//         *residual[0] = x + 2.0 * y + 4.0 * z;
//         *residual[1] = y * z;
//         if let Some(jacobian) = jacobian {
//             *jacobian[(0, 0)] = 1.0;
//             *jacobian[(0, 0)] = 2.0;
//             *jacobian[(0, 0)] = 4.0;
//             *jacobian[(0, 0)] = 0.0;
//             *jacobian[(0, 0)] = z;
//             *jacobian[(0, 0)] = y;
//         }
//     }
// }

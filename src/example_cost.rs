use crate::cost_function;

pub type Matrix2x1d = nalgebra::SMatrix<f64, 2, 1>;
pub type Matrix3x1d = nalgebra::SMatrix<f64, 3, 1>;
pub type Matrix2x3d = nalgebra::SMatrix<f64, 2, 3>;
pub struct ExampleStatic {}

impl cost_function::TinySolverF64<3, 2> for ExampleStatic {
    fn cost_function(
        params: &mut nalgebra::SMatrix<f64, { Self::NUM_PARAMETERS }, 1>,
        residual: &mut nalgebra::SMatrix<f64, { Self::NUM_RESIDUALS }, 1>,
        jacobian: Option<
            &mut nalgebra::SMatrix<f64, { Self::NUM_RESIDUALS }, { Self::NUM_PARAMETERS }>,
        >,
    ) {
        let x = params[0];
        let y = params[1];
        let z = params[2];
        *residual = Matrix2x1d::new(x + 2.0 * y + 4.0 * z, y * z);
        // residual[1] = y * z;
        if let Some(jacobian) = jacobian {
            *jacobian = Matrix2x3d::new(1.0, 2.0, 4.0, 0.0, z, y);
        }
    }

}

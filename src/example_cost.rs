pub mod cost_trait;
pub struct ExampleStatic {
}

impl cost_trait::CostFunc<f64, 3, 2> for ExampleStatic {
    fn apply(params: &[f64; 3], residual: &mut faer_core::Mat<f64>, jacobian: Option<&mut faer_core::Mat<f64>>) {
        let x = params[0];
        let y = params[1];
        let z = params[2];
        *residual = faer_core::mat![[x + 2.0 * y + 4.0 * z],[y*z]];
        // residual[1] = y * z;
        if let Some(jacobian) = jacobian {
            *jacobian = faer_core::mat![[1.0, 2.0, 4.0], [0.0, z, y]];
        }
    }
}

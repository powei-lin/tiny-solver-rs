use nalgebra as na;

use crate::corrector::Corrector;
use crate::factors::{DualStride, FactorImpl, STRIDE};
use crate::loss_functions::Loss;
use crate::parameter_block::ParameterBlock;

pub struct ResidualBlock {
    pub residual_block_id: usize,
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub factor: Box<dyn FactorImpl + Send>,
    pub loss_func: Option<Box<dyn Loss + Send>>,
}
impl ResidualBlock {
    pub fn new(
        residual_block_id: usize,
        dim_residual: usize,
        residual_row_start_idx: usize,
        variable_key_size_list: &[&str],
        factor: Box<dyn FactorImpl + Send>,
        loss_func: Option<Box<dyn Loss + Send>>,
    ) -> Self {
        ResidualBlock {
            residual_block_id,
            dim_residual,
            residual_row_start_idx,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|s| s.to_string())
                .collect(),
            factor,
            loss_func,
        }
    }

    pub fn residual(&self, params: &[&ParameterBlock], with_loss_fn: bool) -> na::DVector<f64> {
        let param_vec: Vec<_> = params.iter().map(|p| p.params.clone()).collect();
        let mut residual = self.factor.residual_func_f64(&param_vec);
        self.check_residual_count(residual.nrows());
        let squared_norm = residual.norm_squared();
        if with_loss_fn {
            if let Some(loss_func) = self.loss_func.as_ref() {
                let rho = loss_func.evaluate(squared_norm);
                // let cost = 0.5 * rho[0];
                let corrector = Corrector::new(squared_norm, &rho);
                corrector.correct_residuals(&mut residual);
            }
        } else {
            // let cost = 0.5 * squared_norm;
        }
        residual
    }
    /// The cost of this block: rho(||r||^2) with a loss function, ||r||^2
    /// without one.
    pub fn cost(&self, params: &[&ParameterBlock]) -> f64 {
        let param_vec: Vec<_> = params.iter().map(|p| p.params.clone()).collect();
        let squared_norm = self.factor.residual_func_f64(&param_vec).norm_squared();
        match self.loss_func.as_ref() {
            Some(loss_func) => loss_func.evaluate(squared_norm)[0],
            None => squared_norm,
        }
    }
    pub fn residual_and_jacobian(
        &self,
        params: &[&ParameterBlock],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let dim_variable = params.iter().map(|x| x.tangent_size()).sum::<usize>();
        let mut residual = na::DVector::<f64>::zeros(self.dim_residual);
        let mut jacobian = na::DMatrix::<f64>::zeros(self.dim_residual, dim_variable);
        // The pass at `lo` seeds the tangent directions [lo, lo + STRIDE), as Ceres'
        // DynamicAutoDiffCostFunction does. One pass always runs, for the residual.
        for lo in (0..dim_variable.max(1)).step_by(STRIDE) {
            let mut offset = 0;
            let inputs: Vec<na::DVector<DualStride>> = params
                .iter()
                .map(|param| {
                    let n = param.tangent_size();
                    let delta = na::DVector::from_fn(n, |j, _| {
                        let d = offset + j;
                        if d >= lo && d < lo + STRIDE {
                            DualStride::new(
                                0.0,
                                num_dual::Derivative::derivative_generic(
                                    na::Const::<STRIDE>,
                                    na::Const::<1>,
                                    d - lo,
                                ),
                            )
                        } else {
                            DualStride::from_re(0.0)
                        }
                    });
                    offset += n;
                    param.plus_dual(delta.as_view())
                })
                .collect();
            let r = self.factor.residual_func_dual(&inputs);
            self.check_residual_count(r.nrows());
            for (i, x) in r.iter().enumerate() {
                residual[i] = x.re;
                let e = x.eps.unwrap_generic(na::Const::<STRIDE>, na::Const::<1>);
                for k in 0..STRIDE.min(dim_variable - lo) {
                    jacobian[(i, lo + k)] = e[k];
                }
            }
        }
        let squared_norm = residual.norm_squared();
        if let Some(loss_func) = self.loss_func.as_ref() {
            let rho = loss_func.evaluate(squared_norm);
            // let cost = 0.5 * rho[0];
            let corrector = Corrector::new(squared_norm, &rho);
            corrector.correct_jacobian(&residual, &mut jacobian);
            corrector.correct_residuals(&mut residual);
        } else {
            // let cost = 0.5 * squared_norm;
        }
        (residual, jacobian)
    }

    fn check_residual_count(&self, returned: usize) {
        assert_eq!(
            returned, self.dim_residual,
            "residual block {}: the factor returned {} residuals, but was added with dim_residual = {}",
            self.residual_block_id, returned, self.dim_residual
        );
    }
}

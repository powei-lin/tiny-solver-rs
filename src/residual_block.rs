use nalgebra as na;
use rayon::prelude::*;

use crate::corrector::Corrector;
use crate::factors::FactorImpl;
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
        variable_key_size_list: &[(&str, usize)],
        factor: Box<dyn FactorImpl + Send>,
        loss_func: Option<Box<dyn Loss + Send>>,
    ) -> Self {
        ResidualBlock {
            residual_block_id,
            dim_residual,
            residual_row_start_idx,
            variable_key_list: variable_key_size_list
                .iter()
                .map(|s| s.0.to_string())
                .collect(),
            factor,
            loss_func,
        }
    }

    pub fn residual(&self, params: &[na::DVector<f64>], with_loss_fn: bool) -> na::DVector<f64> {
        let mut residual = self.factor.residual_func_f64(params);
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
    pub fn residual_and_jacobian(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let variable_rows: Vec<usize> = params.iter().map(|x| x.shape().0).collect();
        let dim_variable = variable_rows.iter().sum::<usize>();
        let variable_row_idx_vec = get_variable_rows(&variable_rows);
        let indentity_mat = na::DMatrix::<f64>::identity(dim_variable, dim_variable);
        let params_with_dual: Vec<na::DVector<num_dual::DualDVec64>> = params
            .par_iter()
            .enumerate()
            .map(|(i, param)| {
                na::DVector::from_row_iterator(
                    param.nrows(),
                    param.row_iter().enumerate().map(|(j, x)| {
                        num_dual::DualDVec64::new(
                            x[0],
                            num_dual::Derivative::some(na::DVector::from(
                                indentity_mat.column(variable_row_idx_vec[i][j]),
                            )),
                        )
                    }),
                )
            })
            .collect();
        let residual_with_jacobian = self.factor.residual_func_dual(&params_with_dual);
        let mut residual = residual_with_jacobian.map(|x| x.re);
        let jacobian = residual_with_jacobian
            .map(|x| x.eps.unwrap_generic(na::Dyn(dim_variable), na::Const::<1>));
        let mut jacobian =
            na::DMatrix::<f64>::from_fn(residual_with_jacobian.nrows(), dim_variable, |r, c| {
                jacobian[r][c]
            });
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
    pub fn residual_and_jacobian2(
        &self,
        params: &[&ParameterBlock],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let variable_rows: Vec<usize> = params.iter().map(|x| x.tangent_size()).collect();
        let dim_variable = variable_rows.iter().sum::<usize>();
        let variable_row_idx_vec = get_variable_rows(&variable_rows);
        let indentity_mat = na::DMatrix::<f64>::identity(dim_variable, dim_variable);

        // ambient size
        let params_plus_tangent_dual: Vec<na::DVector<num_dual::DualDVec64>> = params
            .par_iter()
            .enumerate()
            .map(|(param_idx, param)| {
                let zeros_with_dual = na::DVector::from_row_iterator(
                    param.tangent_size(),
                    (0..param.tangent_size()).map(|j| {
                        num_dual::DualDVec64::new(
                            0.0,
                            num_dual::Derivative::some(na::DVector::from(
                                indentity_mat.column(variable_row_idx_vec[param_idx][j]),
                            )),
                        )
                    }),
                );
                let param_plus_dual = param.plus_dual(zeros_with_dual.as_view());
                param_plus_dual
                // na::DVector::from_row_iterator(
                //     param.ambient_size(),
                //     param.params.row_iter().enumerate().map(|(j, x)| {
                //         num_dual::DualDVec64::new(
                //             x[0],
                //             num_dual::Derivative::some(na::DVector::from(
                //                 indentity_mat.column(variable_row_idx_vec[param_idx][j]),
                //             )),
                //         )
                //     }),
                // )
            })
            .collect();

        // tangent size
        let residual_with_jacobian = self.factor.residual_func_dual(&params_plus_tangent_dual);
        let mut residual = residual_with_jacobian.map(|x| x.re);
        let jacobian = residual_with_jacobian
            .map(|x| x.eps.unwrap_generic(na::Dyn(dim_variable), na::Const::<1>));
        let mut jacobian =
            na::DMatrix::<f64>::from_fn(residual_with_jacobian.nrows(), dim_variable, |r, c| {
                jacobian[r][c]
            });
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
}

fn get_variable_rows(variable_rows: &[usize]) -> Vec<Vec<usize>> {
    let mut result = Vec::with_capacity(variable_rows.len());
    let mut current = 0;
    for &num in variable_rows {
        let next = current + num;
        let range = (current..next).collect();
        result.push(range);
        current = next;
    }
    result
}

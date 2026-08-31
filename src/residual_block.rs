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
        let parameter_values: Vec<_> = params.iter().map(|param| &param.params).collect();
        let mut residual = self
            .factor
            .residual_func_f64_refs(&parameter_values)
            .unwrap_or_else(|| {
                let owned_values: Vec<_> = parameter_values
                    .iter()
                    .map(|value| (*value).clone())
                    .collect();
                self.factor.residual_func_f64(&owned_values)
            });
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
        params: &[&ParameterBlock],
    ) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let parameter_values: Vec<_> = params.iter().map(|param| &param.params).collect();
        let analytic_evaluation = self
            .factor
            .residual_and_jacobians_f64_refs(&parameter_values)
            .or_else(|| {
                let owned_values: Vec<_> = parameter_values
                    .iter()
                    .map(|value| (*value).clone())
                    .collect();
                self.factor.residual_and_jacobians_f64(&owned_values)
            });
        let (mut residual, mut jacobian) =
            if let Some((residual, ambient_jacobians)) = analytic_evaluation {
                assert_eq!(
                    ambient_jacobians.len(),
                    params.len(),
                    "one analytic Jacobian is required per parameter block"
                );
                let tangent_size = params.iter().map(|param| param.tangent_size()).sum();
                let mut tangent_jacobian = na::DMatrix::zeros(residual.len(), tangent_size);
                let mut tangent_offset = 0;
                for ((parameter, ambient_jacobian), parameter_value) in
                    params.iter().zip(ambient_jacobians).zip(&parameter_values)
                {
                    assert_eq!(ambient_jacobian.nrows(), residual.len());
                    assert_eq!(ambient_jacobian.ncols(), parameter.ambient_size());
                    let block = if parameter.manifold.is_some() {
                        ambient_jacobian * parameter_plus_jacobian(parameter, parameter_value)
                    } else {
                        ambient_jacobian
                    };
                    tangent_jacobian
                        .view_mut(
                            (0, tangent_offset),
                            (residual.len(), parameter.tangent_size()),
                        )
                        .copy_from(&block);
                    tangent_offset += parameter.tangent_size();
                }
                (residual, tangent_jacobian)
            } else {
                self.autodiff_residual_and_jacobian(params)
            };
        self.apply_loss(&mut residual, &mut jacobian);
        (residual, jacobian)
    }

    fn autodiff_residual_and_jacobian(
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
                param.plus_dual(zeros_with_dual.as_view())
            })
            .collect();

        // tangent size
        let residual_with_jacobian = self.factor.residual_func_dual(&params_plus_tangent_dual);
        let residual = residual_with_jacobian.map(|x| x.re);
        let jacobian = residual_with_jacobian
            .map(|x| x.eps.unwrap_generic(na::Dyn(dim_variable), na::Const::<1>));
        let jacobian =
            na::DMatrix::<f64>::from_fn(residual_with_jacobian.nrows(), dim_variable, |r, c| {
                jacobian[r][c]
            });
        (residual, jacobian)
    }

    fn apply_loss(&self, residual: &mut na::DVector<f64>, jacobian: &mut na::DMatrix<f64>) {
        let squared_norm = residual.norm_squared();
        if let Some(loss_func) = self.loss_func.as_ref() {
            let rho = loss_func.evaluate(squared_norm);
            let corrector = Corrector::new(squared_norm, &rho);
            corrector.correct_jacobian(residual, jacobian);
            corrector.correct_residuals(residual);
        }
    }
}

fn parameter_plus_jacobian(
    parameter: &ParameterBlock,
    value: &na::DVector<f64>,
) -> na::DMatrix<f64> {
    let tangent_size = parameter.tangent_size();
    let tangent_identity = na::DMatrix::<f64>::identity(tangent_size, tangent_size);
    let delta = na::DVector::from_iterator(
        tangent_size,
        (0..tangent_size).map(|column| {
            num_dual::DualDVec64::new(
                0.0,
                num_dual::Derivative::some(na::DVector::from(tangent_identity.column(column))),
            )
        }),
    );
    let value = value.clone().cast::<num_dual::DualDVec64>();
    let retracted = if let Some(manifold) = &parameter.manifold {
        manifold.plus_dual(value.as_view(), delta.as_view())
    } else {
        value + delta
    };
    na::DMatrix::from_fn(parameter.ambient_size(), tangent_size, |row, column| {
        retracted[row]
            .eps
            .clone()
            .unwrap_generic(na::Dyn(tangent_size), na::Const::<1>)[column]
    })
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

use nalgebra as na;
use rayon::prelude::*;

use crate::factors::Factor;
use crate::loss_functions::Loss;

pub struct ResidualBlock {
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub factor: Box<dyn Factor + Send>,
    pub loss_func: Option<Box<dyn Loss + Send>>,
}
impl ResidualBlock {
    pub fn jacobian(&self, params: &Vec<na::DVector<f64>>) -> (na::DVector<f64>, na::DMatrix<f64>) {
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
        let residual_with_jacobian = self.factor.residual_func(&params_with_dual);
        let mut residual = residual_with_jacobian.map(|x| x.re);
        let jacobian = residual_with_jacobian
            .map(|x| x.eps.unwrap_generic(na::Dyn(dim_variable), na::Const::<1>));
        let mut jacobian =
            na::DMatrix::<f64>::from_fn(residual_with_jacobian.nrows(), dim_variable, |r, c| {
                jacobian[r][c]
            });
        if let Some(loss_func) = self.loss_func.as_ref() {
            loss_func.weight_residual_jacobian_in_place(&mut residual, &mut jacobian);
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

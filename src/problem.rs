use na::Dyn;
use num_dual;

extern crate nalgebra as na;
// use std::collections::HashMap;
pub struct Problem {
    _dim_variable: usize,
    _dim_residual: usize,
    // residual_blocks: Vec<ResidualBlock>,
    // variable_addr_to_col_idx_dict: HashMap<usize, usize>,
    // col_idx_to_variable_dict: HashMap<usize, usize>,
}

pub struct ResidualBlock {
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub residual_func:
        Box<dyn Fn(&Vec<na::DVector<num_dual::DualDVec64>>) -> na::DVector<num_dual::DualDVec64>>,
}
impl ResidualBlock {
    pub fn jacobian(self, params: &Vec<na::DVector<f64>>) -> (na::DVector<f64>, na::DMatrix<f64>) {
        let variable_rows: Vec<usize> = params.iter().map(|x| x.shape().0).collect();
        let dim_variable = variable_rows.iter().fold(0, |acc, x| acc + x);
        let variable_row_idx_vec = get_variable_rows(&variable_rows);
        let indentity_mat = na::DMatrix::<f64>::identity(dim_variable, dim_variable);
        let params_with_dual: Vec<na::DVector<num_dual::DualDVec64>> = params
            .iter()
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
        let residual_with_jacobian = (self.residual_func)(&params_with_dual);
        let residual = residual_with_jacobian.map(|x| x.re);
        let jacobian = residual_with_jacobian
            .map(|x| x.eps.unwrap_generic(na::Dyn(dim_variable), na::Const::<1>));
        let jacobian =
            na::DMatrix::<f64>::from_fn(residual_with_jacobian.nrows(), dim_variable, |r, c| {
                jacobian[r][c]
            });
        (residual, jacobian)
    }
}

// impl ResidualBlock {}

// trait Factor {
// fn cost_function(
//     _params: na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_PARAMETERS>,
// ) -> na::SVector<num_dual::DualSVec64<NUM_PARAMETERS>, NUM_RESIDUALS>
// }

fn get_variable_rows(variable_rows: &[usize]) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = 0;
    for &num in variable_rows {
        let next = current + num;
        let range = (current..next).collect::<Vec<_>>();
        result.push(range);
        current = next;
    }
    result
}

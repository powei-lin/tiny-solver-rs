use nalgebra as na;

pub trait Factor: Send + Sync {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64>;
}
pub struct ResidualBlock {
    pub dim_residual: usize,
    pub residual_row_start_idx: usize,
    pub variable_key_list: Vec<String>,
    pub factor: Box<dyn Factor + Send>,
}
impl ResidualBlock {
    pub fn jacobian(&self, params: &Vec<na::DVector<f64>>) -> (na::DVector<f64>, na::DMatrix<f64>) {
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
        let residual_with_jacobian = self.factor.residual_func(&params_with_dual);
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

use crate::problem;
use std::collections::HashMap;
use std::ops::Add;

extern crate nalgebra as na;

pub trait Optimizer {
    fn optimize(
        &self,
        problem: problem::Problem,
        initial_values: &HashMap<String, na::DVector<f64>>,
    ) -> HashMap<String, na::DVector<f64>>;
    fn apply_dx(
        &self,
        dx: &na::DVector<f64>,
        params: &mut HashMap<String, na::DVector<f64>>,
        variable_name_to_col_idx_dict: &HashMap<String, usize>,
    ) {
        println!("p start {:?}", params);
        for (key, param) in params.iter_mut() {
            if let Some(col_idx) = variable_name_to_col_idx_dict.get(key) {
                let var_size = param.shape().0;
                // updated_params.insert(key.to_string(), param.add(dx.rows(*col_idx, var_size)));
                let updated_param = param.clone().add(dx.rows(*col_idx, var_size));
                param.copy_from(&updated_param);
            }
        }
        println!("p end {:?}", params);
    }
}
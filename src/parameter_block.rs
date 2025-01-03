use std::collections::{HashMap, HashSet};

use nalgebra as na;

pub struct ParameterBlock {
    pub params: na::DVector<f64>,
    pub fixed_variables: HashSet<usize>,
    pub variable_bounds: HashMap<usize, (f64, f64)>,
}

impl ParameterBlock {
    pub fn from_vec(params: na::DVector<f64>) -> Self {
        ParameterBlock {
            params,
            fixed_variables: HashSet::new(),
            variable_bounds: HashMap::new(),
        }
    }
}

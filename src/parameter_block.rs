use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use nalgebra as na;
use num_dual::DualDVec64;

use crate::manifold::Manifold;

#[derive(Clone)]
pub struct ParameterBlock {
    pub params: na::DVector<f64>,
    pub fixed_variables: HashSet<usize>,
    pub variable_bounds: HashMap<usize, (f64, f64)>,
    pub manifold: Option<Arc<dyn Manifold + Sync + Send>>,
}

impl ParameterBlock {
    pub fn from_vec(params: na::DVector<f64>) -> Self {
        ParameterBlock {
            params,
            fixed_variables: HashSet::new(),
            variable_bounds: HashMap::new(),
            manifold: None,
        }
    }
    pub fn set_manifold(&mut self, manifold: Arc<dyn Manifold + Sync + Send>) {
        self.manifold = Some(manifold);
    }
    pub fn ambient_size(&self) -> usize {
        self.params.shape().0
    }
    pub fn tangent_size(&self) -> usize {
        if let Some(m) = &self.manifold {
            m.tangent_size().get()
        } else {
            self.ambient_size()
        }
    }
    pub fn plus_f64(&self, dx: na::DVectorView<f64>) -> na::DVector<f64> {
        let mut new_param = na::DVector::zeros(self.ambient_size());
        if let Some(m) = &self.manifold {
            new_param = m.plus_f64(self.params.as_view(), dx);
        } else {
            self.params.add_to(&dx, &mut new_param);
        }
        new_param
    }
    pub fn plus_dual(&self, dx: na::DVectorView<DualDVec64>) -> na::DVector<DualDVec64> {
        let mut new_param = na::DVector::zeros(self.ambient_size());
        if let Some(m) = &self.manifold {
            new_param = m.plus_dual(self.params.clone().cast::<DualDVec64>().as_view(), dx);
        } else {
            self.params.clone().cast().add_to(&dx, &mut new_param);
        }
        new_param
    }
    pub fn y_minus_f64(&self, y: na::DVectorView<f64>) -> na::DVector<f64> {
        let mut delta_x = na::DVector::zeros(self.tangent_size());
        if let Some(m) = &self.manifold {
            delta_x = m.minus_f64(y, self.params.as_view());
        } else {
            y.sub_to(&self.params, &mut delta_x);
        }
        delta_x
    }
    pub fn y_minus_dual(&self, y: na::DVectorView<DualDVec64>) -> na::DVector<DualDVec64> {
        let mut delta_x = na::DVector::zeros(self.tangent_size());
        if let Some(m) = &self.manifold {
            delta_x = m.minus_dual(y, self.params.clone().cast().as_view());
        } else {
            y.sub_to(&self.params.clone().cast(), &mut delta_x);
        }
        delta_x
    }
    pub fn update_params(&mut self, mut new_param: na::DVector<f64>) {
        // bound
        for (&idx, &(lower, upper)) in &self.variable_bounds {
            new_param[idx] = new_param[idx].max(lower).min(upper);
        }

        // fix
        for &index_to_fix in &self.fixed_variables {
            new_param[index_to_fix] = self.params[index_to_fix];
        }
        self.params = new_param;
    }
}

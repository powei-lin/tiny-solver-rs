use std::{
    collections::{HashMap, HashSet},
    num::NonZero,
};

use nalgebra as na;
use num_dual::DualDVec64;

pub trait AutoDiffManifold<T: na::RealField> {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T>;
    // fn minus
}

pub trait Manifold: AutoDiffManifold<f64> + AutoDiffManifold<num_dual::DualDVec64> {
    fn tangent_size(&self) -> NonZero<usize>;
    fn plus_f64(&self, x: na::DVectorView<f64>, delta: na::DVectorView<f64>) {
        self.plus(x, delta);
    }
    fn plus_dual(&self, x: na::DVectorView<DualDVec64>, delta: na::DVectorView<DualDVec64>) {
        self.plus(x, delta);
    }
}
// impl Manifold where Self: AutoDiffManifold<f64> + AutoDiffManifold<num_dual::DualDVec64>{}
// struct Q{}
// impl <T: na::RealField> AutoDiffManifold<T> for Q{
//     fn plus(&self, x: &nalgebra::DVector<T>) {

//     }
// }
// impl Manifold for Q {
//     fn tangent_size(&self) -> NonZero<usize> {
//         NonZero::new(6).unwrap()
//     }
// }

pub struct ParameterBlock {
    pub params: na::DVector<f64>,
    pub fixed_variables: HashSet<usize>,
    pub variable_bounds: HashMap<usize, (f64, f64)>,
    pub manifold: Option<Box<dyn Manifold + Sync>>,
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
    pub fn plus(&mut self, dx: na::DVectorView<f64>) {
        let mut new_param = na::DVector::zeros(self.ambient_size());
        if let Some(m) = &self.manifold {
            // m.plus(x, delta)
            panic!("Not implemented yet");
        } else {
            self.params.add_to(&dx, &mut new_param);
        }

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

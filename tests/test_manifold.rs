#[cfg(test)]
mod tests {
    use nalgebra as na;
    use std::f64::consts::PI;

    use tiny_solver::manifold::so3::SO3;

    fn equal_to_na(so3: &SO3<f64>) -> bool {
        let q = so3.to_vec();
        let na_so3 =
            na::UnitQuaternion::from_quaternion(na::Quaternion::new(q[3], q[0], q[1], q[2]));
        let diff = so3.log() - na_so3.scaled_axis();
        diff.norm() < 1e-6
    }

    #[test]
    fn test_so3() {
        for _ in 0..10000 {
            let mut rvec = na::DVector::new_random(3);
            rvec /= rvec.norm();
            rvec *= rand::random::<f64>() * PI;
            let r = SO3::exp(rvec.as_view());
            assert!(equal_to_na(&r));
        }
    }
}

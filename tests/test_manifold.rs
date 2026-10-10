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

    /// exp must agree with nalgebra to rounding error at any rotation angle:
    /// below the small-angle threshold of 1e-3, and beyond 2*pi.
    #[test]
    fn test_so3_exp_matches_nalgebra() {
        let axis = na::Vector3::new(1.0, -2.0, 0.5).normalize();
        let small = (1..100).map(|i| 1e-5 * i as f64);
        let large = (1..400).map(|i| 4.0 * PI * i as f64 / 400.0);
        for theta in small.chain(large) {
            let rvec = axis * theta;
            let q = SO3::exp(na::DVector::from_column_slice(rvec.as_slice()).as_view()).to_vec();
            let expected = na::UnitQuaternion::from_scaled_axis(rvec);
            let expected = na::Vector4::new(expected.i, expected.j, expected.k, expected.w);
            let diff = (q - expected).norm().min((q + expected).norm());
            assert!(diff < 1e-14, "theta = {theta}: {q} != +/-{expected}");
        }
    }
}

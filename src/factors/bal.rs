use nalgebra as na;

use super::{AnalyticFactor, Factor};

#[derive(Clone, Copy, Debug)]
pub struct SnavelyReprojectionFactor {
    pub observed_x: f64,
    pub observed_y: f64,
}

impl SnavelyReprojectionFactor {
    fn evaluate<T: na::RealField>(
        &self,
        camera: &na::DVector<T>,
        point_parameters: &na::DVector<T>,
    ) -> na::DVector<T> {
        let point = na::Vector3::new(
            point_parameters[0].clone(),
            point_parameters[1].clone(),
            point_parameters[2].clone(),
        );
        let angle_axis = na::Vector3::new(camera[0].clone(), camera[1].clone(), camera[2].clone());
        let theta_squared = angle_axis.norm_squared();
        let epsilon = T::from_f64(f64::EPSILON).unwrap();
        let rotated = if theta_squared > epsilon {
            let theta = theta_squared.sqrt();
            let axis = angle_axis / theta.clone();
            let cosine = theta.clone().cos();
            let sine = theta.sin();
            point.clone() * cosine.clone()
                + axis.cross(&point) * sine
                + axis.clone() * (axis.dot(&point) * (T::one() - cosine))
        } else {
            point.clone() + angle_axis.cross(&point)
        };

        let translated = na::Vector3::new(
            rotated[0].clone() + camera[3].clone(),
            rotated[1].clone() + camera[4].clone(),
            rotated[2].clone() + camera[5].clone(),
        );
        let projected_x = -translated[0].clone() / translated[2].clone();
        let projected_y = -translated[1].clone() / translated[2].clone();
        let radius_squared =
            projected_x.clone() * projected_x.clone() + projected_y.clone() * projected_y.clone();
        let distortion = T::one()
            + radius_squared.clone() * (camera[7].clone() + camera[8].clone() * radius_squared);
        let predicted_x = camera[6].clone() * distortion.clone() * projected_x;
        let predicted_y = camera[6].clone() * distortion * projected_y;

        na::dvector![
            predicted_x - T::from_f64(self.observed_x).unwrap(),
            predicted_y - T::from_f64(self.observed_y).unwrap()
        ]
    }
}

impl<T: na::RealField> Factor<T> for SnavelyReprojectionFactor {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        self.evaluate(&params[0], &params[1])
    }
}

impl AnalyticFactor for SnavelyReprojectionFactor {
    fn residual(&self, params: &[na::DVector<f64>]) -> na::DVector<f64> {
        self.evaluate(&params[0], &params[1])
    }

    fn residual_refs(&self, params: &[&na::DVector<f64>]) -> Option<na::DVector<f64>> {
        Some(self.evaluate(params[0], params[1]))
    }

    fn residual_and_jacobians(
        &self,
        params: &[na::DVector<f64>],
    ) -> (na::DVector<f64>, Vec<na::DMatrix<f64>>) {
        self.analytic_jacobians(&params[0], &params[1])
    }

    fn residual_and_jacobians_refs(
        &self,
        params: &[&na::DVector<f64>],
    ) -> Option<(na::DVector<f64>, Vec<na::DMatrix<f64>>)> {
        Some(self.analytic_jacobians(params[0], params[1]))
    }
}

impl SnavelyReprojectionFactor {
    fn analytic_jacobians(
        &self,
        camera: &na::DVector<f64>,
        point_parameters: &na::DVector<f64>,
    ) -> (na::DVector<f64>, Vec<na::DMatrix<f64>>) {
        assert_eq!(camera.len(), 9);
        assert_eq!(point_parameters.len(), 3);
        let point = na::Vector3::new(
            point_parameters[0],
            point_parameters[1],
            point_parameters[2],
        );
        let angle_axis = na::Vector3::new(camera[0], camera[1], camera[2]);
        let skew_angle_axis = skew(&angle_axis);
        let theta_squared = angle_axis.norm_squared();
        let (rotation, right_jacobian) = if theta_squared > 1e-12 {
            let theta = theta_squared.sqrt();
            let sine = theta.sin();
            let cosine = theta.cos();
            let skew_squared = skew_angle_axis * skew_angle_axis;
            (
                na::Matrix3::identity()
                    + skew_angle_axis * (sine / theta)
                    + skew_squared * ((1.0 - cosine) / theta_squared),
                na::Matrix3::identity() - skew_angle_axis * ((1.0 - cosine) / theta_squared)
                    + skew_squared * ((theta - sine) / (theta_squared * theta)),
            )
        } else {
            let skew_squared = skew_angle_axis * skew_angle_axis;
            (
                na::Matrix3::identity() + skew_angle_axis + skew_squared * 0.5,
                na::Matrix3::identity() - skew_angle_axis * 0.5 + skew_squared / 6.0,
            )
        };
        let rotated = rotation * point;
        let translated = rotated + na::Vector3::new(camera[3], camera[4], camera[5]);
        let projected_x = -translated.x / translated.z;
        let projected_y = -translated.y / translated.z;
        let radius_squared = projected_x * projected_x + projected_y * projected_y;
        let distortion = 1.0 + radius_squared * (camera[7] + camera[8] * radius_squared);
        let distortion_slope = camera[7] + 2.0 * camera[8] * radius_squared;
        let focal = camera[6];
        let projection_xy = na::Matrix2::new(
            focal * (distortion + 2.0 * projected_x * projected_x * distortion_slope),
            2.0 * focal * projected_x * projected_y * distortion_slope,
            2.0 * focal * projected_x * projected_y * distortion_slope,
            focal * (distortion + 2.0 * projected_y * projected_y * distortion_slope),
        );
        let inverse_z = 1.0 / translated.z;
        let normalized_projection = na::Matrix2x3::new(
            -inverse_z,
            0.0,
            translated.x * inverse_z * inverse_z,
            0.0,
            -inverse_z,
            translated.y * inverse_z * inverse_z,
        );
        let projection_point = projection_xy * normalized_projection;
        let rotation_point = -rotation * skew(&point) * right_jacobian;
        let mut camera_jacobian = na::DMatrix::zeros(2, 9);
        camera_jacobian
            .view_mut((0, 0), (2, 3))
            .copy_from(&(projection_point * rotation_point));
        camera_jacobian
            .view_mut((0, 3), (2, 3))
            .copy_from(&projection_point);
        camera_jacobian[(0, 6)] = distortion * projected_x;
        camera_jacobian[(1, 6)] = distortion * projected_y;
        camera_jacobian[(0, 7)] = focal * projected_x * radius_squared;
        camera_jacobian[(1, 7)] = focal * projected_y * radius_squared;
        camera_jacobian[(0, 8)] = focal * projected_x * radius_squared * radius_squared;
        camera_jacobian[(1, 8)] = focal * projected_y * radius_squared * radius_squared;
        let point_jacobian = projection_point * rotation;
        let residual = na::dvector![
            focal * distortion * projected_x - self.observed_x,
            focal * distortion * projected_y - self.observed_y
        ];

        (
            residual,
            vec![
                camera_jacobian,
                na::DMatrix::from_fn(2, 3, |row, col| point_jacobian[(row, col)]),
            ],
        )
    }
}

fn skew(vector: &na::Vector3<f64>) -> na::Matrix3<f64> {
    na::Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResidualBlock;
    use crate::factors::AnalyticFactorAdapter;
    use crate::parameter_block::ParameterBlock;

    #[test]
    fn zero_rotation_projection_has_finite_autodiff_jacobian() {
        for camera_values in [
            na::dvector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            na::dvector![0.1, -0.2, 0.05, 0.2, -0.1, 0.3, 2.0, 0.01, -0.001],
        ] {
            let camera = ParameterBlock::from_vec(camera_values);
            let point = ParameterBlock::from_vec(na::dvector![1.0, 2.0, -4.0]);
            let autodiff_block = ResidualBlock::new(
                0,
                2,
                0,
                &["camera", "point"],
                Box::new(SnavelyReprojectionFactor {
                    observed_x: 0.25,
                    observed_y: 0.75,
                }),
                None,
            );
            let analytic_block = ResidualBlock::new(
                1,
                2,
                0,
                &["camera", "point"],
                Box::new(AnalyticFactorAdapter::new(SnavelyReprojectionFactor {
                    observed_x: 0.25,
                    observed_y: 0.75,
                })),
                None,
            );

            let (residual, jacobian) = autodiff_block.residual_and_jacobian(&[&camera, &point]);
            let (analytic_residual, analytic_jacobian) =
                analytic_block.residual_and_jacobian(&[&camera, &point]);

            assert_eq!(jacobian.shape(), (2, 12));
            assert!(jacobian.iter().all(|value| value.is_finite()));
            assert!((&analytic_residual - residual).norm() < 1e-10);
            assert!((&analytic_jacobian - jacobian).norm() < 1e-9);
        }
    }
}

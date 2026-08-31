use std::collections::HashMap;
use std::fs::read_to_string;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::sync::Arc;

use nalgebra as na;

use crate::loss_functions::HuberLoss;
use crate::manifold::se3::SE3Manifold;
use crate::{ParameterBlockOrdering, factors, problem};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalObservation {
    pub camera_index: usize,
    pub point_index: usize,
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BalDataset {
    pub cameras: Vec<na::DVector<f64>>,
    pub points: Vec<na::DVector<f64>>,
    pub observations: Vec<BalObservation>,
}

impl BalDataset {
    pub fn into_problem(
        self,
    ) -> (
        problem::Problem,
        HashMap<String, na::DVector<f64>>,
        ParameterBlockOrdering,
    ) {
        let mut problem = problem::Problem::new();
        let mut initial_values = HashMap::with_capacity(self.cameras.len() + self.points.len());
        let mut ordering = ParameterBlockOrdering::new();
        let camera_keys: Vec<_> = (0..self.cameras.len())
            .map(|index| format!("camera_{index}"))
            .collect();
        let point_keys: Vec<_> = (0..self.points.len())
            .map(|index| format!("point_{index}"))
            .collect();

        for (key, camera) in camera_keys.iter().zip(self.cameras) {
            initial_values.insert(key.clone(), camera);
            ordering.add_element_to_group(key, 1);
        }
        for (key, point) in point_keys.iter().zip(self.points) {
            initial_values.insert(key.clone(), point);
            ordering.add_element_to_group(key, 0);
        }
        for observation in self.observations {
            let camera_key = &camera_keys[observation.camera_index];
            let point_key = &point_keys[observation.point_index];
            problem.add_residual_block(
                2,
                &[camera_key, point_key],
                Box::new(factors::AnalyticFactorAdapter::new(
                    factors::bal::SnavelyReprojectionFactor {
                        observed_x: observation.x,
                        observed_y: observation.y,
                    },
                )),
                None,
            );
        }

        (problem, initial_values, ordering)
    }
}

pub fn read_bal(path: impl AsRef<Path>) -> std::io::Result<BalDataset> {
    parse_bal(&read_to_string(path)?)
}

pub fn parse_bal(input: &str) -> std::io::Result<BalDataset> {
    let mut tokens = input.split_whitespace();
    let num_cameras = parse_bal_value(&mut tokens, "camera count")?;
    let num_points = parse_bal_value(&mut tokens, "point count")?;
    let num_observations = parse_bal_value(&mut tokens, "observation count")?;
    let mut observations = Vec::with_capacity(num_observations);
    for observation_index in 0..num_observations {
        let camera_index = parse_bal_value(&mut tokens, "camera index")?;
        let point_index = parse_bal_value(&mut tokens, "point index")?;
        let x = parse_bal_value(&mut tokens, "observed x")?;
        let y = parse_bal_value(&mut tokens, "observed y")?;
        if camera_index >= num_cameras || point_index >= num_points {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("observation {observation_index} has an out-of-range parameter index"),
            ));
        }
        observations.push(BalObservation {
            camera_index,
            point_index,
            x,
            y,
        });
    }

    let mut cameras = Vec::with_capacity(num_cameras);
    for _ in 0..num_cameras {
        cameras.push(parse_bal_vector(&mut tokens, 9, "camera parameter")?);
    }
    let mut points = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        points.push(parse_bal_vector(&mut tokens, 3, "point parameter")?);
    }
    if tokens.next().is_some() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "BAL file contains trailing values",
        ));
    }

    Ok(BalDataset {
        cameras,
        points,
        observations,
    })
}

fn parse_bal_vector(
    tokens: &mut std::str::SplitWhitespace<'_>,
    dimension: usize,
    label: &str,
) -> std::io::Result<na::DVector<f64>> {
    let values = (0..dimension)
        .map(|_| parse_bal_value(tokens, label))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(na::DVector::from_vec(values))
}

fn parse_bal_value<T>(tokens: &mut std::str::SplitWhitespace<'_>, label: &str) -> std::io::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let token = tokens
        .next()
        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, format!("missing {label}")))?;
    token.parse().map_err(|error| {
        Error::new(
            ErrorKind::InvalidData,
            format!("invalid {label} '{token}': {error}"),
        )
    })
}

pub fn translation_quaternion_to_na<T: na::RealField>(
    tx: &T,
    ty: &T,
    tz: &T,
    qx: &T,
    qy: &T,
    qz: &T,
    qw: &T,
) -> na::Isometry3<T> {
    let rotation = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
        qw.clone(),
        qx.clone(),
        qy.clone(),
        qz.clone(),
    ));
    na::Isometry3::from_parts(
        na::Translation3::new(tx.clone(), ty.clone(), tz.clone()),
        rotation,
    )
}

pub fn read_g2o(filename: &str) -> (problem::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = problem::Problem::new();
    let mut init_values = HashMap::<String, na::DVector<f64>>::new();
    for line in read_to_string(filename).unwrap().lines() {
        let line: Vec<&str> = line.split(' ').collect();
        match line[0] {
            "VERTEX_SE2" => {
                let x = line[2].parse::<f64>().unwrap();
                let y = line[3].parse::<f64>().unwrap();
                let theta = line[4].parse::<f64>().unwrap();
                init_values.insert(format!("x{}", line[1]), na::dvector![theta, x, y]);
            }
            "EDGE_SE2" => {
                let id0 = format!("x{}", line[1]);
                let id1 = format!("x{}", line[2]);
                let dx = line[3].parse::<f64>().unwrap();
                let dy = line[4].parse::<f64>().unwrap();
                let dtheta = line[5].parse::<f64>().unwrap();
                // todo add info matrix
                let edge = factors::BetweenFactorSE2 { dx, dy, dtheta };
                problem.add_residual_block(
                    3,
                    &[&id0, &id1],
                    Box::new(edge),
                    Some(Box::new(HuberLoss::new(1.0))),
                );
            }
            "VERTEX_SE3:QUAT" => {
                let x = line[2].parse::<f64>().expect("Failed to parse g2o");
                let y = line[3].parse::<f64>().expect("Failed to parse g2o");
                let z = line[4].parse::<f64>().expect("Failed to parse g2o");
                let qx = line[5].parse::<f64>().expect("Failed to parse g2o");
                let qy = line[6].parse::<f64>().expect("Failed to parse g2o");
                let qz = line[7].parse::<f64>().expect("Failed to parse g2o");
                let qw = line[8].parse::<f64>().expect("Failed to parse g2o");
                let var_name = format!("x{}", line[1]);
                problem.set_variable_manifold(&var_name, Arc::new(SE3Manifold));
                init_values.insert(var_name, na::dvector![qx, qy, qz, qw, x, y, z]);
            }
            "EDGE_SE3:QUAT" => {
                let id0 = format!("x{}", line[1]);
                let id1 = format!("x{}", line[2]);
                let dtx = line[3].parse::<f64>().expect("Failed to parse g2o");
                let dty = line[4].parse::<f64>().expect("Failed to parse g2o");
                let dtz = line[5].parse::<f64>().expect("Failed to parse g2o");
                let dqx = line[6].parse::<f64>().expect("Failed to parse g2o");
                let dqy = line[7].parse::<f64>().expect("Failed to parse g2o");
                let dqz = line[8].parse::<f64>().expect("Failed to parse g2o");
                let dqw = line[9].parse::<f64>().expect("Failed to parse g2o");
                let edge = factors::BetweenFactorSE3 {
                    dtx,
                    dty,
                    dtz,
                    dqx,
                    dqy,
                    dqz,
                    dqw,
                };
                problem.add_residual_block(
                    6,
                    &[&id0, &id1],
                    Box::new(edge),
                    Some(Box::new(HuberLoss::new(1.0))),
                );
            }
            _ => {
                println!("err");
                break;
            }
        }
    }
    let x0 = init_values.get("x0").unwrap();
    let origin_factor = factors::PriorFactor { v: x0.clone() };
    problem.add_residual_block(
        x0.shape().0,
        &["x0"],
        Box::new(origin_factor),
        Some(Box::new(HuberLoss::new(1.0))),
    );
    (problem, init_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bal_and_builds_point_first_problem() {
        let dataset = parse_bal("1 1 1\n0 0 0.25 0.75\n0 0 0 0 0 0 2 0 0\n1 2 -4\n").unwrap();

        assert_eq!(dataset.cameras.len(), 1);
        assert_eq!(dataset.points.len(), 1);
        assert_eq!(dataset.observations.len(), 1);
        assert_eq!(dataset.cameras[0][6], 2.0);
        assert_eq!(dataset.points[0], na::dvector![1.0, 2.0, -4.0]);

        let (problem, initial_values, ordering) = dataset.into_problem();
        assert_eq!(problem.num_residual_blocks(), 1);
        assert_eq!(problem.num_residuals(), 2);
        assert_eq!(initial_values.len(), 2);
        assert_eq!(ordering.group_id("point_0"), Some(0));
        assert_eq!(ordering.group_id("camera_0"), Some(1));
    }
}

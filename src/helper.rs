use std::collections::HashMap;
use std::fs::read_to_string;
use std::sync::Arc;

use nalgebra as na;

use crate::loss_functions::HuberLoss;
use crate::manifold::se3::SE3Manifold;
use crate::{factors, problem};

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

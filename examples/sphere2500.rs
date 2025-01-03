use std::collections::HashMap;
use std::fs::read_to_string;
use std::time::Instant;

use nalgebra as na;
use plotters::prelude::*;

use tiny_solver::{
    factors, loss_functions::HuberLoss, optimizer::Optimizer, problem, GaussNewtonOptimizer,
};

fn read_g2o(filename: &str) -> (problem::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = problem::Problem::new();
    let mut init_values = HashMap::<String, na::DVector<f64>>::new();
    for line in read_to_string(filename).unwrap().lines() {
        let line: Vec<&str> = line.split(' ').collect();
        match line[0] {
            "VERTEX_SE3:QUAT" => {
                let x = line[2].parse::<f64>().expect("Failed to parse g20");
                let y = line[3].parse::<f64>().expect("Failed to parse g20");
                let z = line[4].parse::<f64>().expect("Failed to parse g20");
                let qx = line[5].parse::<f64>().expect("Failed to parse g20");
                let qy = line[6].parse::<f64>().expect("Failed to parse g20");
                let qz = line[7].parse::<f64>().expect("Failed to parse g20");
                let qw = line[8].parse::<f64>().expect("Failed to parse g20");
                init_values.insert(
                    format!("x{}", line[1]),
                    na::dvector![x, y, z, qx, qy, qz, qw],
                );
            }
            "EDGE_SE3:QUAT" => {
                let id0 = format!("x{}", line[1]);
                let id1 = format!("x{}", line[2]);
                let dtx = line[3].parse::<f64>().expect("Failed to parse g20");
                let dty = line[4].parse::<f64>().expect("Failed to parse g20");
                let dtz = line[5].parse::<f64>().expect("Failed to parse g20");
                let dqx = line[6].parse::<f64>().expect("Failed to parse g20");
                let dqy = line[7].parse::<f64>().expect("Failed to parse g20");
                let dqz = line[8].parse::<f64>().expect("Failed to parse g20");
                let dqw = line[9].parse::<f64>().expect("Failed to parse g20");
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
                    7,
                    &[(&id0, 7), (&id1, 7)],
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
    let origin_factor = factors::PriorFactor {
        v: na::dvector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };
    problem.add_residual_block(
        7,
        &[("x0", 7)],
        Box::new(origin_factor),
        Some(Box::new(HuberLoss::new(1.0))),
    );
    (problem, init_values)
}

fn main() {
    // init logger
    env_logger::init();

    let (problem, init_values) = read_g2o("tests/data/sphere2500.g2o");
    let init_points: Vec<(f64, f64)> = init_values.values().map(|v| (v[1], v[2])).collect();
    let root_drawing_area = BitMapBackend::new("m3500_rs.png", (1024, 1024)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root_drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-50f64..50f64, -80f64..20f64)
        .unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .unwrap();
    scatter_ctx
        .draw_series(
            init_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
        )
        .unwrap();
    let gn = GaussNewtonOptimizer::new();
    let start = Instant::now();
    let result = gn.optimize(&problem, &init_values, None);
    let duration = start.elapsed();
    println!("Time elapsed in total is: {:?}", duration);
    let result_points: Vec<(f64, f64)> = result.unwrap().values().map(|v| (v[1], v[2])).collect();
    scatter_ctx
        .draw_series(
            result_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.filled())),
        )
        .unwrap();
}

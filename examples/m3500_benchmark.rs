use std::collections::HashMap;
use std::fs::read_to_string;
use std::time::Instant;

use nalgebra as na;
use plotters::prelude::*;

use tiny_solver::{
    factors, loss_functions::HuberLoss, optimizer::Optimizer, problem, GaussNewtonOptimizer,
    LevenbergMarquardtOptimizer,
};

fn read_g2o(filename: &str) -> (problem::Problem, HashMap<String, na::DVector<f64>>) {
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
                    &[(&id0, 3), (&id1, 3)],
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
        v: na::dvector![0.0, 0.0, 0.0],
    };
    problem.add_residual_block(
        3,
        &[("x0", 3)],
        Box::new(origin_factor),
        Some(Box::new(HuberLoss::new(1.0))),
    );
    (problem, init_values)
}

fn main() {
    // init logger
    env_logger::init();

    let (problem, init_values) = read_g2o("tests/data/input_M3500_g2o.g2o");
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

extern crate nalgebra as na;
use plotters::prelude::*;

use std::collections::{HashMap, HashSet};
use std::fs::read_to_string;
use tiny_solver::{gauss_newton_optimizer, optimizer::Optimizer, problem, residual_block};

fn read_g2o(filename: &str) -> (problem::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = problem::Problem::new();
    let mut init_values = HashMap::<String, na::DVector<f64>>::new();
    for line in read_to_string(filename).unwrap().lines() {
        let line: Vec<&str> = line.split(' ').collect();
        match line[0] {
            "VERTEX_SE2" => {
                println!("{:?}", line);
                let x = line[2].parse::<f64>().unwrap();
                let y = line[3].parse::<f64>().unwrap();
                let theta = line[4].parse::<f64>().unwrap();
                init_values.insert(format!("x{}", line[1]), na::dvector![theta, x, y]);
            }
            "EDGE_SE2" => {
                println!("{:?}", line);
                break;
            }
            _ => {
                println!("err");
                break;
            }
        }
    }
    (problem, init_values)
}

fn main() {
    let (problem, init_values) = read_g2o("tests/data/input_M3500_g2o.g2o");
    println!("{:?}", init_values);
    let random_points: Vec<(f64, f64)> = init_values.iter().map(|(_, v)| (v[1], v[2])).collect();
    let root_drawing_area = BitMapBackend::new("01.png", (1024, 1024)).into_drawing_area();

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
            random_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
        )
        .unwrap();
}

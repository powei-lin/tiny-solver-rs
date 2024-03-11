use std::time::Instant;
extern crate nalgebra as na;
use plotters::prelude::*;

use std::collections::HashMap;
use std::fs::read_to_string;
use tiny_solver::{gauss_newton_optimizer, optimizer::Optimizer, problem, residual_block};

#[derive(Default)]
pub struct CostFactorSE2 {
    dx: f64,
    dy: f64,
    dtheta: f64,
}
impl residual_block::Factor for CostFactorSE2 {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        let t_origin_k0 = &params[0];
        let t_origin_k1 = &params[1];
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(t_origin_k0[1].clone(), t_origin_k0[2].clone()),
            t_origin_k0[0].clone(),
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(t_origin_k1[1].clone(), t_origin_k1[2].clone()),
            t_origin_k1[0].clone(),
        );
        let se2_k0_k1 = na::Isometry2::new(
            na::Vector2::<num_dual::DualDVec64>::new(
                num_dual::DualDVec64::from_re(self.dx),
                num_dual::DualDVec64::from_re(self.dy),
            ),
            num_dual::DualDVec64::from_re(self.dtheta),
        );

        let se2_diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        return na::dvector![
            se2_diff.translation.x.clone(),
            se2_diff.translation.y.clone(),
            se2_diff.rotation.angle()
        ];
    }
}
#[derive(Default)]
pub struct BetweenFactor {}
impl residual_block::Factor for BetweenFactor {
    fn residual_func(
        &self,
        params: &Vec<na::DVector<num_dual::DualDVec64>>,
    ) -> na::DVector<num_dual::DualDVec64> {
        return params[0].clone();
    }
}

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
                let edge = CostFactorSE2 {
                    dx: dx,
                    dy: dy,
                    dtheta: dtheta,
                };
                problem.add_residual_block(3, vec![(id0, 3), (id1, 3)], Box::new(edge));
            }
            _ => {
                println!("err");
                break;
            }
        }
    }
    let origin_factor = BetweenFactor {};
    problem.add_residual_block(3, vec![("x0".to_string(), 3)], Box::new(origin_factor));
    (problem, init_values)
}

fn main() {
    let (problem, init_values) = read_g2o("tests/data/input_M3500_g2o.g2o");
    let init_points: Vec<(f64, f64)> = init_values.iter().map(|(_, v)| (v[1], v[2])).collect();
    let root_drawing_area = BitMapBackend::new("m3500_resul.png", (1024, 1024)).into_drawing_area();

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
    let gn = gauss_newton_optimizer::GaussNewtonOptimizer {};
    let start = Instant::now();
    let result = gn.optimize(problem, &init_values);
    let duration = start.elapsed();
    println!("Time elapsed in total is: {:?}", duration);
    let result_points: Vec<(f64, f64)> = result.iter().map(|(_, v)| (v[1], v[2])).collect();
    scatter_ctx
        .draw_series(
            result_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.filled())),
        )
        .unwrap();
}

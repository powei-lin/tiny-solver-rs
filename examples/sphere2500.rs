use std::time::Instant;

use plotters::prelude::*;

use tiny_solver::helper::read_g2o;
use tiny_solver::{optimizer::Optimizer, GaussNewtonOptimizer};

fn main() {
    // init logger
    env_logger::init();

    let (problem, init_values) = read_g2o("tests/data/sphere2500.g2o");
    let init_points: Vec<(f64, f64)> = init_values.values().map(|v| (v[4], v[6])).collect();
    let root_drawing_area =
        BitMapBackend::new("sphere2500_rs.png", (1024, 1024)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root_drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-60f64..60f64, -110f64..10f64)
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
    let result_points: Vec<(f64, f64)> = result.unwrap().values().map(|v| (v[4], v[6])).collect();
    scatter_ctx
        .draw_series(
            result_points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.filled())),
        )
        .unwrap();
}

use std::time::Instant;

use plotters::prelude::*;

use tiny_solver::{GaussNewtonOptimizer, helper::read_g2o, optimizer::Optimizer};

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

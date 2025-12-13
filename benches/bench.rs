use diol::prelude::*;
use tiny_solver::{GaussNewtonOptimizer, LevenbergMarquardtOptimizer, helper::read_g2o, optimizer::Optimizer};

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register_many(
        "g2o",
        list![
            bench_g2o_gn,
            bench_g2o_lm
        ],
        ["input_M3500_g2o", "parking-garage", "sphere2500"],
    );
    bench.run()?;
    Ok(())
}
fn bench_g2o_gn(bencher: Bencher, data_path: &str) {
    let data_path = format!("tests/data/{}.g2o", data_path);
    let (problem, init_values) = read_g2o(&data_path);
    let gn = GaussNewtonOptimizer::default();
    bencher.bench(|| {
        let result = gn.optimize(&problem, &init_values, None);
        std::hint::black_box(result);
    });
}

fn bench_g2o_lm(bencher: Bencher, data_path: &str) {
    let data_path = format!("tests/data/{}.g2o", data_path);
    let (problem, init_values) = read_g2o(&data_path);
    let gn = LevenbergMarquardtOptimizer::default();
    bencher.bench(|| {
        let result = gn.optimize(&problem, &init_values, None);
        std::hint::black_box(result);
    });
}
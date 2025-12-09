use diol::prelude::*;
use tiny_solver::{GaussNewtonOptimizer, helper::read_g2o, optimizer::Optimizer};

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register(
        "g2o",
        bench_g2o,
        ["input_M3500_g2o", "parking-garage", "sphere2500"],
    );
    bench.run()?;
    Ok(())
}
fn bench_g2o(bencher: Bencher, data_path: &str) {
    let data_path = format!("tests/data/{}.g2o", data_path);
    let (problem, init_values) = read_g2o(&data_path);
    let gn = GaussNewtonOptimizer::new();
    bencher.bench(|| {
        let result = gn.optimize(&problem, &init_values, None);
        std::hint::black_box(result);
    });
}

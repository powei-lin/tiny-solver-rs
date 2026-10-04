#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver::Optimizer;
    use tiny_solver::helper::read_g2o;
    use tiny_solver::optimizer::OptimizerOptions;

    /// r(x, y) = x[0] - y[0]
    struct DifferenceFactor;
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for DifferenceFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            na::dvector![params[0][0].clone() - params[1][0].clone()]
        }
    }

    /// Like ceres, which lays out parameter blocks in the order they are added,
    /// the columns follow the order in which the residual blocks first use each
    /// variable. Variables that no residual block uses go last, in name order.
    #[test]
    fn columns_follow_the_order_variables_are_first_used() {
        let mut problem = tiny_solver::Problem::new();
        let prior = |v: na::DVector<f64>| Box::new(tiny_solver::factors::PriorFactor { v });
        problem.add_residual_block(2, &["f"], prior(na::dvector![0.0, 0.0]), None);
        problem.add_residual_block(1, &["c", "e"], Box::new(DifferenceFactor), None);
        problem.add_residual_block(1, &["a"], prior(na::dvector![0.0]), None);
        problem.add_residual_block(1, &["e", "b"], Box::new(DifferenceFactor), None);
        problem.add_residual_block(1, &["d", "f"], Box::new(DifferenceFactor), None);
        let initial_values: HashMap<String, na::DVector<f64>> = [
            ("a", 1),
            ("b", 2),
            ("c", 1),
            ("d", 1),
            ("e", 3),
            ("f", 2),
            ("u", 1),
            ("z0", 2),
        ]
        .into_iter()
        .map(|(name, size)| (name.to_string(), na::DVector::zeros(size)))
        .collect();

        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let columns = problem.get_variable_name_to_col_idx_dict(&parameter_blocks);

        let expected = HashMap::from(
            [
                ("f", 0),
                ("c", 2),
                ("e", 3),
                ("a", 6),
                ("b", 7),
                ("d", 9),
                ("u", 10),
                ("z0", 11),
            ]
            .map(|(name, column)| (name.to_string(), column)),
        );
        assert_eq!(columns, expected);
    }

    /// Solving the same problem again must give bit-identical results, even
    /// though every HashMap gets its own random iteration order.
    #[test]
    fn repeated_solves_are_bitwise_identical() {
        let optimizers: [Box<dyn Optimizer>; 2] = [
            Box::new(tiny_solver::GaussNewtonOptimizer::default()),
            Box::new(tiny_solver::LevenbergMarquardtOptimizer::default()),
        ];
        for optimizer in optimizers {
            let solve = || {
                let (problem, initial_values) = read_g2o("tests/data/input_M3500_g2o.g2o");
                let options = OptimizerOptions {
                    max_iteration: 2,
                    ..Default::default()
                };
                optimizer
                    .optimize(&problem, &initial_values, Some(options))
                    .unwrap()
            };
            let bits =
                |v: &na::DVector<f64>| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };

            let reference = solve();
            for _ in 0..2 {
                let result = solve();
                for (key, value) in &reference {
                    assert_eq!(bits(&result[key]), bits(value), "variable {key}");
                }
            }
        }
    }
}

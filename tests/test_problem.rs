#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver;

    #[test]
    fn new_problem() {
        let problem = tiny_solver::Problem::new();
        assert_eq!(problem.total_residual_dimension, 0);
        assert_eq!(problem.fixed_variable_indexes.len(), 0);
        assert_eq!(problem.variable_bounds.len(), 0);
    }

    #[test]
    fn add_residual_block() {
        let mut problem = tiny_solver::Problem::new();
        let block_id1 = problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        assert_eq!(problem.total_residual_dimension, 1);

        let block_id2 = problem.add_residual_block(
            1,
            &["y"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        assert!(block_id1 != block_id2);
        assert_eq!(problem.total_residual_dimension, 2);
    }

    #[test]
    fn remove_residual_block() {
        let mut problem = tiny_solver::Problem::new();
        let block_id = problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        assert_eq!(problem.total_residual_dimension, 1);

        let mut block = problem.remove_residual_block(block_id);
        assert!(block.is_some());
        assert_eq!(problem.total_residual_dimension, 0);

        block = problem.remove_residual_block(block_id);
        assert!(block.is_none());
        assert_eq!(problem.total_residual_dimension, 0);
    }

    #[test]
    fn fix_variable() {
        let mut problem = tiny_solver::Problem::new();
        problem.fix_variable("x", 0);

        assert_eq!(problem.fixed_variable_indexes.len(), 1);
        assert_eq!(problem.fixed_variable_indexes["x"].len(), 1);
        assert!(problem.fixed_variable_indexes["x"].contains(&0));

        problem.fix_variable("x", 1);

        assert_eq!(problem.fixed_variable_indexes.len(), 1);
        assert_eq!(problem.fixed_variable_indexes["x"].len(), 2);
        assert!(problem.fixed_variable_indexes["x"].contains(&1));
    }

    #[test]
    fn unfix_variable() {
        let mut problem = tiny_solver::Problem::new();
        problem.fix_variable("x", 0);

        assert_eq!(problem.fixed_variable_indexes.len(), 1);
        assert_eq!(problem.fixed_variable_indexes["x"].len(), 1);
        assert!(problem.fixed_variable_indexes["x"].contains(&0));

        problem.unfix_variable("x");

        assert_eq!(problem.fixed_variable_indexes.len(), 0);
    }

    #[test]
    fn set_variable_bounds() {
        let mut problem = tiny_solver::Problem::new();

        problem.set_variable_bounds("x", 0, 0.0, 1.0);
        assert_eq!(problem.variable_bounds.len(), 1);
        assert_eq!(problem.variable_bounds["x"].len(), 1);
        assert!(problem.variable_bounds["x"].contains_key(&0));
        assert_eq!(problem.variable_bounds["x"][&0], (0.0, 1.0));
    }

    #[test]
    fn remove_variable_bounds() {
        let mut problem = tiny_solver::Problem::new();

        problem.set_variable_bounds("x", 0, 0.0, 1.0);
        assert_eq!(problem.variable_bounds.len(), 1);

        problem.remove_variable_bounds("x");
        assert_eq!(problem.variable_bounds.len(), 0);
    }

    #[test]
    fn compute_residual_and_jacobian() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        struct CustomFactor {}
        impl<T: na::RealField> tiny_solver::factors::Factor<T> for CustomFactor {
            fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
                println!("residual function: {:?}", params.len());
                let x = &params[0][0];
                let y = &params[1][0];
                let z = &params[1][1];

                na::dvector![
                    x.clone()
                        + y.clone() * T::from_f64(2.0).unwrap()
                        + z.clone() * T::from_f64(4.0).unwrap(),
                    y.clone() * z.clone()
                ]
            }
        }

        problem.add_residual_block(2, &["x", "yz"], Box::new(CustomFactor {}), None);

        // the initial values for x is 0.7 and yz is [-30.2, 123.4]
        let initial_values = HashMap::<String, na::DVector<f64>>::from([
            ("x".to_string(), na::dvector![0.7]),
            ("yz".to_string(), na::dvector![-30.2, 123.4]),
        ]);
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let variable_name_to_col_idx_dict =
            problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
        let total_variable_dimension = parameter_blocks.values().map(|p| p.tangent_size()).sum();
        let symbolic_structure = problem.build_symbolic_structure(
            &parameter_blocks,
            total_variable_dimension,
            &variable_name_to_col_idx_dict,
        );

        let (residuals, jac) = problem.compute_residual_and_jacobian(
            &parameter_blocks,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        );

        assert_eq!(residuals.nrows(), 3);
        assert_eq!(residuals.ncols(), 1);
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), 3);
    }
}

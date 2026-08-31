#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;

    #[test]
    fn new_problem() {
        let problem = tiny_solver::Problem::new();
        assert_eq!(problem.total_residual_dimension, 0);
        assert_eq!(problem.fixed_variable_indexes.len(), 0);
        assert_eq!(problem.variable_bounds.len(), 0);
    }

    #[test]
    fn parameter_layout_follows_groups_and_is_deterministic() {
        let problem = tiny_solver::Problem::new();
        let initial_values = HashMap::from([
            ("unlisted".to_string(), na::dvector![0.0]),
            ("camera".to_string(), na::dvector![0.0, 0.0, 0.0]),
            ("point".to_string(), na::dvector![0.0, 0.0]),
        ]);
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let mut ordering = tiny_solver::ParameterBlockOrdering::new();
        ordering.add_element_to_group("camera", 1);
        ordering.add_element_to_group("point", 0);

        let layout = problem
            .parameter_layout(&parameter_blocks, Some(&ordering))
            .unwrap();

        assert_eq!(layout.variable_name_to_col_idx["point"], 0);
        assert_eq!(layout.variable_name_to_col_idx["camera"], 2);
        assert_eq!(layout.variable_name_to_col_idx["unlisted"], 5);
        assert_eq!(layout.schur_elimination_dimension, 2);
        assert_eq!(layout.schur_elimination_block_sizes, [2]);
        assert_eq!(layout.schur_retained_block_sizes, [3, 1]);
        assert_eq!(layout.parameter_block_sizes, [2, 3, 1]);
        assert_eq!(layout.total_dimension, 6);

        let default_layout = problem.parameter_layout(&parameter_blocks, None).unwrap();
        assert_eq!(default_layout.schur_elimination_dimension, 0);
        assert!(default_layout.schur_elimination_block_sizes.is_empty());
        assert_eq!(default_layout.schur_retained_block_sizes, [3, 2, 1]);
    }

    #[test]
    fn parameter_layout_rejects_non_independent_schur_group() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["point_a", "point_b"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![0.0],
            }),
            None,
        );
        let initial_values = HashMap::from([
            ("point_a".to_string(), na::dvector![0.0]),
            ("point_b".to_string(), na::dvector![0.0]),
        ]);
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let mut ordering = tiny_solver::ParameterBlockOrdering::new();
        ordering.add_element_to_group("point_a", 0);
        ordering.add_element_to_group("point_b", 0);

        let error = problem
            .parameter_layout(&parameter_blocks, Some(&ordering))
            .unwrap_err();

        assert!(error.contains("not independent"));
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
    fn removing_a_residual_block_reindexes_remaining_rows() {
        let mut problem = tiny_solver::Problem::new();
        let first_id = problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![1.0],
            }),
            None,
        );
        let second_id = problem.add_residual_block(
            1,
            &["y"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![2.0],
            }),
            None,
        );

        problem.remove_residual_block(first_id).unwrap();

        assert_eq!(problem.num_residual_blocks(), 1);
        assert_eq!(problem.num_residuals(), 1);
        assert!(!problem.has_residual_block(first_id));
        assert!(problem.has_residual_block(second_id));
        assert_eq!(
            problem.residual_block_variable_keys(second_id).unwrap(),
            &["y"]
        );

        let initial_values = HashMap::from([("y".to_string(), na::dvector![5.0])]);
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let residuals = problem.compute_residuals(&parameter_blocks, true);
        assert_eq!(residuals[(0, 0)], 3.0);
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

    #[test]
    fn compute_residual_and_jacobian_with_fixed_variable() {
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
        problem.fix_variable("x", 0);

        // the initial values for x is 0.7 and yz is [-30.2, 123.4]
        let initial_values = HashMap::<String, na::DVector<f64>>::from([
            ("x".to_string(), na::dvector![0.7]),
            ("yz".to_string(), na::dvector![-30.2, 123.4]),
        ]);
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let variable_name_to_col_idx_dict =
            problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
        let total_variable_dimension = parameter_blocks
            .values()
            .map(|p| {
                if p.manifold.is_some() {
                    p.tangent_size()
                } else {
                    p.tangent_size() - p.fixed_variables.len()
                }
            })
            .sum();
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
        assert_eq!(jac.ncols(), 2); // x is fixed, so 3 - 1 = 2
    }
}

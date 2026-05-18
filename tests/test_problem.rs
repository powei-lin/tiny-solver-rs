#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver;
    use tiny_solver::Optimizer;

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

    // ─────────────────────────────────────────────
    // Schur complement marginalization tests
    // ─────────────────────────────────────────────

    /// Independent variables: prior x=1, prior y=2.
    /// Marginalize y → the resulting factor on x must recover x=1 when solved.
    #[test]
    fn marginalize_independent_variables() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![1.0],
            }),
            None,
        );
        problem.add_residual_block(
            1,
            &["y"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![2.0],
            }),
            None,
        );

        let initial_values = HashMap::from([
            ("x".to_string(), na::dvector![0.0]),
            ("y".to_string(), na::dvector![0.0]),
        ]);

        let marg = problem.marginalize(&initial_values, &["y"]);
        assert!(marg.is_some(), "marginalize should succeed");
        let marg = marg.unwrap();

        // Should only keep variable x
        assert_eq!(marg.variable_names, vec!["x".to_string()]);

        // Add the marginalization prior to a new problem and solve
        let mut new_problem = tiny_solver::Problem::new();
        let names: Vec<String> = marg.variable_names.clone();
        let dim = marg.sqrt_info.nrows();
        let names_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        new_problem.add_residual_block(dim, &names_refs, Box::new(marg), None);

        let new_initial = HashMap::from([("x".to_string(), na::dvector![0.0])]);
        let optimizer = tiny_solver::GaussNewtonOptimizer {};
        let result = optimizer.optimize(&new_problem, &new_initial, None).unwrap();

        let x = result["x"][0];
        assert!(
            (x - 1.0).abs() < 1e-10,
            "expected x ≈ 1.0, got {x}"
        );
    }

    /// Coupled system: r1 = x−1, r2 = x+y−3.
    /// Full solution: x=1, y=2.
    /// Marginalize y, add the resulting prior to a new problem, solve → x=1.
    #[test]
    fn marginalize_coupled_variables_and_solve() {
        // Coupling factor: x + y − 3 = 0
        struct SumFactor;
        impl<T: na::RealField> tiny_solver::factors::Factor<T> for SumFactor {
            fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
                let x = params[0][0].clone();
                let y = params[1][0].clone();
                na::dvector![x + y - T::from_f64(3.0).unwrap()]
            }
        }

        let mut problem = tiny_solver::Problem::new();
        // prior x = 1
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![1.0],
            }),
            None,
        );
        // coupling x + y = 3
        problem.add_residual_block(1, &["x", "y"], Box::new(SumFactor), None);

        let initial_values = HashMap::from([
            ("x".to_string(), na::dvector![0.0]),
            ("y".to_string(), na::dvector![0.0]),
        ]);

        // Verify the full problem solves correctly
        let optimizer = tiny_solver::GaussNewtonOptimizer {};
        let full_result = optimizer
            .optimize(&problem, &initial_values, None)
            .unwrap();
        assert!((full_result["x"][0] - 1.0).abs() < 1e-10);
        assert!((full_result["y"][0] - 2.0).abs() < 1e-10);

        // Marginalize y at the linearization point (0, 0)
        let marg = problem
            .marginalize(&initial_values, &["y"])
            .expect("marginalize should succeed");

        assert_eq!(marg.variable_names, vec!["x".to_string()]);
        assert_eq!(marg.sqrt_info.nrows(), 1);

        // Solve new problem using only the marginalization prior
        let mut new_problem = tiny_solver::Problem::new();
        let names: Vec<String> = marg.variable_names.clone();
        let dim = marg.sqrt_info.nrows();
        let names_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        new_problem.add_residual_block(dim, &names_refs, Box::new(marg), None);

        let result = optimizer
            .optimize(
                &new_problem,
                &HashMap::from([("x".to_string(), na::dvector![0.0])]),
                None,
            )
            .unwrap();

        let x = result["x"][0];
        assert!((x - 1.0).abs() < 1e-10, "expected x ≈ 1.0, got {x}");
    }

    /// Visual-odometry style: one 2-D camera pose connected to one 2-D landmark
    /// through a projection factor, plus a prior anchoring the pose at the origin.
    ///
    /// After marginalizing the landmark the residual prior on the pose should
    /// recover (tx, ty) = (0, 0).
    #[test]
    fn marginalize_visual_odometry_style() {
        // Projection factor: landmark − pose − observation = 0
        struct ProjectionFactor {
            obs_x: f64,
            obs_y: f64,
        }
        impl<T: na::RealField> tiny_solver::factors::Factor<T> for ProjectionFactor {
            fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
                // params[0] = pose (tx, ty), params[1] = landmark (lx, ly)
                let tx = params[0][0].clone();
                let ty = params[0][1].clone();
                let lx = params[1][0].clone();
                let ly = params[1][1].clone();
                na::dvector![
                    lx - tx - T::from_f64(self.obs_x).unwrap(),
                    ly - ty - T::from_f64(self.obs_y).unwrap()
                ]
            }
        }

        let mut problem = tiny_solver::Problem::new();
        // prior: pose anchored at origin
        problem.add_residual_block(
            2,
            &["pose"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![0.0, 0.0],
            }),
            None,
        );
        // projection: observed landmark at (3, 4) relative to the camera
        problem.add_residual_block(
            2,
            &["pose", "landmark"],
            Box::new(ProjectionFactor {
                obs_x: 3.0,
                obs_y: 4.0,
            }),
            None,
        );

        let initial_values = HashMap::from([
            ("pose".to_string(), na::dvector![0.0, 0.0]),
            ("landmark".to_string(), na::dvector![0.0, 0.0]),
        ]);

        // Verify full solution
        let optimizer = tiny_solver::GaussNewtonOptimizer {};
        let full_result = optimizer
            .optimize(&problem, &initial_values, None)
            .unwrap();
        assert!((full_result["pose"][0]).abs() < 1e-10);
        assert!((full_result["pose"][1]).abs() < 1e-10);
        assert!((full_result["landmark"][0] - 3.0).abs() < 1e-10);
        assert!((full_result["landmark"][1] - 4.0).abs() < 1e-10);

        // Marginalize the landmark at the initial linearization point
        let marg = problem
            .marginalize(&initial_values, &["landmark"])
            .expect("marginalize should succeed");

        assert_eq!(marg.variable_names, vec!["pose".to_string()]);
        assert_eq!(marg.sqrt_info.nrows(), 2);

        // Solve with the marginalization prior alone
        let mut new_problem = tiny_solver::Problem::new();
        let names: Vec<String> = marg.variable_names.clone();
        let dim = marg.sqrt_info.nrows();
        let names_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        new_problem.add_residual_block(dim, &names_refs, Box::new(marg), None);

        let pose_initial = HashMap::from([("pose".to_string(), na::dvector![0.0, 0.0])]);
        let result = optimizer
            .optimize(&new_problem, &pose_initial, None)
            .unwrap();

        let tx = result["pose"][0];
        let ty = result["pose"][1];
        assert!((tx).abs() < 1e-10, "expected tx ≈ 0.0, got {tx}");
        assert!((ty).abs() < 1e-10, "expected ty ≈ 0.0, got {ty}");
    }

    /// Verify that the Schur complement prior is consistent with the full problem.
    /// Marginalize y from a system and then optimize from a perturbed starting
    /// point — the prior should guide x to the same optimum as the full solve.
    #[test]
    fn marginalize_from_perturbed_start() {
        // Coupling: x − y = 0  (forces x == y when combined with priors)
        struct DiffFactor;
        impl<T: na::RealField> tiny_solver::factors::Factor<T> for DiffFactor {
            fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
                na::dvector![params[0][0].clone() - params[1][0].clone()]
            }
        }

        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![5.0],
            }),
            None,
        );
        problem.add_residual_block(
            1,
            &["y"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![5.0],
            }),
            None,
        );
        problem.add_residual_block(1, &["x", "y"], Box::new(DiffFactor), None);

        // Linearization point near the optimum
        let lin_point = HashMap::from([
            ("x".to_string(), na::dvector![4.5]),
            ("y".to_string(), na::dvector![4.5]),
        ]);

        let marg = problem
            .marginalize(&lin_point, &["y"])
            .expect("marginalize should succeed");

        // Start x away from the optimum
        let mut new_problem = tiny_solver::Problem::new();
        let names: Vec<String> = marg.variable_names.clone();
        let dim = marg.sqrt_info.nrows();
        let names_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        new_problem.add_residual_block(dim, &names_refs, Box::new(marg), None);

        let perturbed = HashMap::from([("x".to_string(), na::dvector![0.0])]);
        let optimizer = tiny_solver::GaussNewtonOptimizer {};
        let result = optimizer
            .optimize(&new_problem, &perturbed, None)
            .unwrap();

        // The prior encodes the information that x should be near 5.0
        let x = result["x"][0];
        assert!((x - 5.0).abs() < 0.1, "expected x near 5.0, got {x}");
    }
}


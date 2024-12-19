

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver;

    #[test]
    fn new_problem() {
        let problem = tiny_solver::Problem::new();
        assert_eq!(problem.total_variable_dimension, 0);
        assert_eq!(problem.total_residual_dimension, 0);
        assert_eq!(problem.variable_name_to_col_idx_dict.len(), 0);
        assert_eq!(problem.fixed_variable_indexes.len(), 0);
        assert_eq!(problem.variable_bounds.len(), 0);
    }

    #[test]
    fn add_residual_block() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &[("x", 1)],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        assert_eq!(problem.total_residual_dimension, 1);
        assert_eq!(problem.total_variable_dimension, 1);
        assert_eq!(problem.variable_name_to_col_idx_dict["x"], 0);
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
    fn combine_variables() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            1,
            &[("x", 1), ("y", 2)],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![3.0],
            }),
            None,
        );

        let variable_key_value_map = HashMap::from([("x".to_string(), na::dvector![0.0]), ("y".to_string(),  na::dvector![1.0, 2.0])]); 
        let combined_variables = problem.combine_variables(&variable_key_value_map);
        assert_eq!(combined_variables.len(), problem.total_variable_dimension);
        assert_eq!(combined_variables[0], 0.0);
        assert_eq!(combined_variables[1], 1.0);
        assert_eq!(combined_variables[2], 2.0);
    }
}

pub struct Status {
    pub gradient_threshold: f64,
    relative_step_threshold: f64,
    error_threshold: f64,
    pub initial_scale_factor: f64,
    pub max_iterations: usize,
}
impl Status {
    pub fn defualt() -> Status {
        Status {
            gradient_threshold: 1e-16,
            relative_step_threshold: 1e-16,
            error_threshold: 1e-16,
            initial_scale_factor: 1e-3,
            max_iterations: 100,
        }
    }
}

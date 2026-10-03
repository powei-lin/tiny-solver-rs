#[cfg(test)]
mod tests {
    use core::f64;
    use tiny_solver::loss_functions::*;

    #[test]
    fn arctan_loss() {
        let tolerance = 100.0;
        let asymptote = tolerance * f64::consts::PI / 2.0;

        let arctan_loss = ArctanLoss::new(tolerance);

        let rho1 = arctan_loss.evaluate(1.0);
        let rho2 = arctan_loss.evaluate(30.0);

        // Test that rho[0] grows linearly with the scale
        assert!(rho1[0] < rho2[0]);

        // Test that rho[1] decreases linearly with the scale
        assert!(rho1[1] > rho2[1]);

        // Test that scales largely above the tolerance are asymptotically bounded
        assert!(arctan_loss.evaluate(tolerance * tolerance * tolerance)[0] < asymptote);
    }

    /// rho[1] must be the derivative of rho[0], in both regions of each loss.
    #[test]
    fn rho_derivative_matches_rho() {
        let losses: [(&str, Box<dyn Loss>); 3] = [
            ("huber", Box::new(HuberLoss::new(0.5))),
            ("cauchy", Box::new(CauchyLoss::new(0.5))),
            ("arctan", Box::new(ArctanLoss::new(0.5))),
        ];
        for (name, loss) in &losses {
            for s in [0.01, 0.1, 1.0, 4.0, 25.0] {
                let h = 1e-6 * s;
                let numeric = (loss.evaluate(s + h)[0] - loss.evaluate(s - h)[0]) / (2.0 * h);
                let analytic = loss.evaluate(s)[1];
                assert!(
                    (numeric - analytic).abs() <= 1e-6 * analytic,
                    "{name}, s = {s}: rho' = {analytic}, numerical derivative of rho = {numeric}"
                );
            }
        }
    }
}

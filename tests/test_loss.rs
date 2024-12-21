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
}

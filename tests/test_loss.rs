#[cfg(test)]
mod tests {
    use core::f64;
    use tiny_solver::loss_functions::*;

    #[test]
    fn arctan_loss() {
        let tolerance = 100.0;
        let asymptote = tolerance * f64::consts::PI / 2.0;

        let arctan_loss = ArctanLoss::new(tolerance);
        
        assert!(arctan_loss.evaluate(1.0)[0] < arctan_loss.evaluate(30.0)[0]);
        assert!(arctan_loss.evaluate(tolerance * tolerance * tolerance)[0] < asymptote);
    }
}

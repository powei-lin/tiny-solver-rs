use std::ops::Mul;

use tiny_solver_rs::tiny_solver::TinySolver;

pub type Matrix3x1d = nalgebra::SMatrix<f64, 3, 1>;
pub struct ExampleStatic {}

impl TinySolver<3, 2> for ExampleStatic {
    fn cost_function(
        params: nalgebra::SVector<num_dual::DualSVec64<3>, 3>,
    ) -> nalgebra::SVector<num_dual::DualSVec64<3>, 2> {
        let x = params[0];
        let y = params[1];
        let z = params[2];
        return nalgebra::SVector::from([x + y.mul(2.0) + z.mul(4.0), y * z]);
    }
}
fn main() {
    let mut x0 = Matrix3x1d::new(0.76026643, -30.01799744, 0.55192142);
    let result = ExampleStatic::solve_inplace(&mut x0);
    println!("auto grad: {}", x0);
    println!(
        "residaul: {}",
        ExampleStatic::cost_function(x0.map(num_dual::DualSVec64::from_re))
    );
    println!("{:#?}", result);
}

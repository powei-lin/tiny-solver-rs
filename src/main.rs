mod tiny_solver;
mod example_cost;
extern crate nalgebra as na;

use tiny_solver::TinySolverF64;
use example_cost::{ExampleStatic, Matrix3x1d};

fn main() {
    let mut x0 = Matrix3x1d::new(0.76026643, -30.01799744, 0.55192142);
    ExampleStatic::solve(&mut x0);
}

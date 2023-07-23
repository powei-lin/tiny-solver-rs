# tiny-solver-rs
Inspired by ceres-solver and tiny-solver. This is a re-implemented version of [tiny-solver](https://github.com/keir/tinysolver/tree/master) in rust.

### Usage
You need to implement the cost function of `TinySolver<NUM_PARAMETERS, NUM_RESIDUALS>`.
Just like ceres-solver.

See full example in `examples/example_solve.rs`.
```rust
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
```

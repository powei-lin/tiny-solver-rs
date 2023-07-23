extern crate nalgebra as na;
use na::*;
use num_dual::*;
// use num_dual::{jacobian, DualSVec64, DualNum};

fn f<D: num_dual::DualNum<f64>>(x: D, y: D) -> D {
    // x.powd(x.clone()) + y
    x.powi(3) * y.powi(2)
}

fn f2(xy: na::SVector<num_dual::DualSVec64<3>, 3>) -> na::SVector<num_dual::DualSVec64<3>, 2> {
    return na::SVector::from([
        xy[0] * xy[1].powi(3) * xy[2],
        xy[0].powi(2) * xy[1] * xy[2].powi(2),
    ]);
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    println!("testttt");
    let xy = na::SVector::from([5.0, 3.0, 2.0]);
    let (f, jac) = num_dual::jacobian(f2, xy);
    println!("{f} {jac}");
    println!("{}", f2(xy.map(DualVec::from_re)));
    print_type_of(&jac);
    let mm = na::SMatrix::<f64, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    let mut m1 = na::DMatrix::<f64>::zeros(3, 3);
    m1.set_diagonal(&m1.diagonal().add_scalar(2.0));
    println!("m1 {m1}");
    print_type_of(&mm);
    assert_eq!(f[0], 270.0); // xy³z
    assert_eq!(f[1], 300.0); // x²yz²
    assert_eq!(jac[(0, 0)], 54.0); // y³z
    assert_eq!(jac[(0, 1)], 270.0); // 3xy²z
    assert_eq!(jac[(0, 2)], 135.0); // xy³
    assert_eq!(jac[(1, 0)], 120.0); // 2xyz²
    assert_eq!(jac[(1, 1)], 100.0); // x²z²
    assert_eq!(jac[(1, 2)], 300.0); // 2x²yz
}

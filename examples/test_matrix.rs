use core::num;
use num_dual;
use std::ops::Mul;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

extern crate nalgebra as na;
use na::{Const, Dyn, Vector};
pub type Matrix3x1d = na::SMatrix<f64, 3, 1>;
pub type Matrix1x1d = na::SMatrix<f64, 1, 1>;

fn dy_mat(a: &RefCell<&mut na::DVector<f64>>) {
    println!("{:?}", a);
}
fn dy_mat2(a: &Box<&mut na::DVector<f64>>, b: &Box<&mut na::DVector<f64>>) {
    if a.as_ref() == b.as_ref() {
        println!("the same.");
    }
    println!("{}", a);
}

fn cost_function(
    params: nalgebra::SVector<num_dual::DualSVec64<4>, 3>,
    params2: nalgebra::SVector<num_dual::DualSVec64<4>, 1>,
) -> nalgebra::SVector<num_dual::DualSVec64<4>, 3> {
    let x = params[0];
    let y = params[1];
    let z = params[2];
    let a = params2[0];
    return nalgebra::SVector::from([x + y.mul(2.0) + z.mul(4.0), y * z, a]);
}

fn cost_function_dyn(
    params: na::DVector<num_dual::DualDVec64>,
) -> na::DVector<num_dual::DualDVec64> {
    let x = params[0].clone();
    let y = params[1].clone();
    let z = params[2].clone();
    return na::dvector![x + y.clone().mul(2.0) + z.clone().mul(4.0), y * z];
}
fn main() {
    println!("hello");
    let mut x0 = Matrix3x1d::new(0.76026643, -30.01799744, 0.55192142);
    let mut x1 = Matrix1x1d::new(0.76026643);
    // num_dual::jacobian(cost_function, x0);
    // .map(num_dual::DualSVec64::from_re)).map(|x| x.re);
    let x = num_dual::DualSVec64::new(
        1.2,
        num_dual::Derivative::some(Vector::from([1.0, 0.0, 0.0, 0.0])),
    );
    let y = num_dual::DualSVec64::new(
        1.0,
        num_dual::Derivative::some(Vector::from([0.0, 1.0, 0.0, 0.0])),
    );
    let z = num_dual::DualSVec64::new(
        1.2,
        num_dual::Derivative::some(Vector::from([0.0, 0.0, 1.0, 0.0])),
    );
    let a = num_dual::DualSVec64::new(
        1.2,
        num_dual::Derivative::some(Vector::from([0.0, 0.0, 0.0, 1.0])),
    );

    let a = cost_function(na::SVector::from([x, y, z]), na::SVector::from([a]));
    println!("{}", a[1].eps.unwrap_generic(Const::<4>, Const::<1>));
    let x0 =
        num_dual::DualDVec64::new(1.2, num_dual::Derivative::some(na::dvector![1.0, 0.0, 0.0]));
    let y0 =
        num_dual::DualDVec64::new(1.2, num_dual::Derivative::some(na::dvector![0.0, 1.0, 0.0]));
    let z0 =
        num_dual::DualDVec64::new(1.2, num_dual::Derivative::some(na::dvector![0.0, 0.0, 1.0]));
    let a = cost_function_dyn(na::dvector![x0, y0, z0]);
    println!("{}", a[1].eps.clone().unwrap_generic(Dyn(3), Const::<1>));
    // let mut x1 = Matrix1x1d::new(0.2);
    // let mut aa = HashMap::<u32, u16>::new();
    // aa.insert(1, 1);
    // let mut a = na::DVector::from_element(3, 2.0);
    // let dv = RefCell::new(&mut a);
    // dy_mat(&dv);
    // dy_mat(&dv);
}

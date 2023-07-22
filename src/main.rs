extern crate nalgebra as na;
use num_dual;

fn f<D: num_dual::DualNum<f64>>(x: D, y: D) -> D {
    x.powd(x.clone()) + y
}

fn f2<D: num_dual::DualNum<f64>>(xvec: &na::SMatrix<D, 3, 1>) -> na::SMatrix<D, 2, 1> {
    let x = xvec[(0, 0)].clone();
    let y = xvec[(1, 0)].clone();
    let z = xvec[(2, 0)].clone();
    na::SMatrix::<D, 2, 1>::new(x + y.clone().mul(2.0) + z.clone().mul(4.0), y * z)
}

fn main() {
    println!("testttt");
    // let mut x0 = mat![[0.76026643], [-30.01799744], [0.55192142]];
    // let x = x0.read(0, 0);
    // let y = x0.read(1, 0);
    // let z = x0.read(2, 0);
    // let residual = mat![[x + 2.0 * y + 4.0 * z], [y * z]];
    // let jacobian = mat![[1.0, 2.0, 4.0], [0.0, z, y]];
    // println!("{}", jacobian.ncols());
    let xvec = na::SMatrix::<num_dual::Dual64, 3, 1>::new(
        num_dual::Dual::from(1.0),
        num_dual::Dual::from(1.0),
        num_dual::Dual::from(1.0),
    );
    let out = f2(&xvec);
    println!("{xvec}");
    println!("{out}");
    // let (x, y) = (5.0, 4.0);
    // // Calculate a simple derivative using dual numbers
    // let x_dual = num_dual::Dual64::from(x).derivative();
    // let y_dual = num_dual::Dual64::from(y);
    // println!("{}", f(x_dual, y_dual)); // 2000 + [1200]ε

    // // // or use the provided function instead
    // let (_, df) = num_dual::first_derivative(|x| f(x, y.into()), x);
    // println!("{df}"); // 1200

    // // Calculate a gradient
    // let (value, grad) = gradient(|v| f(v[0], v[1]), SMatrix::from([x, y]));
    // println!("{value} {grad}"); // 2000 [1200, 1000]

    // // Calculate a Hessian
    // let (_, _, hess) = hessian(|v| f(v[0], v[1]), SMatrix::from([x, y]));
    // println!("{hess}"); // [[480, 600], [600, 250]]

    // // for x=cos(t) and y=sin(t) calculate the third derivative w.r.t. t
    // let (_, _, _, d3f) = third_derivative(|t| f(t.cos(), t.sin()), 1.0);
    // println!("{d3f}"); // 7.358639755305733
}

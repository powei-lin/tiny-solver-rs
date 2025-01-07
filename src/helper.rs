use std::ops::Mul;

use nalgebra as na;

pub fn translation_quaternion_to_na<T: na::RealField>(
    tx: &T,
    ty: &T,
    tz: &T,
    qx: &T,
    qy: &T,
    qz: &T,
    qw: &T,
) -> na::Isometry3<T> {
    let rotation = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
        qw.clone(),
        qx.clone(),
        qy.clone(),
        qz.clone(),
    ));
    na::Isometry3::from_parts(
        na::Translation3::new(tx.clone(), ty.clone(), tz.clone()),
        rotation,
    )
}

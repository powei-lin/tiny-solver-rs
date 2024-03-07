extern crate nalgebra as na;

fn dy_mat(a:&Box<&mut na::DVector<f64>>){
    println!("{}", a);
}
fn main(){
    println!("hello");
    let mut a = na::DVector::from_element(3, 2.0);
    let dv = Box::new(&mut a);
    dy_mat(&dv);
    dy_mat(&dv);
}
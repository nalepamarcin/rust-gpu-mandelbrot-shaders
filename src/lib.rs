#![no_std]
#![feature(register_attr)]
#![register_attr(spirv)]

#[allow(unused_imports)]
use spirv_std;


#[spirv(compute(threads(32)))]
pub fn main() {}

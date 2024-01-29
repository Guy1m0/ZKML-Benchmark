use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

pub fn tensor() -> Tensor<FP16x16> {
    Tensor::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(41497, true), FixedTrait::<FP16x16>::new(59082, false), FixedTrait::<FP16x16>::new(3356, false), FixedTrait::<FP16x16>::new(28099, true), FixedTrait::<FP16x16>::new(5223, false), FixedTrait::<FP16x16>::new(114592, false), FixedTrait::<FP16x16>::new(9526, true), FixedTrait::<FP16x16>::new(73020, false), FixedTrait::<FP16x16>::new(143520, true), FixedTrait::<FP16x16>::new(18322, true)].span())
}

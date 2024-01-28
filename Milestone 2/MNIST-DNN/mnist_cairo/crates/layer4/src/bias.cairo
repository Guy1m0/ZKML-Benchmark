use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

pub fn tensor() -> Tensor<FP16x16> {
    Tensor::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(8069, true), FixedTrait::<FP16x16>::new(5495, false), FixedTrait::<FP16x16>::new(7349, false), FixedTrait::<FP16x16>::new(7470, true), FixedTrait::<FP16x16>::new(3300, false), FixedTrait::<FP16x16>::new(9144, false), FixedTrait::<FP16x16>::new(4965, true), FixedTrait::<FP16x16>::new(4196, false), FixedTrait::<FP16x16>::new(7465, true), FixedTrait::<FP16x16>::new(2497, true)].span())
}

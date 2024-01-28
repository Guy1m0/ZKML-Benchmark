use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

pub fn tensor() -> Tensor<FP16x16> {
    Tensor::<FP16x16>::new(array![20].span(), array![FixedTrait::<FP16x16>::new(3781, true), FixedTrait::<FP16x16>::new(914, true), FixedTrait::<FP16x16>::new(13493, false), FixedTrait::<FP16x16>::new(1630, true), FixedTrait::<FP16x16>::new(6773, false), FixedTrait::<FP16x16>::new(6132, false), FixedTrait::<FP16x16>::new(2031, false), FixedTrait::<FP16x16>::new(3301, true), FixedTrait::<FP16x16>::new(10645, true), FixedTrait::<FP16x16>::new(9515, false), FixedTrait::<FP16x16>::new(10655, false), FixedTrait::<FP16x16>::new(2472, true), FixedTrait::<FP16x16>::new(9282, true), FixedTrait::<FP16x16>::new(474, false), FixedTrait::<FP16x16>::new(20227, false), FixedTrait::<FP16x16>::new(13947, false), FixedTrait::<FP16x16>::new(18326, true), FixedTrait::<FP16x16>::new(10048, false), FixedTrait::<FP16x16>::new(18846, false), FixedTrait::<FP16x16>::new(1869, true)].span())
}

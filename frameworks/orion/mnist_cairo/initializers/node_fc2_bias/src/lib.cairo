use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc2_bias() -> Tensor<FP16x16> {
    let mut shape = array![10];

    let mut data = array![FP16x16 { mag: 18670, sign: true }, FP16x16 { mag: 13391, sign: false }, FP16x16 { mag: 12405, sign: false }, FP16x16 { mag: 8532, sign: true }, FP16x16 { mag: 11752, sign: false }, FP16x16 { mag: 9879, sign: false }, FP16x16 { mag: 19432, sign: true }, FP16x16 { mag: 10998, sign: false }, FP16x16 { mag: 13790, sign: true }, FP16x16 { mag: 11129, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}
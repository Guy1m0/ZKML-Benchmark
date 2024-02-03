use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc1_bias() -> Tensor<FP16x16> {
    let mut shape = array![25];

    let mut data = array![FP16x16 { mag: 1269, sign: false }, FP16x16 { mag: 6489, sign: true }, FP16x16 { mag: 25337, sign: false }, FP16x16 { mag: 16099, sign: false }, FP16x16 { mag: 11294, sign: true }, FP16x16 { mag: 18863, sign: false }, FP16x16 { mag: 10781, sign: false }, FP16x16 { mag: 15731, sign: false }, FP16x16 { mag: 854, sign: false }, FP16x16 { mag: 11867, sign: true }, FP16x16 { mag: 10570, sign: false }, FP16x16 { mag: 18345, sign: false }, FP16x16 { mag: 16403, sign: false }, FP16x16 { mag: 2638, sign: false }, FP16x16 { mag: 14102, sign: false }, FP16x16 { mag: 10671, sign: false }, FP16x16 { mag: 14429, sign: false }, FP16x16 { mag: 13676, sign: false }, FP16x16 { mag: 4778, sign: false }, FP16x16 { mag: 15753, sign: false }, FP16x16 { mag: 269, sign: false }, FP16x16 { mag: 10043, sign: false }, FP16x16 { mag: 13745, sign: true }, FP16x16 { mag: 6233, sign: false }, FP16x16 { mag: 18134, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}


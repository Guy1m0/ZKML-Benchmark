#[starknet::contract]
mod OrionRunner {
	use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
	use orion::operators::nn::{NNTrait, FP16x16NN};
	use orion::numbers::FP16x16;
	use layer2::weights::tensor2 as w2;
	use layer2::bias::tensor2 as b2;
	use layer3::weights::tensor3 as w3;
	use layer3::bias::tensor3 as b3;
	use layer4::weights::tensor4 as w4;
	use layer4::bias::tensor4 as b4;
	use orion_pt_mnist::input::input;

        
	#[storage]
	struct Storage { 
		id: u8,
 	}

	#[external(v0)]
	fn main(self: @ContractState){
		let _0 = input();
	let _1: Tensor<FP16x16> = NNTrait::gemm(_0, w1(), b1(), Option::Some(1), Option::Some(1), false, true);
	let _2: Tensor<FP16x16> = NNTrait::relu(@_1);
	let _3: Tensor<FP16x16> = NNTrait::gemm(_2, w3(), b3(), Option::Some(1), Option::Some(1), false, true);
	let _4: Tensor<FP16x16> = NNTrait::log_softmax(@_3, 1);
		_4
	}
}
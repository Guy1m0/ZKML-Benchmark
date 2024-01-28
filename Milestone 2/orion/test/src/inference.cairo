#[starknet::contract]
mod OrionRunner {
	use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
	use orion::operators::nn::{NNTrait, FP16x16NN};
	use orion::numbers::FP16x16;
	use layer1::weights::tensor1 as w1;
	use layer1::bias::tensor1 as b1;
	use simple_1::input::input;

        
	#[storage]
	struct Storage { 
		id: u8,
 	}

	#[external(v0)]
	fn main(self: @ContractState) -> Tensor<FP16x16>{
		let _0 = input();
	let _1: Tensor<FP16x16> = NNTrait::gemm(_0, w1(), b1(), Option::Some(1), Option::Some(1), false, true);
	let _2: Tensor<FP16x16> = NNTrait::relu(@_1, 1);
		_2
	}
}
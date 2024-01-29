package main

import (
	"errors"
	"fmt"
	"mlgo/common"
	"mlgo/ml"
)

type mnist_hparams struct{
	n_input int32;
	n_hidden int32;
	n_classes int32;
}

type mnist_model struct {
	hparams mnist_hparams;

	fc1_weight *ml.Tensor;
	fc1_bias *ml.Tensor;

	fc2_weight *ml.Tensor;
	fc2_bias *ml.Tensor;

}

const (
	READ_FROM_BIDENDIAN = true
	OUTPUT_TO_BIDENDIAN = true
)

func MIPS_mnist_model_load(model *mnist_model) error {
	fmt.Println("start MIPS_mnist_model_load")
	model_bytes := common.ReadBytes(common.MODEL_ADDR, READ_FROM_BIDENDIAN)
	index := 0
	fmt.Println("model_bytes len: ", len(model_bytes))

	// verify magic
	{
		magic := common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
		fmt.Printf("magic: %x\n", magic)
		if magic != 0x67676d6c {
			return errors.New("invalid model file (bad magic)")
		}
	}

	// Read FC1 layer 1
	{
		fmt.Println("reading fc1")
		n_dims := int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN))
		fmt.Println("n_dims: ", n_dims)
		ne_weight := make([]int32, 0)
		for i := int32(0); i < n_dims; i++ {
			ne_weight = append(ne_weight, int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)))
		}
		fmt.Println("ne_weight: ", ne_weight)
		// FC1 dimensions taken from file, eg. 768x500
		model.hparams.n_input = ne_weight[0]
		model.hparams.n_hidden = ne_weight[1]
		
		if READ_FROM_BIDENDIAN {
			fc1_weight_data_size := model.hparams.n_input * model.hparams.n_hidden
			fc1_weight_data := common.DecodeFloat32List(model_bytes[index:index + 4 * int(fc1_weight_data_size)])
			index += 4 * int(fc1_weight_data_size)
			model.fc1_weight = ml.NewTensor2DWithData(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden), fc1_weight_data)
		} else {
			model.fc1_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
			fmt.Println("len(model.fc1_weight.Data): ", len(model.fc1_weight.Data))
			for i := 0; i < len(model.fc1_weight.Data); i++{
				model.fc1_weight.Data[i] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
				if i % 10000 == 0 {
					fmt.Println("loading fc1_weight: ", i)
				}
			}
		}

		fmt.Println("index: ", index)

		ne_bias := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_bias = append(ne_bias, int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)))
		}

		if READ_FROM_BIDENDIAN {
			fc1_bias_data_size := int(model.hparams.n_hidden)
			fc1_bias_data := common.DecodeFloat32List(model_bytes[index:index + 4*fc1_bias_data_size])
			index += 4*fc1_bias_data_size
			model.fc1_bias = ml.NewTensor1DWithData(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), fc1_bias_data)
		} else {
			model.fc1_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
			fmt.Println("len(model.fc1_bias.Data): ", len(model.fc1_bias.Data))
			for i := 0; i < len(model.fc1_bias.Data); i++ {
				model.fc1_bias.Data[i] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
				if i % 10000 == 0 {
					fmt.Println("loading fc1_bias: ", i)
				}
			}
		}

	}

	// Read Fc2 layer 2
	{
		fmt.Println("reading fc2")
		n_dims := int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN))
		ne_weight := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_weight = append(ne_weight, int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)))
		}

		// FC1 dimensions taken from file, eg. 10x500
		model.hparams.n_classes = ne_weight[1]

		if READ_FROM_BIDENDIAN {
			fc2_weight_data_size := int(model.hparams.n_hidden * model.hparams.n_classes)
			fc2_weight_data := common.DecodeFloat32List(model_bytes[index:index + 4*fc2_weight_data_size])
			index += 4*fc2_weight_data_size
			model.fc2_weight = ml.NewTensor2DWithData(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes), fc2_weight_data)
		} else {
			model.fc2_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes))
			for i := 0; i < len(model.fc2_weight.Data); i++{
				model.fc2_weight.Data[i] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
			}
		}

		ne_bias := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_bias = append(ne_bias, int32(common.ReadInt32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)))
		}

		if READ_FROM_BIDENDIAN {
			fc2_bias_data_size := int(model.hparams.n_classes)
			fc2_bias_data := common.DecodeFloat32List(model_bytes[index:index + 4*fc2_bias_data_size])
			index += 4*fc2_bias_data_size
			model.fc2_bias = ml.NewTensor1DWithData(nil, ml.TYPE_F32, uint32(model.hparams.n_classes), fc2_bias_data)
		} else {
			model.fc2_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_classes))
			for i := 0; i < len(model.fc2_bias.Data); i++ {
				model.fc2_bias.Data[i] = common.ReadFP32FromBytes(model_bytes, &index, READ_FROM_BIDENDIAN)
			}
		}

		ml.PrintTensor(model.fc2_bias, "model.fc2_bias")
	}

	fmt.Println("current index: ", index)

	return nil
}

// input is 784 bytes
func MIPS_InputProcess() []float32 {
	fmt.Println("start MIPS_InputProcess")
	buf := common.ReadBytes(common.INPUT_ADDR, READ_FROM_BIDENDIAN)
	fmt.Println("buf len: ", len(buf))
	digits := make([]float32, 784)
	
	// render the digit in ASCII
	var c string
	for row := 0; row < 28; row++{
		for col := 0; col < 28; col++ {
			digits[row*28 + col] = float32(buf[row*28 + col])
			if buf[row*28 + col] > 230 {
				c += "*"
			} else {
				c += "_"
			}
		}
		c += "\n"
	}
	fmt.Println(c)

	return digits
}

func MIPS_mnist_eval(model *mnist_model, digit []float32) int {
	fmt.Println("start MIPS_mnist_eval")
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: 1}

	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.n_input))
	copy(input.Data, digit)

	// fc1 MLP = Ax + b
	fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
	fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)
	
	// softmax
	final := ml.SoftMax(ctx0, fc2)

	// run the computation
	ml.BuildForwardExpand(&graph, final)
	ml.GraphCompute(ctx0, &graph)

	ml.PrintTensor(final, "final tensor")

	maxIndex := 0
	for i := 0; i < 10; i++{
		if final.Data[i] > final.Data[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

func MIPS_StoreInMemory(ret int) {
	retBytes := common.IntToBytes(ret, OUTPUT_TO_BIDENDIAN)
	common.Output(retBytes, OUTPUT_TO_BIDENDIAN)
}

func MIPS_MNIST() {
	fmt.Println("Start MIPS MNIST")
	input := MIPS_InputProcess()
	model := new(mnist_model)
	err := MIPS_mnist_model_load(model)
	if err != nil {
		fmt.Println(err)
		common.Halt()
	}
	ret := MIPS_mnist_eval(model, input)
	fmt.Println("Predicted digit is ", ret)
	MIPS_StoreInMemory(ret)
}
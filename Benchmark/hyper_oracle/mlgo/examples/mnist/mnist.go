package mnist

import (
	"errors"
	"fmt"
	"math"
	"mlgo/ml"
	"os"
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

func mnist_model_load(fname string, model *mnist_model) error {

	file, err := os.Open(fname)
	if err != nil {
		return err
	}
	defer file.Close()


	// verify magic
	{
		magic := readInt(file)
		if magic != 0x67676d6c {
			return errors.New("invalid model file (bad magic)")
		}
	}

	// Read FC1 layer 1
	{
		n_dims := int32(readInt(file))
		ne_weight := make([]int32, 0)
		for i := int32(0); i < n_dims; i++ {
			ne_weight = append(ne_weight, int32(readInt(file)))
		}
		// FC1 dimensions taken from file, eg. 768x500
		model.hparams.n_input = ne_weight[0]
		model.hparams.n_hidden = ne_weight[1]

		model.fc1_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_input), uint32(model.hparams.n_hidden))
		for i := 0; i < len(model.fc1_weight.Data); i++{
			model.fc1_weight.Data[i] = readFP32(file)
		}

		ne_bias := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_bias = append(ne_bias, int32(readInt(file)))
		}

		model.fc1_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden))
		for i := 0; i < len(model.fc1_bias.Data); i++ {
			model.fc1_bias.Data[i] = readFP32(file)
		}
	}

	// Read Fc2 layer 2
	{
		n_dims := int32(readInt(file))
		ne_weight := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_weight = append(ne_weight, int32(readInt(file)))
		}

		// FC1 dimensions taken from file, eg. 10x500
		model.hparams.n_classes = ne_weight[1]

		model.fc2_weight = ml.NewTensor2D(nil, ml.TYPE_F32, uint32(model.hparams.n_hidden), uint32(model.hparams.n_classes))
		for i := 0; i < len(model.fc2_weight.Data); i++{
			model.fc2_weight.Data[i] = readFP32(file)
		}

		ne_bias := make([]int32, 0)
		for i := 0; i < int(n_dims); i++ {
			ne_bias = append(ne_bias, int32(readInt(file)))
		}

		model.fc2_bias = ml.NewTensor1D(nil, ml.TYPE_F32, uint32(model.hparams.n_classes))
		for i := 0; i < len(model.fc2_bias.Data); i++ {
			model.fc2_bias.Data[i] = readFP32(file)
		}
		printTensor(model.fc2_bias, "model.fc2_bias")

	}

	return nil
}

func mnist_eval(model *mnist_model, threadCount int, digit []float32) int {

	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: threadCount}

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

	printTensor(final, "final tensor")

	maxIndex := 0
	for i := 0; i < 10; i++{
		if final.Data[i] > final.Data[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

func ExpandGraph(model *mnist_model, threadCount int, digit []float32) (*ml.Graph, *ml.Context) {
	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: threadCount}

	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.n_input))
	copy(input.Data, digit)

	// fc1 MLP = Ax + b
	fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
	fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)
	
	// softmax
	final := ml.SoftMax(ctx0, fc2)

	// run the computation
	ml.BuildForwardExpand(&graph, final)
	return &graph, ctx0
}

func LoadModel(modeFile string) (*mnist_model, error) {
	model := new(mnist_model)
	err := mnist_model_load(modeFile, model)
	return model, err
}

// NB! INT = 32 bits
func readInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

func readString(file *os.File, len uint32) string {
	buf := make([]byte, len)
	if count, err := file.Read(buf); err != nil || count != int(len) {
		return ""
	}
	return string(buf)
}


func readFP32(file *os.File) float32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0.0
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	return math.Float32frombits(bits)
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func printTensor(tensor *ml.Tensor, name string) {
	var dt string
	if tensor.Type == ml.TYPE_F16 {
		dt = "FP16"
	}
	if tensor.Type == ml.TYPE_F32 {
		dt = "FP32"
	}
	if tensor.Type == ml.TYPE_Q4_0 {
		dt = "INT4"
	}

	fmt.Printf("\n\n=== [ %s | %s | %d:%d:%d ] ===\n",
		name, dt, tensor.NE[0], tensor.NE[1], tensor.NE[2])

	for nn := 0; nn < min(12, int(tensor.NE[1])); nn++ {
		fmt.Printf("\n %d x %d ...\t", nn, tensor.NE[0])
		for ii := 0; ii < min(12, int(tensor.NE[0])); ii++ {
			fmt.Printf("%.3f\t", tensor.Data[nn*int(tensor.NE[0])+ii])
		}
	}
}

func main(){
	fmt.Println("hello world")
}
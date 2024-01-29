package mnist

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"mlgo/ml"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestMNIST(t *testing.T) {
	modelFile := "models/mnist/ggml-model-f32.bin"
	digitFile := "models/mnist/t10k-images.idx3-ubyte"

	ml.SINGLE_THREAD = true
	model := new(mnist_model)
	if err := mnist_model_load(modelFile, model); err != nil {
		fmt.Println(err)
		return
	}

	// load a random test digit
	fin, err := os.Open(digitFile)
	if err != nil {
		fmt.Println(err)
		return
	}
	 // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
	rand.Seed(time.Now().UnixNano())
	fin.Seek(int64(16 + 784 * (rand.Int() % 10000)), 0)
	buf := make([]byte, 784)
	digits := make([]float32, 784)
	if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {
		fmt.Println(err, count)
		return
	}
	
	// render the digit in ASCII
	for row := 0; row < 28; row++{
		for col := 0; col < 28; col++ {
			digits[row*28 + col] = float32(buf[row*28 + col])
			var c string
			if buf[row*28 + col] > 230 {
				c = "*"
			} else {
				c = "_"
			}
			fmt.Printf(c)
		}
		fmt.Println("")
	}
	fmt.Println("")

	res := mnist_eval(model, 1, digits)
	fmt.Println("Predicted digit is ", res)
}


func IntToBytes(n int) []byte {
    x := int32(n)

    bytesBuffer := bytes.NewBuffer([]byte{})
    binary.Write(bytesBuffer, binary.BigEndian, x)
    return bytesBuffer.Bytes()
}

func BytesToInt(b []byte) int {
    bytesBuffer := bytes.NewBuffer(b)

    var x int32
    binary.Read(bytesBuffer, binary.BigEndian, &x)

    return int(x)
}

func TestByteInt(t *testing.T){
	a := int(0x67676d6c)
	aBytes := IntToBytes(a)
	aInt := BytesToInt(aBytes)
	aInt2 := (int(aBytes[0]) << 24) | (int(aBytes[1]) << 16) | (int(aBytes[2]) << 8) | int(aBytes[3])
	fmt.Println("a ", a);
	fmt.Println("aBytes ", aBytes)
	fmt.Println("aInt ", aInt)
	fmt.Println("aInt2 ", aInt2)
}

func add(a *int) {
	*a = *a + 1
}

func TestSlice(t *testing.T){
	a := 2
	{
		add(&a)
	}
	fmt.Println(a)
}

func TestSaveInput(t *testing.T) {
	digitFile := "models/mnist/t10k-images.idx3-ubyte"
	// load a random test digit
	fin, err := os.Open(digitFile)
	if err != nil {
		fmt.Println(err)
		return
	}
	fin.Seek(int64(16 + 784 * 0), 0)
	buf := make([]byte, 784)
	digits := make([]float32, 784)
	if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {
		fmt.Println(err, count)
		return
	}
	
	// render the digit in ASCII
	for row := 0; row < 28; row++{
		for col := 0; col < 28; col++ {
			digits[row*28 + col] = float32(buf[row*28 + col])
			var c string
			if buf[row*28 + col] > 230 {
				c = "*"
			} else {
				c = "_"
			}
			fmt.Printf(c)
		}
		fmt.Println("")
	}
	fmt.Println("")

	fout, err := os.Create("models/mnist/input_7")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer fout.Close()
	_, err = fout.Write(buf)
	if err != nil {
		fmt.Println(err)
		return
	}

}

func TestMNISTConvert(t *testing.T) {
	modelFile := "models/mnist/ggml-model-f32.bin"
	digitFile := "models/mnist/t10k-images.idx3-ubyte"

	ml.SINGLE_THREAD = true
	model := new(mnist_model)
	if err := mnist_model_load(modelFile, model); err != nil {
		fmt.Println(err)
		return
	}

	// load a random test digit
	fin, err := os.Open(digitFile)
	if err != nil {
		fmt.Println(err)
		return
	}
		// Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
	rand.Seed(time.Now().UnixNano())
	fin.Seek(int64(16 + 784 * (rand.Int() % 10000)), 0)
	buf := make([]byte, 784)
	digits := make([]float32, 784)
	if count, err := fin.Read(buf); err != nil || count != int(len(buf)) {
		fmt.Println(err, count)
		return
	}

	// render the digit in ASCII
	for row := 0; row < 28; row++{
		for col := 0; col < 28; col++ {
			digits[row*28 + col] = float32(buf[row*28 + col])
			var c string
			if buf[row*28 + col] > 230 {
				c = "*"
			} else {
				c = "_"
			}
			fmt.Printf(c)
		}
		fmt.Println("")
	}
	fmt.Println("")

	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: 1}

	input := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(model.hparams.n_input))
	copy(input.Data, digits)

	// fc1 MLP = Ax + b
	fc1 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc1_weight, input), model.fc1_bias)
	fc2 := ml.Add(ctx0, ml.MulMat(ctx0, model.fc2_weight, ml.Relu(ctx0, fc1)), model.fc2_bias)
	
	// softmax
	final := ml.SoftMax(ctx0, fc2)

	// run the computation
	ml.BuildForwardExpand(&graph, final)
	// stop here
	nodeID := 5
	ml.GraphComputeByNodes(ctx0, &graph, nodeID)

	ml.PrintTensor(graph.Nodes[nodeID], "final_before")

	// continue 
	// ml.ComputeNodeForward(graph.Nodes[5])
	
	// ml.PrintTensor(final, "final_after")

	// test coding and encoding
	envBytes := ml.SaveComputeNodeEnvToBytes(uint32(nodeID), graph.Nodes[nodeID], &graph, true)
	nodeID_, tensorGraphList_ , err := ml.DecodeComputeNodeEnv(envBytes, true, false)

	// save bytes from mips test
	{
		fout, err := os.Create("models/mnist/node_5")
		if err != nil {
			fmt.Println(err)
			return
		}
		defer fout.Close()
		_, err = fout.Write(envBytes)
		if err != nil {
			fmt.Println(err)
			return
		}
	}

	// save => tensorOnGraph[]
	tensorGraphList := ml.SaveComputeNodeEnv(graph.Nodes[5], &graph)

	fmt.Println("nodeID Equal: ", nodeID_ == uint32(nodeID))
	fmt.Println("tensorGraphList: ", reflect.DeepEqual(tensorGraphList_, tensorGraphList))

	// reconstruct
	tensorList := make([]*ml.Tensor, 0)
	tensorMap := make(map[uint32]*ml.Tensor)
	for i := 0; i < len(tensorGraphList); i++ {
		tensor := tensorGraphList[i].ToTensor(nil)
		tensorMap[tensorGraphList[i].NodeID] = tensor
		tensorList = append(tensorList, tensor)
	}
	// fill in the nodeid
	for i := 0; i < len(tensorList); i++ {
		tensor := tensorList[i]
		tensorG := tensorGraphList[i]
		if src0, ok := tensorMap[tensorG.Src0NodeID]; ok {
			tensor.Src0 = src0
		}
		if src1, ok := tensorMap[tensorG.Src1NodeID]; ok {
			tensor.Src1 = src1
		}
	}

	// compute
	ml.ComputeNodeForward(tensorMap[uint32(nodeID)])
	
	ml.PrintTensor(final, "final_after")

	tensor := final
	tensorOnGraph := tensor.ToTensorOnGraph(&graph)
	tensorOnGraphBytes := tensorOnGraph.Encoding(false)
	// bytesLen := common.BytesToInt32(tensorOnGraphBytes[:4], false)
	// fmt.Println(int(bytesLen) == len(tensorOnGraphBytes) - 4)
	tensorOnGraph2 := ml.DecodeTensorOnGraph(tensorOnGraphBytes, false, false)
	fmt.Println(reflect.DeepEqual(tensor.Data, tensorOnGraph.Data))
	fmt.Println(reflect.DeepEqual(tensorOnGraph, tensorOnGraph2))
	fmt.Println(tensorOnGraph.Src0NodeID, tensorOnGraph.Src1NodeID)
}
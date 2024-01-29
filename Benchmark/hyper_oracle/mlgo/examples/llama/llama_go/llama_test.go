package llama

import (
	"fmt"
	"mlgo/ml"
	"os"
	"reflect"
	"testing"
)

func TestLLaMA(t *testing.T) {
	modelFile := "../models/llama-7b-fp32.bin"
	prompt := "Why Golang is so popular?"
	threadCount := 32
	ctx, err := LoadModel(modelFile, true)
	fmt.Println("Load Model Finish")
	if err != nil {
		fmt.Println("load model error: ", err)
		return 
	}
	embd := ml.Tokenize(ctx.Vocab, prompt, true)
	err = Eval(ctx, embd, uint32(len(embd)), 0, threadCount)
	fmt.Println("Eval Model Finish")
}

func TestLLaMAEvalGraph(t *testing.T) {
	modelFile := "../models/llama-7b-fp32.bin"
	prompt := "Why Golang is so popular?"
	threadCount := 32
	ctx, err := LoadModel(modelFile, true)
	fmt.Println("Load Model Finish")
	if err != nil {
		fmt.Println("load model error: ", err)
		return 
	}
	embd := ml.Tokenize(ctx.Vocab, prompt, true)
	graph, mlctx, err := ExpandGraph(ctx, embd, uint32(len(embd)), 0, threadCount)
	nodeID := int(graph.NodesCount) - 1
	ml.GraphComputeByNodes(mlctx, graph, nodeID)
	ml.PrintTensor(graph.Nodes[nodeID], "before")

	envBytes := ml.SaveComputeNodeEnvToBytes(uint32(nodeID), graph.Nodes[nodeID], graph, true)
	nodeID_, tensorGraphList_ , err := ml.DecodeComputeNodeEnv(envBytes, true, false)
	// save bytes from mips
	{
		fout, err := os.Create(fmt.Sprintf("../data/node_%v", nodeID))
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
	tensorGraphList := ml.SaveComputeNodeEnv(graph.Nodes[nodeID], graph)
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

	// ml.ComputeNodeForward(graph.Nodes[nodeID])
	ml.PrintTensor(tensorMap[uint32(nodeID)], "after")

	fmt.Println("graph node number: ", graph.NodesCount)
	fmt.Println("Eval Model Finish")
}
package main

import (
	"fmt"
	"mlgo/common"
	"mlgo/ml"
)

const (
	READ_FROM_BIDENDIAN = true
	OUTPUT_TO_BIDENDIAN = true
)

// read from memory [size: [envData]]
// output: nodeID, tensorOnGraph, error
func ReadTensorGraph() (uint32, []*ml.TensorOnGraph, error){
	fmt.Println("Start Read Tensor Graph")
	dataBytes := common.ReadBytes(common.INPUT_ADDR, READ_FROM_BIDENDIAN)
	nodeID, tensorGraphList, err := ml.DecodeComputeNodeEnv(dataBytes, READ_FROM_BIDENDIAN, true)
	return nodeID, tensorGraphList, err
}

func ComputeTensorGraph(nodeID uint32, tensorGraphList []*ml.TensorOnGraph) {
	fmt.Println("State Compute Tensor Graph")
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
	ml.ComputeNodeForward(tensorMap[uint32(nodeID)])
	ml.PrintTensor(tensorMap[uint32(nodeID)], "final_after")
}

func main() {
	nodeID, tensorGraphList, err := ReadTensorGraph()
	if err != nil {
		return 
	}
	ComputeTensorGraph(nodeID, tensorGraphList)
}
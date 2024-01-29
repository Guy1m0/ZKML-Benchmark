package ml

import (
	"errors"
	"fmt"
	"math"
	"mlgo/common"
	"os"
)

// Tensor on Graph for stroage
type TensorOnGraph struct {
	Type DType

	NodeID uint32 // nodeID == 99999 no exist (> graph.NodeCount)

	Dims uint32
	NE   [MAX_DIMS]uint32 // number of elements
	NB   [MAX_DIMS]uint32 // stride in bytes

	Op optype

	// isParam bool // no need here?

	// GradTensorID uint32 // no need for forward compute?
	Src0NodeID uint32
	Src1NodeID uint32

	// grad *Tensor
	// src0 *Tensor
	// src1 *Tensor
	// opt  [MAX_OPT]*Tensor // FIXME Do we need this?

	TasksCount int

	// performance
	//perfRuns   uint32
	//perfCycles uint32
	//perfTime   uint64

	Data []float32
	//padding [8]byte
}

func (tensor * Tensor) ToTensorOnGraph(graph *Graph) *TensorOnGraph {
	if tensor == nil || graph == nil || graph.Tensor2NodeID == nil {
		return nil 
	}
	t := &TensorOnGraph{
		Type: tensor.Type,
		Dims: tensor.Dims,
		NE: tensor.NE,
		NB: tensor.NB,
		Op: tensor.op,
		TasksCount: tensor.TasksCount,
		Data: tensor.Data,
	}
	t.NodeID = tensor2NodeID(tensor, graph)
	t.Src0NodeID = tensor2NodeID(tensor.Src0, graph)
	t.Src1NodeID = tensor2NodeID(tensor.Src1, graph)
	return t
}

func (tensor *TensorOnGraph) ToTensor(tensorMap map[uint32]*Tensor) *Tensor {
	t := &Tensor{
		Type: tensor.Type,
		Dims: tensor.Dims,
		NE: tensor.NE,
		NB: tensor.NB,
		op: tensor.Op,
		TasksCount: tensor.TasksCount,
		Data: tensor.Data,
	}
	if tensorMap != nil {
		t.Src0 = tensorMap[tensor.Src0NodeID]
		t.Src1 = tensorMap[tensor.Src1NodeID]
	}
	return t
}

func tensor2NodeID(tensor *Tensor, graph *Graph) uint32 {
	if id, ok := graph.Tensor2NodeID[tensor]; ok {
		return id
	} else {
		return math.MaxUint32
	}
}

func (tensor *TensorOnGraph) Encoding(toBigEndian bool) []byte {
	data := make([]byte, 0)
	data = append(data, common.IntToBytes(int(tensor.Type), toBigEndian)...) // Type
	data = append(data, common.IntToBytes(int(tensor.NodeID), toBigEndian)...) // NodeID
	data = append(data, common.IntToBytes(int(tensor.Dims), toBigEndian)...) // Dims
	data = append(data, common.IntToBytes(int(tensor.Op), toBigEndian)...) // Op
	data = append(data, common.IntToBytes(int(tensor.Src0NodeID), toBigEndian)...) // Src0NodeID
	data = append(data, common.IntToBytes(int(tensor.Src1NodeID), toBigEndian)...) // Src1NodeID
	data = append(data, common.IntToBytes(int(tensor.TasksCount), toBigEndian)...) // TasksCount

	// encoding list
	// NE
	data = append(data, common.IntToBytes(MAX_DIMS, toBigEndian)...)
	for i := 0; i < MAX_DIMS; i++ {
		data = append(data, common.IntToBytes(int(tensor.NE[i]), toBigEndian)...)
	}
	// NB
	data = append(data, common.IntToBytes(MAX_DIMS, toBigEndian)...)
	for i := 0; i < MAX_DIMS; i++ {
		data = append(data, common.IntToBytes(int(tensor.NB[i]), toBigEndian)...)
	}
	// Data
	data = append(data, common.IntToBytes(len(tensor.Data), toBigEndian)...)
	for i := 0; i < len(tensor.Data); i++ {
		data = append(data, common.Float32ToBytes(tensor.Data[i], toBigEndian)...)
	}
	// append the data size ahead
	// data = append(common.IntToBytes(len(data), toBigEndian), data...)
	return data
}

func DecodeTensorOnGraph(data []byte, fromBigEndian bool, currentBigEndian bool) *TensorOnGraph {
	if (len(data) == 0) {
		return nil
	}
	t := 0
	tensorType := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	nodeId := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	dims := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	op := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	src0NodeID := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	src1NodeID := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	tasksCount := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	//NE
	neSize := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4
	ne := [4]uint32{0, 0, 0, 0}
	for i := 0; i < int(neSize); i++ {
		ne[i] = uint32(common.BytesToInt32(data[t:t+4], fromBigEndian))
		t += 4
	}

	// NB
	nbSize := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4
	nb := [4]uint32{0, 0, 0, 0}
	for i := 0; i < int(nbSize); i++ {
		nb[i] = uint32(common.BytesToInt32(data[t:t+4], fromBigEndian))
		t += 4
	}

	// Data
	dataSize := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4
	tensorData := make([]float32, 0)
	if currentBigEndian && fromBigEndian {
		// this code should be only used in MIPS!
		tensorData = common.DecodeFloat32List(data[t:t+4*int(dataSize)])
		t += 4*int(dataSize)
	} else {
		tensorData = make([]float32, dataSize)
		for i := 0; i < int(dataSize); i++ {
			tensorData[i] = common.BytesToFloat32(data[t:t+4], fromBigEndian)
			t += 4
		}
	}


	tensor := &TensorOnGraph{
		Type: DType(tensorType),
		NodeID: uint32(nodeId),
		Dims: uint32(dims),
		Op: optype(op),
		Src0NodeID: uint32(src0NodeID),
		Src1NodeID: uint32(src1NodeID),
		TasksCount: int(tasksCount),
		NE: ne,
		NB: nb,
		Data: tensorData,
	}

	return tensor
}

func ComputeNodeForward(node *Tensor) {
	if node == nil {
		return 
	}
	node.TasksCount = 1
	params := ComputeParams{
		Type: TASK_COMPUTE,
		ith:  0,
		nth:  uint32(node.TasksCount),
	}
	ComputeForward(nil, &params, node)
}

// =======================================================================

// compute [0, nodeID)
func GraphComputeByNodes(ctx *Context, graph *Graph, nodeID int) {

	maxThreads := graph.ThreadsCount

	// --- init N job goroutines and channel to send tasks for them

	graph.Jobs = make(chan *ComputeParams, maxThreads) // TODO Right place to init?
	defer close(graph.Jobs)

	// TODO Investigate https://pkg.go.dev/runtime#LockOSThread
	for i := 0; i < maxThreads; i++ {
		go Job(graph.Jobs)
	}

	// --- initialize tasks

	{
		// thread scheduling for the different operations
		// TasksCount might be 0, 1, or ThreadsCount
		for i := uint32(0); i < graph.NodesCount; i++ {

			////struct ggml_tensor * node = cgraph->nodes[i];
			node := graph.Nodes[i]

			if DEBUG {
				fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
			}

			switch node.op {

			case OP_DUP:
				node.TasksCount = 1
			case OP_ADD:
				node.TasksCount = 1 // TODO threads
			case OP_SUB:
			case OP_MUL:
			case OP_DIV:
			case OP_SQR:
			case OP_SQRT:
			case OP_SUM:
			case OP_MEAN:
			case OP_REPEAT:
			case OP_ABS:
			case OP_SGN:
			case OP_NEG:
			case OP_STEP:
			case OP_RELU:
				node.TasksCount = 1
			case OP_GELU:
				node.TasksCount = 1 // TODO threads
			case OP_SILU:
				node.TasksCount = 1 // TODO threads
			case OP_NORM:
			case OP_RMS_NORM:
				node.TasksCount = 1 // TODO threads
			case OP_MUL_MAT:
				node.TasksCount = maxThreads
				// TODO: use different scheduling for different matrix sizes
			case OP_SCALE:
				node.TasksCount = 1 // TODO threads
			case OP_CPY:
			case OP_RESHAPE:
			case OP_VIEW:
			case OP_PERMUTE:
			case OP_TRANSPOSE:
			case OP_GET_ROWS:
			case OP_DIAG_MASK_INF:
				node.TasksCount = 1
			case OP_SOFT_MAX:
				node.TasksCount = 1 // TODO threads
			case OP_ROPE:
				////node.TasksCount = 1
			case OP_CONV_1D_1S:
			case OP_CONV_1D_2S:
				node.TasksCount = 1 // TODO threads
				////ASSERT(node->src0->ne[3] == 1);
				////ASSERT(node->src1->ne[2] == 1);
				////ASSERT(node->src1->ne[3] == 1);
			case OP_FLASH_ATTN:
				node.TasksCount = 1 // TODO threads
			case OP_FLASH_FF:
				node.TasksCount = 1 // TODO threads
			case OP_NONE:
				node.TasksCount = 1
			case OP_COUNT:
				fmt.Printf("\n[HALT] Something wrong with compute graph!")
				os.Exit(1)
			}
		}
	}

	nodeID = min(nodeID, int(graph.NodesCount))

	for i := uint32(0); i < uint32(nodeID); i++ {

		node := graph.Nodes[i]

		if DEBUG {
			fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
		}

		params := ComputeParams{
			Type: TASK_INIT,
			ith:  0,
			nth:  uint32(node.TasksCount),
		}

		ComputeForward(graph, &params, node) // TASK_INIT

		// --- COMPUTE

		// BREAKPOINT DEBUG
		//if i > 1300 {
		//	fmt.Printf("\n\n=== HALT #%d ===", i)
		//	os.Exit(0)
		//}

		params.Type = TASK_COMPUTE
		ComputeForward(graph, &params, node)

		// --- FINALIZE

		params.Type = TASK_FINALIZE
		ComputeForward(graph, &params, node)
	}

}

func SaveComputeNodeEnv(node *Tensor, graph *Graph) []*TensorOnGraph{
	tensorOnGraphList := make([]*TensorOnGraph, 0)
	tensorOnGraphList = append(tensorOnGraphList, node.ToTensorOnGraph(graph))
	if node.Src0 != nil {
		tensorOnGraphList = append(tensorOnGraphList, node.Src0.ToTensorOnGraph(graph))
	}
	if node.Src1 != nil {
		tensorOnGraphList = append(tensorOnGraphList, node.Src1.ToTensorOnGraph(graph))
	}
	return tensorOnGraphList
}

// total_bytes_len
// nodeID
// tensorGraph num
// [len, tensor]
func SaveComputeNodeEnvToBytes(nodeID uint32, node *Tensor, graph *Graph, toBigEndian bool) []byte {
	tensorGraphList := SaveComputeNodeEnv(node, graph)
	if len(tensorGraphList) == 0 {
		return nil 
	}
	data := make([]byte, 0)
	// nodeID
	data = append(data, common.IntToBytes(int(nodeID), toBigEndian)...)
	// tensorGraph num
	data = append(data, common.IntToBytes(len(tensorGraphList), toBigEndian)...)
	// tensor
	for i := 0; i < len(tensorGraphList); i++ {
		tensor := tensorGraphList[i]
		bytes := tensor.Encoding(toBigEndian)
		// append size ahead of content
		bytes = append(common.IntToBytes(len(bytes), toBigEndian), bytes...)
		// append into data
		data = append(data, bytes...)
	}
	// total bytes len
	data = append(common.IntToBytes(len(data), toBigEndian), data...)
	return data
}

func DecodeComputeNodeEnv(data []byte, fromBigEndian bool, currentBigEndian bool) (uint32, []*TensorOnGraph, error) {
	t := 0
	totalSize := common.BytesToInt32(data[:4], fromBigEndian)
	t += 4
	if int(totalSize) < len(data) - 4 {
		return 0, nil, errors.New("no enough data")
	}

	// nodeID
	nodeID := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	// tensorNum
	tensorNum := common.BytesToInt32(data[t:t+4], fromBigEndian)
	t += 4

	tensorOnGraphList := make([]*TensorOnGraph, tensorNum)

	for i := 0; i < int(tensorNum); i++ {
		// size
		size := common.BytesToInt32(data[t:t+4], fromBigEndian)
		t += 4
		// tensorOnGraph
		tensor := DecodeTensorOnGraph(data[t:t+int(size)], fromBigEndian, currentBigEndian)
		t += int(size)
		
		tensorOnGraphList[i] = tensor
	}

	return uint32(nodeID), tensorOnGraphList, nil
}
package gpt2

import (
	"errors"
	"fmt"
	"math"
	"mlgo/ml"
	"os"
	"strconv"
)

// default hparams (GPT-2 117M)
/*
   int32_t n_vocab = 50257;
   int32_t n_ctx   = 1024;
   int32_t n_embd  = 768;
   int32_t n_head  = 12;
   int32_t n_layer = 12;
   int32_t ftype   = 1;
*/
type gpt2_hparams struct  {
	n_vocab int32;
	n_ctx int32;
	n_embd int32;
	n_head int32;
	n_layer int32;
	ftype int32;

};

type gpt2_layer struct {
    // normalization
	ln_1_g *ml.Tensor;
	ln_1_b *ml.Tensor;

	ln_2_g *ml.Tensor;
	ln_2_b *ml.Tensor;

	c_attn_attn_w *ml.Tensor;
	c_attn_attn_b *ml.Tensor;

	c_attn_proj_w *ml.Tensor;
	c_attn_proj_b *ml.Tensor;

	c_mlp_fc_w *ml.Tensor;
	c_mlp_fc_b *ml.Tensor;

	c_mlp_proj_w *ml.Tensor;
	c_mlp_proj_b *ml.Tensor;
}

type gpt2_model struct {
    hparams gpt2_hparams ;

	ln_f_g *ml.Tensor;
	ln_f_b *ml.Tensor;

	wte *ml.Tensor;
	wpe *ml.Tensor;
	lm_head *ml.Tensor;

	layers []gpt2_layer;

	memory_k *ml.Tensor;
	memory_v *ml.Tensor;

	tensors map[string]*ml.Tensor;
}

func gpt2_model_load(fname string, model *gpt2_model, vocab *gpt_vocab) error {

	file, err := os.Open(fname)
	if err != nil {
		return err
	}
	defer file.Close()

	{
		magic := readInt(file)
		if magic != 0x67676d6c {
			return errors.New("invalid model file (bad magic)")
		}
	}

	// load hparams
	{
		model.hparams.n_vocab = int32(readInt(file))
		model.hparams.n_ctx = int32(readInt(file))
		model.hparams.n_embd = int32(readInt(file))
		model.hparams.n_head = int32(readInt(file))
		model.hparams.n_layer = int32(readInt(file))
		model.hparams.ftype = int32(readInt(file))

		fmt.Printf("hparams: %v\n", model.hparams)
	}

	// load vocab
	{
		n_vocab := readInt(file)
		if n_vocab != uint32(model.hparams.n_vocab) {
			return errors.New(fmt.Sprintf("n_vocan: %v, model.hparams.n_vocan: %v", n_vocab, model.hparams.n_vocab))
		}

		for i := uint32(0); i < (n_vocab); i++ {
			len := readInt(file)
			word := readString(file, len)
			vocab.token_to_id[word] = i
			vocab.id_to_token[i] = word
		}
	}

	wtype := ml.TYPE_F32
	dtype := ml.TYPE_F32

	// weights 
	{
		n_embd := uint32(model.hparams.n_embd)
		n_layer := uint32(model.hparams.n_layer)
		n_ctx := uint32(model.hparams.n_ctx)
		n_vocab := uint32(model.hparams.n_vocab)

		model.layers = make([]gpt2_layer, n_layer)
		model.tensors = make(map[string]*ml.Tensor)

		model.ln_f_g = ml.NewTensor1D(nil, dtype, uint32(n_embd))
		model.ln_f_b = ml.NewTensor1D(nil, dtype, uint32(n_embd))

		model.wte = ml.NewTensor2D(nil, wtype, uint32(n_embd), uint32(n_vocab))
		model.wpe = ml.NewTensor2D(nil, dtype, uint32(n_embd), uint32(n_ctx))
		model.lm_head = ml.NewTensor2D(nil, wtype, uint32(n_embd), uint32(n_vocab))

		// map by name
		model.tensors["model/ln_f/g"] = model.ln_f_g;
		model.tensors["model/ln_f/b"] = model.ln_f_b;

		model.tensors["model/wte"]     = model.wte;
		model.tensors["model/wpe"]     = model.wpe;
		model.tensors["model/lm_head"] = model.lm_head;

		for i := 0; i < int(n_layer); i++ {
            layer := &model.layers[i];

            layer.ln_1_g        = ml.NewTensor1D(nil, dtype,   n_embd);
            layer.ln_1_b        = ml.NewTensor1D(nil, dtype,   n_embd);

            layer.ln_2_g        = ml.NewTensor1D(nil, dtype,   n_embd);
            layer.ln_2_b        = ml.NewTensor1D(nil, dtype,   n_embd);

            layer.c_attn_attn_w = ml.NewTensor2D(nil, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b = ml.NewTensor1D(nil, dtype, 3*n_embd);

            layer.c_attn_proj_w = ml.NewTensor2D(nil, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ml.NewTensor1D(nil, dtype,   n_embd);

            layer.c_mlp_fc_w    = ml.NewTensor2D(nil, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b    = ml.NewTensor1D(nil, dtype, 4*n_embd);

            layer.c_mlp_proj_w  = ml.NewTensor2D(nil, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b  = ml.NewTensor1D(nil, dtype,   n_embd);

            // map by name
            model.tensors["model/h" + strconv.Itoa(i) + "/ln_1/g"]        = layer.ln_1_g;
            model.tensors["model/h" + strconv.Itoa(i) + "/ln_1/b"]        = layer.ln_1_b;

            model.tensors["model/h" + strconv.Itoa(i) + "/ln_2/g"]        = layer.ln_2_g;
            model.tensors["model/h" + strconv.Itoa(i) + "/ln_2/b"]        = layer.ln_2_b;

            model.tensors["model/h" + strconv.Itoa(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + strconv.Itoa(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + strconv.Itoa(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + strconv.Itoa(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + strconv.Itoa(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            model.tensors["model/h" + strconv.Itoa(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            model.tensors["model/h" + strconv.Itoa(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w;
            model.tensors["model/h" + strconv.Itoa(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
        }
	}

	// key + value
	{
		n_mem := model.hparams.n_layer * model.hparams.n_ctx
		n_element := model.hparams.n_embd * n_mem

		model.memory_k = ml.NewTensor1D(nil, dtype, uint32(n_element))
		model.memory_v = ml.NewTensor1D(nil, dtype, uint32(n_element))

		fmt.Println("n_element in key+value: ", n_element)
	}

	// load weights 
	{
		total_size := 0
		has_lm_head := false

		for {
			n_dim := readInt(file)
			length := readInt(file)
			ttype := readInt(file)

			if n_dim | length | ttype  == 0 {
				// eof
				break
			}

			nelements := 1
			ne := make([]int32, 2)
			for i := 0; i < int(n_dim); i++ {
				ne[i] = int32(readInt(file))
				nelements *= int(ne[i])
			}

			// read name len
			name := readString(file, length)
			if _, ok := model.tensors[name]; !ok {
				return errors.New(fmt.Sprintf("unknow tensor: %s", name))
			}
			tensor := model.tensors[name]

			// read data
			for i := 0; i < len(tensor.Data); i++{
				tensor.Data[i] = readFP32(file)
			}

			// GPT-2 models share the WTE tensor as the LM head
			if name == "model/wte" && !has_lm_head {
				copy(tensor.Data, model.lm_head.Data)
			}

			if name == "model/lm_head" {
				has_lm_head = true
			}

			total_size += len(tensor.Data) * 4

		}
	}

	return nil 
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// func gpt2_eval(model *gpt2_model, n_thread int, n_past int, embd_inp []uint32, embd_w []float32, mem_per_token uint32) {
// 	N := len(embd_inp)

// 	n_embd := model.hparams.n_embd
// 	n_layer := model.hparams.n_layer
// 	n_ctx := model.hparams.n_ctx
// 	n_head := model.hparams.n_head
// 	n_vocab := model.hparams.n_vocab

// 	gf := ml.Graph{ThreadsCount: n_thread}
// 	embd := ml.NewTensor1D(nil, ml.TYPE_F32, uint32(N))
// 	for i := 0; i < N; i++ {
// 		embd.Data[i] = float32(embd_inp[i])
// 	}

// 	position := ml.NewTensor1D(nil, ml.TYPE_F32, uint32(N))
// 	for i := 0; i < N; i++ {
// 		position.Data[i] = float32(n_past + 1)
// 	}

// 	inpL := ml.Add(nil, ml.GetRows(nil, model.wte, embd), ml.GetRows(nil, model.wpe, position))

// 	for il := 0; il < int(n_layer); il++ {
// 		// TODO: replace with ggml_norm
// 		cur := ml.RMSNorm(nil, inpL)
// 		cur = ml.Add(nil, ml.Mul(nil, ml.Repeat(nil, model.layers[il].ln_1_g, cur), cur), ml.Repeat(nil, model.layers[il].ln_1_b, cur))

// 		cur = ml.MulMat(nil, model.layers[il].c_attn_attn_w, cur)
// 		cur = ml.Add(nil, ml.Repeat(nil, model.layers[il].c_attn_attn_b, cur), cur)

// 		// self-attention
// 		{
// 			Qcur := ml.View1D()
// 		}
// 	}

// }

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

type gpt_vocab struct {
	token_to_id map[string]uint32
	id_to_token map[uint32]string
}

func NewVocab() *gpt_vocab {
	return &gpt_vocab{
		token_to_id: make(map[string]uint32),
		id_to_token: make(map[uint32]string),
	}
}
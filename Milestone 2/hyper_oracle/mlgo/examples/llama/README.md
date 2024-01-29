# LLaMA.go

Part of this code is borrowed from [llama.go](https://github.com/gotzmann/llama.go)


## How to Run

First, obtain and convert original LLaMA models on your own, or just download ready-to-rock ones (for LLaMA 1):

LLaMA-7B: [llama-7b-fp32.bin](https://nogpu.com/llama-7b-fp32.bin)

LLaMA-13B: [llama-13b-fp32.bin](https://nogpu.com/llama-13b-fp32.bin)

For LLaMA-2, please convert original LLaMA models on your own
```shell
python3 ./scripts/convert-pth-to-ggml.py /LLaMA-Path/ 0
```

This model stores FP32 weights, so you'll needs at least 32Gb of RAM (not VRAM or GPU RAM) for LLaMA-7B. 
Double to 64Gb for LLaMA-13B.

Please make sure that the LLaMA model is saved in path `mlgo/examples/llama/models/llama-7b-fp32.bin`

```shell
go build
./llama --threads 8 --model ./models/llama-7b-fp32.bin --temp 0.80 --context 128 --predict 128 --prompt "How to combine AI and blockchain?"
```


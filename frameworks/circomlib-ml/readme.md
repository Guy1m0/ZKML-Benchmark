# How to Benchmark
The demo video has been uploaded to youtube: https://www.youtube.com/watch?v=spH1DYuvEmk

Please first run the setup bash code
```bash
bash ./setup-circom.sh
```

Then, check all the supported models by running:
```
python benchmark.py --list
```
After getting a list of supported models, select the one you want to test and run following command:
```
python benchmark.py --model <model name> --size <test size> --save <optional flag to save the result locally or not>
```


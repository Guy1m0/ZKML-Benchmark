# How to Benchmark
The demo video has been uploaded to youtube: https://www.youtube.com/watch?v=spH1DYuvEmk

First of all, check all the supported models by running:
```
python benchmark.py --list
```
After getting a list of supported models, select the one you want to test and run following command:
```
python benchmark.py --model <model name> --size <test size> --save <optional flag to save the result locally or not>
```

if you get following error:
```
OSError: [Errno 8] Exec format error: './bin/time_circuit'
```
It means that your system not supports default binary file. Therefore, you can either use the provided pre-compiled bin for m1_mac and intel_mac or compile it by yourself. 

For the m1_mac users, you can run following command to do the benchmark:
```
python benchmark.py --model <model name> --size <test size> --save <optional flag to save the result locally or not> --arm
```

For the intel_mac users, you may need to manually move all these files from './bin/intel_mac/' to './bin/' by replacing the default bin files.

If neither of above works on your system, you need to run the bash command:
```
bash ./setup-zkml.sh
```
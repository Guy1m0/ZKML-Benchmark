ZKML Benchmarking
===
> **Disclaimer**: The benchmark settings for each framework have been determined solely based on my interpretation of their respective documentation and codebases. It is possible that misinterpretations have occurred, potentially leading to suboptimal environment configurations, improper testing data preprocessing, or incorrect parameter selection. Consequently, these factors may have influenced the accuracy of the benchmark results. If you detect any such errors or unexpected results, please do not hesitate to contact me via my Telegram account @Guy1m0. I am eager to address any inaccuracies to ensure the benchmarks presented are as reliable and comprehensive as possible. 

# Introduction

As machine learning continues its expansion in the global market, the reliability and integrity of its models have become paramount. This is especially true in the context of Machine Learning as a Service (MLaaS), where there's an inherent need to ensure **model authenticity**, which means guaranteeing that the offered model not only matches its description but also operates within accurate parameters while maintaining a degree of privacy. To achieve this, zk-SNARK (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) has garnered significant attention. Its ability to produce short proofs, regardless of the size of the input data, makes it a prime candidate for integration with ML frameworks like EZKL and Daniel Kang's zkml. However, the challenge of translating ML models into circuits optimized for zero-knowledge-proof systems is non-trivial, particularly for complex neural networks.

Consequently, progress has been made in crafting custom circuits using existing architectures, like zk-STARK, Halo2, and Plonk2. Although these custom circuits can accommodate the sophisticated operations of modern ML models, they often fall short of being scalable, generalized solutions. This situation presents developers with a dilemma: selecting the framework that best suits their specific needs.

To address this issue, I'm developing a zkML benchmark. This tool is designed to assist developers in understanding the trade-offs and performance differences among various frameworks. While many frameworks offer their own benchmark sets, making direct comparisons is complex due to the numerous variables that affect performance. My approach focuses on establishing uniform conditions across all frameworks to provide a practical and straightforward comparison.


# How to Benchmark
The demo video has been uploaded to youtube: https://www.youtube.com/watch?v=spH1DYuvEmk

In general, please first check all the supported model by running for each frameworks:
```
python benchmark.py --list
```

and then run the benchmark given the output from previous command:
```
python benchmark.py --model <model name> --size <test size>
```

It is worth noting that the argments needed for each frameworks may vary. Therefore, user may chack all the commands as follows:

```
python benchmark.py -h
```

> For frameworks proposed by hyper_oracle and so_cathie, please first run the setup bash code

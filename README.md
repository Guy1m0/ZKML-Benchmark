ZKML Benchmarking
===
> Disclaimer:

# Introduction


As machine learning continues to expand in the global market, the reliability and integrity of its models have become paramount. Considering the case of Machine Learning as a Service (MLaaS), there's an inherent need to guarantee that the offered model truly matches its description (also noted as **model authenticity**), while operating within accurate parameters and maintaining a degree of privacy. Zero-knowledge proof, more specifically zero-knowledge Machine Learning (zkML), steps in to bridge this trust gap, vouching for both the integrity of computations and the confidentiality of underlying weights and structures.

To achieve this, zk-SNARK (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) has garnered significant attention. Its ability to produce short proofs, regardless of the size of the input data, makes it a prime candidate for integration with ML frameworks like EZKL and Daniel Kang's zkml. However, translating ML models into circuits optimized for these zero-knowledge-proof systems remains a challenge, even when the models aren't intricate neural networks. 

Consequently, progress has been made in crafting custom circuits using existing architectures, like zk-STARK, Halo2, and Plonk2. While these custom circuits can support the complex operations intrinsic to modern ML models, they often stray from being scalable generalized solutions. Naturally, this landscape presents developers with a quandary: which framework aligns best with their specific requirements?

Addressing this challenge, I'm developing a zkML benchmark. This tool aids developers in discerning the trade-offs and performance nuances among various frameworks. While manually setting up and conducting benchmarks for the increasing number of frameworks can be daunting, and though these frameworks often come with their benchmark sets, direct comparisons become intricate due to the myriad of performance-impacting variables. My approach has been to establish uniform conditions across all frameworks, aiming to offer a pragmatic and straightforward comparison among them.

It's worth noting that machine learning models generally undergo two phases: the **training phase** and the **inference phase**. Due to the computational complexity associated with verifying training in-circuit, this benchmark primarily focuses on frameworks designed for the inference stage of ML models. 



# How to Benchmark

For each frameworks please run the bash code
```
python benchmark.py --model <model name> --size <test size>
```

The argments needed for each frameworks may vary. 

> For socathie, need to do the setupt-circom.sh and then trusted setup first
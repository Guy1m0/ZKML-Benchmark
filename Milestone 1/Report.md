# ZKML Benchmarking Project Report

## Introduction

As machine learning gains attention globally, ensuring the integrity and authenticity of machine learning models, particularly in Machine Learning as a Service (MLaaS), has become crucial. Zero-knowledge Machine Learning (zkML), and specifically zk-SNARK, emerges as a key player in bridging the trust gap, ensuring both computational integrity and data privacy. This project aims to develop a zkML benchmarking tool to guide developers through the complexities of the zkML landscape, focusing on the inference phase of ML models due to the high computational demands of the training phase.

## Current Landscape
Even various of proving systems have been introduced in recent years, and each of them is proved to be practical in certain application scenarieno and advanced in no trusted-setup, proving efficiency, deterministic proof size etc. However, naively converting a ML model to an arithmetic circuit is not feasible all the time, since the computation might be quite taxing for millions of paramters used in NN model. Therefore, many research works and industrial frameworks have been proposed to fix this issue. The landscape can be categorized into several key areas:

- Model-to-Proof Compilers: Convert conventional ML model from a common format (e.g. ONNX, Keras, etc) into a verifiable computational circuit
  - **EZKL:** A library and command-line tool that converts ONNX models into zk-SNARK circuits.
  - **Orion:** Generates Validity ML that enables the verification of the inference by leveraging Cario and ONNX's capabilities.
  - **keras2circom&Circomlib-ml:** A python tool that transpiles Keras models into Circom circuits.
  - **LinearA:** A framework bridges Apache TVM to arithmetic circuits in ZKP systems.
  - **Daniel Kang's zkml:** Constructs proofs of ML model execution in Halo2.

- zkML-Specific Proving Systems: Generate efficient verification of ML models with circuit-unfriendly operations
  - **zkCNN:** A novel approach for verifying convolutional neural networks.
  - **Zator:** Utilizes recursive zk-SNARKs for deep neural network verification.
  - **ZeroGravity:** A system for proving an inference run for a pre-trained, public WNN and a private input.


- Hardware Acceleration: Build specialized hardware to support proof generation
  - **Supranational:** Offers GPU acceleration solutions.
  - **Accseal:** Develops ASIC chips for ZKP hardware acceleration.
  - **Icicle:** A CUDA implementation of general functions used in zero-knowledge proof.

- Application: Design for zkML use cases
  <!-- - **ZKP Neural Networks:** *evalutation of neural networks inside zero knowledge proofs* -->
  - **Worldcoin:** Integrates zkML for a privacy-preserving proof of personhood protocol.
  - **ZKaptcha:** Enhances captcha services by analyzing user behaviors through zkML.
  - **ZKaggle:** Bounty platform for hosting, verifying, and paying out bounties
  - **RockyBot:** On-chain verifiable ML trading bot.

### Proving System

Generalized proving system, like Plonkish, STARK, R1CS and GKR, is the backend of abovementioned zkML frameworks and also the main enabler in bringing zkML to realization. However, the advancement of verifiable ML is hindered by non-arithmetic operations, notably activitaion functions such as ReLU. For this reason, plonkish-based proving system, like Halo2 and Plonky2, tends to be the most popular backends for zkML, since the table-style arithmetization schemem can handle neural network non-linearities well via lookup arguments. However, such lookup table sacrifice the useability as it coming with a notable cost of prover memory consumption. With respect to the performance metric, other proof systems have their own benefits as well. For example, R1CS-based proof sytesm is outstanding for its small proof sizes and GKR-based appears best suited to tackle large models. While improvements and optimizations are being made in new proof system these years, accuracy loss generated during quantization is prevailed in almost all the systems.

Taken together, it's essential to highlight the main challenges of compiling neural networks to ZKP systems:

1. **Floating Point in ZKP:** Neural networks are often trained using floating point numbers. In zkML, this brings about the challenge of quantizing these numbers into fixed-point representations, without significant accuracy loss.
2. **Compatibility:** ZKP systems aren't inherently compatible with the complex operations (e.g. activitation function, matrix multiplication, etc) commonly used in neural networks. 
3. **Performance:** Crafting ZKP for ML models is a delicate act of balancing various trade-offs. Researchers must consider memory usage, proof size, and prover time to ensure optimal performance.

Therefore, some progress has been made in designing zkML-Specific proving systems to optimize the proof for the advanced ML models, including zkCNN, vCNN, and pvCNN. As the these names suggest, they are optimized for CNN models, and thus can only be applied to certain CV tasks, such as MNIST or CIFAR-10. 

In short, no exisitng proving systems is good enough to address these challenges and handle various ML tasks. This situation also applies to the zkML framework. This is why we need this benchamrk to provide a rigorous benchmark for the zkML ecosystem, enabling developers to make informed decisions based on empirical evidence. 


### Shortlisted Project

Because of time limitation, we only benchmark a shortlisted zkml projects to perform a comparison with respect to **A, B, C, and D**. It is worth noting that these selected projects certainly can not represent the comprehensive view of the current landscape. After closely examining almost all the exisitng open-sourced zkml projects, we try to find best examplers to represent the trends of zkml framework through a subset of these frameworks based on their **Github stars**, **PR activity**, **utility**, and **proof system**. We believe there must be few outstanding frameworks not be included this time, and promise to update the benchmark results in response to new coming ones. 

A basic information of shortlisted frameworks is provided as follows.


| Name         | Model Format | Star | Proof System |  Link |
| ---------    | ------------ | ---- | ------------ | -------|
| EZKL         |     ONNX     | 713  |   Halo 2 *   | [GitHub Repo](https://github.com/zkonduit/ezkl) |
| Orion        |     ONNX     | 132  |   ZK-STARK   | [GitHub Repo](https://github.com/gizatechxyz/orion) |
| DDKang ZKML  |   TFLite     | 314  |   Halo 2     | [GitHub Repo](https://github.com/ddkang/zkml) |
| keras2circom |   tf.keras   | 69   | R1CS Groth16   | [GitHub Repo](https://github.com/socathie/keras2circom) | 
| opML         |  ONNX (WiP)* | 65   | Fraud Proof***| [GitHub Repo](https://github.com/hyperoracle/opml) |


## Benchmark Methodology

### ZKML Score

ZK Score, as detailed in [this reference](https://medium.com/@ingonyama/zk-score-zk-hardware-ranking-standard-6bcc76414bc9), focuses on infrastructure benchmarks in the zkML field. The score is based on modular multiplications per Watt, offering a standard for comparing hardware efficiencies in the zkML space.

### ZKML Comparison Method Pitfalls

#### Complexity Theory and PoC Implementation
- Theoretical models don't always account for practical constants that determine performance.
- Choosing appropriate systems, protocols, and applications for comparison can be challenging.

### End-to-End Benchmarking Tools
- Existing tools like EF zkalc, The Pantheon of ZKP, and ZK Bench might not have fully optimized circuits, making it hard to draw meaningful conclusions.




> Discuss how the proving system will infect the prover time, memory size, etc. Then make conclusion that, directly compare these may not be fair cross different proving system. Instead, i will mark the difference, list pros and cons, leave the reader to decide which one is the best fit.
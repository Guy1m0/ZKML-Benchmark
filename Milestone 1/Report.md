# ZKML Benchmarking Project Report

## Introduction

As machine learning gains attention globally, ensuring the integrity and authenticity of machine learning models, particularly in Machine Learning as a Service (MLaaS), has become crucial. Zero-knowledge Machine Learning (zkML), and specifically zk-SNARK, emerges as a key player in bridging the trust gap, ensuring both computational integrity and data privacy. This project aims to develop a zkML benchmarking tool to guide developers through the complexities of the zkML landscape, focusing on the inference phase of ML models due to the high computational demands of the training phase.

## Current Landscape
Even various of proving systems have been introduced in recent years, and each of them is proved to be practical in certain application scenarieno and advanced in no trusted-setup, proving efficiency, deterministic proof size etc. However, naively converting a ML model to an arithmetic circuit is not feasible all the time, since the computation might be quite taxing for millions of paramters used in NN model. Therefore, many research works and framework have been proposed to fix this issues. The landscape can be categorized into several key areas:


> Include almost all the popular repo here...
- Model-to-Proof Compilers: Convert conventional ML model from a common format (e.g. ONNX, Keras, etc) into a verifiable computational circuit
  - **EZKL:** Converts ONNX models into zk-SNARK circuits.
  - **Orion:** 
  - **Circomlib-ml:** Transpiles Keras models into Circom circuits.
  - **Tachikoma&Uchikoma:** Bridges Apache TVM to arithmetic circuits in ZKP systems.
  - **Daniel Kang's zkml:** Constructs proofs of ML model execution in zk-SNARKs.

- zkML-Specific Proving Systems: Generate efficient verification of ML models with circuit-unfriendly operations
  - **zkCNN:** A novel approach for verifying convolutional neural networks.
  - **Zator:** Utilizes recursive zk-SNARKs for deep neural network verification.
  - **ZeroGravity:** *Zero Gravity is a system for proving an inference run (i.e. a classification) for a pre-trained, public WNN and a private input.*
    - [intro](https://hackmd.io/@benjaminwilson/zero-gravity)


- Hardware Acceleration: Build specialized hardware to support proof generation
  - **Supranational:** Offers GPU acceleration solutions.
  - **Accseal:** Develops ASIC chips for ZKP hardware acceleration.
  - **Icicle:**

- Application: Design for zkML use cases
  - **ZKP Neural Networks:** *evalutation of neural networks inside zero knowledge proofs*
  - **Worldcoin:** Integrates zkML for a privacy-preserving proof of personhood protocol.
  - **ZKaptcha:** Enhances captcha services by analyzing user behaviors through zkML.
  - **ZKaggle:** *Bounty platform for hosting, verifying, and paying out bounties*
  - **RockyBot:** *On-chain verifiable ML trading bot*
### Proving System

> ref Modulus Labs' "The Cost of Intelligence"

something like
> Plonkish proof systems tend to be the most popular backends for zkML for this reason. Halo2 and Plonky2 with their table-style arithmetization scheme can handle neural network non-linearities well via lookup arguments. In addition, the former has a vibrant developer tooling ecosystem coupled with flexibility, making it the de facto backend for many projects including EZKL.

> Other proof systems have their benefits as well. R1CS-based proof systems include Groth16 for its small proof sizes and Gemini for its handling of extremely large circuits and linear time prover. STARK-based systems like the Winterfell prover/verifier library are also useful especially when implemented via Giza’s tooling that takes a Cairo program’s trace as an input and generates a STARK proof using Winterfell to attest to the correctness of the output.

### Shortlisted Project

> create a table for each selected project in name, repo link, proving system, support, special feature


### Codebases
However, since not all of listed above project are 
#### Active Projects
   - **EZKL:** [GitHub Repo](https://github.com/zkonduit/ezkl) - Utilizes Halo2, supports ONNX.
   - **Orion:** [GitHub Repo](https://github.com/gizatechxyz/orion) - Based on STARKs, supports ONNX and TFLite models.
   - **DDKang ZKML:** [GitHub Repo](https://github.com/ddkang/zkml) - Utilizes Halo2  supports TFLite models.
   - **opML:** [GitHub Repo](https://github.com/hyperoracle/opml) - Currently working on ONNX support.
   - **circomlib-ml:** [GitHub Repo](https://github.com/socathie/circomlib-ml) - Likely based on Stark, supports TFLite models with Circom and SnarkJS.

#### Inactive Projects
   - **Zator, 0g, proto-neural-zkp:** [Various Repos] - Show less recent activity but contribute to the field's development.



## ZKML Score

ZK Score, as detailed in [this reference](https://medium.com/@ingonyama/zk-score-zk-hardware-ranking-standard-6bcc76414bc9), focuses on infrastructure benchmarks in the zkML field. The score is based on modular multiplications per Watt, offering a standard for comparing hardware efficiencies in the zkML space.

## ZKML Comparison Method Pitfalls

### Complexity Theory and PoC Implementation
- Theoretical models don't always account for practical constants that determine performance.
- Choosing appropriate systems, protocols, and applications for comparison can be challenging.

### End-to-End Benchmarking Tools
- Existing tools like EF zkalc, The Pantheon of ZKP, and ZK Bench might not have fully optimized circuits, making it hard to draw meaningful conclusions.


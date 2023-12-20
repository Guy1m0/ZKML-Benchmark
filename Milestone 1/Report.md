# ZKML Benchmarking Project Report

## Introduction

As machine learning gains traction globally, ensuring the integrity and authenticity of machine learning models, particularly in Machine Learning as a Service (MLaaS), has become crucial. Zero-knowledge Machine Learning (zkML), and specifically zk-SNARK, emerges as a key player in bridging the trust gap, ensuring both computational integrity and data privacy. This project aims to develop a zkML benchmarking tool to guide developers through the complexities of the zkML landscape, focusing on the inference phase of ML models due to the high computational demands of the training phase.

## Current Landscape
Even various of proving system have been introduced in recent years, and each of them is proved to be practical in certain application scenarieno that advanced in no trusted-setup, proving efficiency, deterministic proof size etc. However, naively converting a ML model to an supported arithmetic circuit is not feasible all the time, since the computation might be quite taxing for millions of paramters used in NN model. Therefore, many research works and framework have been proposed to  fix this issues. The landscape can be categorized into several key areas:

### 1. Model-to-Proof Compilers
These compilers convert conventional ML model from an common formats (e.g. ONNX, Keras, etc) into verifiable computational circuits:
   - **EZKL:** Converts ONNX models into zk-SNARK circuits.
   - **Circomlib-ml:** Transpiles Keras models into Circom circuits.
   - **Tachikoma&Uchikoma:** Bridges Apache TVM to arithmetic circuits in ZKP systems.
   - **Daniel Kang's zkml:** Constructs proofs of ML model execution in zk-SNARKs.
    - not sure about this 

### 2. zkML-Specific Proving Systems
Focused on efficient verification of ML models with circuit-unfriendly operations:
   - **zkCNN:** A novel approach for verifying convolutional neural networks.
   - **Zator:** Utilizes recursive zk-SNARKs for deep neural network verification.

### 3. Hardware Acceleration
Projects in this domain aim to build specialized hardware to support proof generation:
   - **Supranational:** Offers GPU acceleration solutions.
   - **Accseal:** Develops ASIC chips for ZKP hardware acceleration.

### 4. Applications
Applications of zkML in various sectors:
   - **Worldcoin:** Integrates zkML for a privacy-preserving proof of personhood protocol.
   - **ZKaptcha:** Enhances captcha services by analyzing user behaviors through zkML.

### Codebases
However, since not all of listed above project are 
#### Active Projects
   - **EZKL:** [GitHub Repo](https://github.com/zkonduit/ezkl) - Utilizes Halo2, supports ONNX.
   - **Orion:** [GitHub Repo](https://github.com/gizatechxyz/orion) - Based on STARKs, supports ONNX and TFLite models.
   - **DDKang ZKML:** [GitHub Repo](https://github.com/ddkang/zkml) - Utilizes SNARKs, supports TFLite models.
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

## Milestone 1: Research Existing zkML Frameworks

### Documentation
- A comprehensive document detailing the research and survey of the zkML ecosystem.
- Framework environment setup summaries, test results, and a pros and cons analysis for each framework.

### Testing Guide
- Methodologies used to test each framework, including memory usage, proof generation time, on-chain support, and accuracy loss.
- A deep dive into chosen frameworks, reviewing their documentation, demos, and unique features.

---

This revised report structure reflects the updated landscape of zkML frameworks and their current state, providing an in-depth view of the field for developers and researchers.
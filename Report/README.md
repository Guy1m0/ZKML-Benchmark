# Current Landscape

In today's rapidly growing zero-knowledge machine learning (zkML) areas, innovation is pervasive across multiple categories. Here's a snapshot of the current landscape:

1. **Model-to-Proof Compilers:** These tools essentially act as translators, converting familiar ML model formats into verifiable computational circuits. Notable contributions in this category include:
   * EZKL: Known for converting ONNX models into zk-SNARK circuits.
   * Circomlib-ml: Transpile a Keras model into a Circom circuit.
   * Tachikoma&Uchikoma: Bridge Apache TVM to arithmetic circuits in ZKP systems.
   * Daniel Kang's zkml: Construct proofs of ML model execution in zk-SNARKs
<!--    * Other significant players in this domain include the Nil Foundation and Risc Zero. -->
2. **zkML-Specific Proving Systems:** These systems are tools designed for ML models that are based on circuit-unfriendly operations. Their primary objective is to ensure efficient verification for these advanced models. Here's a concise overview of the current landscape in this category
   * zkCNN: Present a novel approach to tackle the challenges faced in verifying convolutional neural networks
   * Zator: Leverage recursive zk-SNARKs to verify deep neural networks.

3. **Hardware Acceleration:** Various projects focus on building special hardware (such as FPGA, GPU, and ASIC chip) to provide additional memory and support parallelizable computing to further accelerate proof generation. Standouts in this arena are:
   * Supranational: GPU acceleration solution
   * Accseal: ASIC chips for ZKP hardware acceleration

4. **Applications:**
   * Worldcoin: Integrate zkML to bolster their privacy-preserving proof of personhood protocol.
   * ZKaptcha: Employ zkML to elevate their captcha services by analyzing user behaviors.
   * Other notable entities in this category are Giza, Gensyn and Worldcoin.

# Codebases
A shortlist of keeping updated projects

> LinearA hasn't updated since Nov 2022
> zkCNN only updated till Feb 2023
> zator till May 2023

## Active 

* EZKL: [repo](https://github.com/zkonduit/ezkl)
   * Based on Halo2
      * Using Rust
   * Onnx Support
* Orion: [repo](https://github.com/gizatechxyz/orion)
   * Based on STARKs
      * Using Cairo (based on Rust)
   * Onnx Support?
   * TFLite model Support
* DDKang ZKML: [repo](https://github.com/ddkang/zkml)
   * Based on SNARKs
      * Using Rust
   * TFLite model Support
* opML: [repo](https://github.com/hyperoracle/opml/tree/main)
   * Onnx Support (WIP)
* circomlib-ml: [repo](https://github.com/socathie/circomlib-ml)
   * Probably based on Stark
      * Using Circom and SnarkJS
   * TFLite model

> Circom: Hardware Description Language (HDL )

## Inactive 

* Zator: [repo](https://github.com/lyronctk/zator)
   * Based on Nova and Spartan
* 0g: [repo](https://github.com/zkp-gravity/0g)
   * Based on Weightless NN mode
* proto-neural-zkp: [repo](https://github.com/worldcoin/proto-neural-zkp)
   * Based on Plonky2

## Generalized Model-to-Proof Compilers

* Risc Zero:
* Nil Foundation:
* 

### Risc Zero

### Nil Foundation

# ZKML Score
check the [ref](https://medium.com/@ingonyama/zk-score-zk-hardware-ranking-standard-6bcc76414bc9)

As mentioned in the above link:
> The primary value of the ZK Score lies in its simplicity. Currently, the ZK space is deeply involved in middleware and infrastructure R&D. We have very little happening at the application layer. Continuing our Aleo example, the foundation develops a prover to run applications, but we don’t know yet which applications are going to be killer applications. The same is true for ZKML. We know which ML we want to use, and infrastructure is getting better and better, but no real use cases are running at scale yet. This is why we think that for the time being and as a first step, we should focus on infrastructure benchmarks, starting with hardware.

ZK Score is based on the **modular multiplications** as the unit we measure for throughput, and the benchmark is runed by all bit sizes of interest (256-bit or 384-bit fields). It also needs measure power consumption over the same unit of time. Therefore, ZK Score definition of:

> Lots of tutorials provided is used to test on MNIST dataset, maybe not enought for practical use

**MMOPS/Watt == MMOP/Joule**

## ZKML Comparison Method Pitfalls

* Complexity Theory: 
   * No accounting for the constants that determine which one is better
   * ZK engineering is messy
* Adding PoC implementation
   * Which system or protocol we want to compare against (R1CS, Plonk+)
   * Selection of applications (Classification?)

* End-to-End Behcmkaring Tools: EF zkalc, The Pantheon of ZKP, and ZK bench
   * Implemented circuits might not be fully optimized
   * Hard to deduce meaningful conclusions by looking at the results


# Milestone 1: Research Existing zkML Frameworks
* Estimated Duration: 6 weeks
* FTE: 0.375
* Costs: US$5,625
* Estimated delivery date: 10th December 2023

**Deliverables and Specifications:**

0b. **Documentation**:
   * Draft a well-structured document detailing our research and survey process.
   * Summarize each selected framework's environment setup.
   * Record the basic test results for each.
   * Highlight the pros and cons of each framework.

0c. **Testing Guide**:
   * Enumerate the methodologies used to test each framework.
   * Discuss the hands-on benchmarking using the provided demos of the shortlisted frameworks.
   * Detail the parameters and results for:
     * Memory usage
     * Proof generation time
     * On-chain support
     * Accuracy loss

1. **Functionality: Preliminary Survey of zkML Landscape**:
   * Conduct a broad review of the zkML ecosystem.
   * Focus on the major categories, namely 'Model-to-proof Compilers' and 'zkML-specific Proving Systems'.
   * Select a subset of these frameworks based on their GitHub stars, PR activity, utility, and representation for a more detailed analysis.

2. **Functionality: Deep Dive into Chosen Frameworks**:
   * Conduct a thorough review of the provided documentation for each selected framework.
   * Engage actively with available demos, and if provided, the benchmarking tools.
   * Identify and understand their unique features, especially with respect to advantages in handling specific ML models or showcasing scalability.
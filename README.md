ZKML Benchmarking
===


# Introduction

As machine learning continues to expand in the global market, the reliability and integrity of its models have become paramount. Considering the case of Machine Learning as a Service (MLaaS), there's an inherent need to guarantee that the offered model truly matches its description (also noted as **model authenticity**), while operating within accurate parameters and maintaining a degree of privacy. Zero-knowledge proof, more specifically zero-knowledge Machine Learning (zkML), steps in to bridge this trust gap, vouching for both the integrity of computations and the confidentiality of underlying weights and structures.

To achieve this, zk-SNARK (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) has garnered significant attention. Its ability to produce short proofs, regardless of the size of the input data, makes it a prime candidate for integration with ML frameworks like EZKL and Daniel Kang's zkml. However, translating ML models into circuits optimized for these zero-knowledge-proof systems remains a challenge, even when the models aren't intricate neural networks. 

Consequently, progress has been made in crafting custom circuits using existing architectures, like zk-STARK, Halo2, and Plonk2. While these custom circuits can support the complex operations intrinsic to modern ML models, they often stray from being scalable generalized solutions. Naturally, this landscape presents developers with a quandary: which framework aligns best with their specific requirements?

Addressing this challenge, I'm developing a zkML benchmark. This tool aids developers in discerning the trade-offs and performance nuances among various frameworks. While manually setting up and conducting benchmarks for the increasing number of frameworks can be daunting, and though these frameworks often come with their benchmark sets, direct comparisons become intricate due to the myriad of performance-impacting variables. My approach has been to establish uniform conditions across all frameworks, aiming to offer a pragmatic and straightforward comparison among them.

It's worth noting that machine learning models generally undergo two phases: the **training phase** and the **inference phase**. Due to the computational complexity associated with verifying training in-circuit, this benchmark primarily focuses on frameworks designed for the inference stage of ML models. 


## Overview
While we recognize the vastness of this ecosystem, our proposal focuses on a selection of popular or representative frameworks. This concentration aims to give developers a clearer perspective when faced with the pivotal task of choosing a framework that aligns seamlessly with their project's requirements.


### Current Landscape

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


### Challenges
Before delving into the benchmark details, it's essential to highlight the prevailing challenges of compiling neural networks to ZKP systems:

1. **Floating Point in ZKP:** Neural networks are often trained using floating point numbers. In zkML, this brings about the challenge of quantizing these numbers into fixed-point representations, without significant accuracy loss.
2. **Compatibility:** ZKP systems aren't inherently compatible with the constructs commonly used in neural networks. While certain efforts, like those of EZKL and Zator, aim to address this issue, they often have to make sacrifices. For instance, while ONNX may offer scalability, it is supported by a limited number of operators in EZKL. On the other hand, Zator doesn't present a generalized solution.
3. **Performance:** Crafting ZKP for ML models is a delicate act of balancing various trade-offs. Researchers must consider memory usage, proof size, and prover time to ensure optimal performance.



# Project Description
Our project aims to provide a rigorous benchmark for the zkML ecosystem, enabling developers to make informed decisions based on empirical evidence. By standardizing the evaluation process, we will shed light on the strengths and weaknesses of various zkML frameworks, illuminating areas for improvement and guiding the community towards best practices. The project is divided into three key deliverables: 

1. Methodology for Benchmarking: This method will detail the procedures, tools, and metrics used to evaluate zkML frameworks. It will ensure consistent and repeatable testing, allowing for a fair comparison between different solutions.

2. Evaluation: Leveraging our established methodology, we will perform comprehensive evaluations of various zkML frameworks. By subjecting each framework to the same set of tests, we'll produce objective data that can be used to gauge the current state of the zkML ecosystem.

3. Documentation: To this end, we'll create documentation that outlines our methodology, presents our evaluation results, and discusses the implications of our findings. 


<!-- > test on the same tasks (CV and non-CV)and datasets (CIFAR10, MNIST).  -->
<!-- > May intentionally add something that is the challenge in the current zkML frameworks -->


# Plan Roadmap
This section describes the development roadmap for our zkML benchmarking project. We've organized our work into three main milestones, corresponding to our primary deliverables: research the current landscape, benchmarking, and documentation. 

## Overview

* Total Estimated Duration: 12 weeks
* Full-time equivalent (FTE): 0.375
* Starting Date: 30th Oct 2023
* Total Costs: US$11,250
* Cost Estimation: US$62.5/hour x 15 hours/week x 12 weeks






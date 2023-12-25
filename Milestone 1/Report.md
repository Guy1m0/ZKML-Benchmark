# ZKML Benchmarking Project Report

## Introduction

As machine learning gains global attention, ensuring the integrity and authenticity of machine learning models, particularly in Machine Learning as a Service (MLaaS), has become crucial. Zero-knowledge Machine Learning (zkML), and specifically zk-SNARK, emerges as a key technology in bridging the trust gap, ensuring both computational integrity and data privacy. This project aims to develop a zkML benchmarking tool to guide developers through the complexities of the zkML landscape, focusing on the inference phase of ML models due to the high computational demands of the training phase.

## Current Landscape

<!-- Despite the introduction of various proving systems in recent years, each excelling in specific applications and advancements in non-trusted-setup, proving efficiency, and deterministic proof size, the naive conversion of an ML model into an arithmetic circuit is not always feasible.  -->

Many research works and industrial frameworks have been proposed to convert an ML model into an arithmetic circuit. The landscape is categorized into several key areas:

<!-- Even various of proving systems have been introduced in recent years, and each of them is proved to be practical in certain application scenarieno and advanced in no trusted-setup, proving efficiency, deterministic proof size etc. However, naively converting a ML model to an arithmetic circuit is not feasible all the time, since the computation might be quite taxing for millions of paramters used in NN model. Therefore, many research works and industrial frameworks have been proposed to fix this issue. The landscape can be categorized into several key areas: -->

- Model-to-Proof Compilers: Convert conventional ML model formats (e.g. ONNX, Keras, etc) into verifiable computational circuits
  - **EZKL:** A library and command-line tool for converting ONNX models into zk-SNARK circuits.
  - **Orion:** Generates Validity ML enabling verification of inference by leveraging Cario and ONNX's capabilities.
  - **keras2circom&Circomlib-ml:** Python tools transpiling Keras models into Circom circuits.
  - **LinearA:** A framework bridging Apache TVM to arithmetic circuits in ZKP systems.
  - **Daniel Kang's zkml:** Constructs proofs for ML model execution written in Halo2.

- zkML-Specific Proving Systems: Generate efficient verification for ML models with circuit-unfriendly operations
  - **zkCNN:** A novel approach for verifying convolutional neural networks.
  - **Zator:** Utilizes recursive zk-SNARKs for deep neural network verification.
  - **ZeroGravity:** Verifies pre-trained, public WNN inferences with private inputs.


- Hardware Acceleration: Build specialized hardware to support proof generation
  - **Supranational:** GPU acceleration solutions.
  - **Accseal:** ASIC chips for ZKP hardware acceleration.
  - **Icicle:** CUDA implementation for general functions in zero-knowledge proof.

- Application: Design zkML use cases
  <!-- - **ZKP Neural Networks:** *evalutation of neural networks inside zero knowledge proofs* -->
  - **Worldcoin:** Integrates zkML for privacy-preserving proof of personhood protocols.
  - **ZKaptcha:** Enhances captcha services by analyzing user behaviors through zkML.
  - **ZKaggle:** A Bounty platform for hosting, verifying, and paying out bounties
  - **RockyBot:** On-chain verifiable ML trading bot.

### Proving System

Despite the introduction of various proving systems in recent years, each excelling in specific applications and advancements in non-trusted-setup, proving efficiency, and deterministic proof size, the naive conversion of an ML model into an arithmetic circuit is not always feasible.

Generalized proving system like Plonkish, STARK, R1CS and GKR are the backbone of the aforementioned zkML frameworks and crucial for realizing zkML. However, the naive conversion of an ML model into an arithmetic circuit is not always feasible.


For example, the advancement of verifiable ML is hindered by non-arithmetic operations, notably activitaion functions such as ReLU, sigmoid,and tanh. Plonkish-based systems, such as Halo2 and Plonky2, are popular due to their table-style arithmetization schemes that handle neural network non-linearities efficiently through lookup arguments. Yet, these lookups come with notable prover memory consumption costs. In terms of performance, other systems like R1CS excel in small proof sizes, and GKR-based systems seem best suited for large models. Despite the introduction of various proving systems in recent years, each excelling in specific applications and advancements in non-trusted-setup, proving efficiency, and deterministic proof size, almost all systems face accuracy loss during quantization.

Taken together, it's essential to highlight the main challenges of compiling neural networks to ZKP systems:

1. **Floating Point in ZKP:** Neural networks are often trained using floating point numbers. In zkML, this brings about the challenge of quantizing these numbers into fixed-point representations, without significant accuracy loss.
2. **Compatibility:** ZKP systems aren't inherently compatible with the complex operations (e.g. activitation function, matrix multiplication, etc) commonly used in neural networks. 
3. **Performance:** Crafting ZKP for ML models is a delicate act of balancing various trade-offs. Researchers must consider memory usage, proof size, and prover time to ensure optimal performance.

Therefore, some progress has been made in designing zkML-Specific proving systems to optimize the proof for the advanced ML models, including zkCNN, vCNN, and pvCNN. As these names suggest, they are optimized for CNN models, and thus can only be applied to certain CV tasks, such as MNIST or CIFAR-10. 

In summary, no exisitng proving systems sufficiently addresses these challenges across various ML tasks, highlighting the need for comprehensive benchmarking in the zkML ecosystem.


### Shortlisted Project

Due to time constraints, we benchmark a shortlisted selection of zkML projects to compare aspects such as **A, B, C, and D**. This selection does not represent the entire landscape but aims to exemplify current trends through a subset of frameworks. We plan to update the benchmark results in response to emerging frameworks.

<!-- Basic Information of Shortlisted Frameworks

Because of time limitation, we only benchmark a shortlisted zkml projects to perform a comparison with respect to **A, B, C, and D**. It is worth noting that these selected projects certainly can not represent the comprehensive view of the current landscape. After closely examining almost all the exisitng open-sourced zkml projects, we try to find best examplers to represent the trends of zkml framework through a subset of these frameworks based on their **Github stars**, **PR activity**, **utility**, and **proof system**. We believe there must be few outstanding frameworks not be included this time, and promise to update the benchmark results in response to new coming ones.  -->

A basic information of shortlisted frameworks is provided as follows.


| Name         | Model Format | Star | Proof System |  Link |
| ---------    | ------------ | ---- | ------------ | -------|
| EZKL         |     ONNX     | 713  |   Halo 2 *   | [GitHub Repo](https://github.com/zkonduit/ezkl) |
| Orion        |     ONNX     | 132  |   ZK-STARK   | [GitHub Repo](https://github.com/gizatechxyz/orion) |
| DDKang ZKML  |   TFLite     | 314  |   Halo 2     | [GitHub Repo](https://github.com/ddkang/zkml) |
| keras2circom |   tf.keras   | 69   | R1CS Groth16   | [GitHub Repo](https://github.com/socathie/keras2circom) | 
| opML         |  ONNX (WiP)* | 65   | Fraud Proof***| [GitHub Repo](https://github.com/hyperoracle/opml) |

Halo2*: EZKL customs halo2 circuits through aggregation proofs and recursion and optimizes the conversion using fusion and abstraction.

ONNX(WiP)**: Work in Progress

Fraud Proof***: Instead a traditional zk proofs to prove the validity of computation, opML provide an any-trust guarantee using the fraud proof system that any honest validator can force opML to behave correctly

## Benchmark Methodology

The methodology adopted for benchmarking zkML frameworks is designed to systematically evaluate and compare the capabilities and performance of various zk proof systems for the verifiable inference. This evaluation focuses on aspects crucial for practical deployment in MLaaS scenarios, such as proof generation time, memory usage, and system compatibility.


### Selection of zkML Frameworks

The selected frameworks for benchmarking include:

- **EZKL (Halo 2)**
- **Orion (ZK-STARK)**
- **DDKang ZKML (Halo 2)**
- **keras2circom (R1CS Groth16)**
- **opML (Fraud Proof)**

These frameworks have been chosen based on their popularity (indicated by GitHub stars), the proof system they utilize, and their support for different ML model formats. This diverse selection ensures a comprehensive analysis across various zk proof systems.

### Key Metrics for Evaluation

The primary metrics for benchmarking include:

1. **Proof Generation Time:** The duration taken by each framework to generate a proof. This metric is crucial for assessing the efficiency of the proof system.
2. **Maximum Prover Memory Usage:** The peak memory usage during the proof generation process, indicating the resource intensity of the system.
3. **Framework Compatibility:** Evaluation of each framework's compatibility with different machine learning model formats and operating systems.

### Planned Benchmarking Tasks
Our benchmarking methodology includes tasks on two computer vision (CV) datasets - MNIST and CIFAR10 - to evaluate the frameworks under different levels of complexity:

1. MNIST Dataset:
  - Simplicity of the Task: MNIST, being a dataset of handwritten digits, represents a less complex task, suitable for evaluating the basic capabilities of zkML frameworks.
  - Framework Assessment: We will observe how each framework handles relatively simple image data and whether it can maintain accuracy and efficiency.
2. CIFAR10 Dataset:
  - Increased Complexity: CIFAR10, with its more complex image data (like animals, vehicles, etc.), increases the challenge for zkML frameworks.
  - Parameter Variation: We will test the frameworks on this dataset with an increasing number of parameters and layers, pushing the boundaries of each framework's capacity.
  - Accuracy Loss Measurement: If applicable, we will measure and compare the accuracy loss across different frameworks, providing insight into their robustness and fidelity in more complex tasks.

### Benchmarking Process

1. **Circuit Design for MLPs:** For each zkML framework, we design circuits that represent Multi-Layer Perceptrons (MLPs), focusing on typical structures used in ML models.
2. **Uniform Testing Conditions:** To ensure fairness, the same benchmark suites are used across all frameworks. These suites include a consistent set of MLP architectures, varying in parameters, FLOPs, and number of layers.
3. **Encoding MLPs:** Each framework’s unique approach to encoding MLPs is critically analyzed. This involves understanding how each proof system converts ML operations into its respective arithmetization scheme.
4. **Exclusion of Pre-Processing Steps:** Measurements focus solely on the proof generation phase, deliberately excluding pre-processing or witness generation steps.
5. **Highlight Differences**:Clearly mark the differences in performance and capabilities across various proving systems. This includes detailing how each system performs under different benchmarking tasks and conditions.
6. **List Pros and Cons**: Provide a comprehensive list of advantages and disadvantages for each framework and its associated proving system. This helps in understanding the suitability of each framework for specific applications.

### Other Consideration

In addition to the primary metrics for evaluating zkML frameworks, several other considerations play a crucial role in our comprehensive benchmarking process. These considerations include the analysis of related benchmarking works, support from the community, the alignment of zkML frameworks with their underlying proving systems and the fairness in comparative analysis.

#### ZK Score

The ZK Score, as discussed by Ingonyama, is a significant metric in evaluating the efficiency of zk proof systems, particularly from a hardware acceleration perspective. It measures the throughput of modular multiplications per watt, offering a standard for comparing hardware efficiencies. While ZK Score provides valuable insights, especially in the context of hardware acceleration projects, its applicability to zkML frameworks needs careful consideration. 

- **Applicability Limitations:** ZK Score predominantly focuses on the hardware aspect and may not fully encapsulate the software or algorithmic efficiencies inherent in zkML frameworks. Thus, while useful, it might not be entirely suitable for a comprehensive assessment of all aspects of zkML frameworks.
<!-- #### Special Features in Frameworks

- **Specialized Circuit Designs:** Certain frameworks, like EZKL with its specialized Halo2 circuit, incorporate unique optimizations and features. These specialized designs can significantly impact performance and efficiency, and thus, warrant detailed analysis.
- **Optimization Techniques:** The benchmarking process also considers any special optimization techniques used within the frameworks, such as algorithmic improvements or adaptations made specifically for zk proof systems. -->

#### Support and Community Involvement

- **Well-Organized Committee:** The level of support and development from a well-organized committee or community behind a zkML framework is crucial. Active community involvement often leads to more robust and user-friendly frameworks.
- **Documentation Quality:** High-quality, comprehensive documentation is essential for the usability and accessibility of zkML frameworks. It significantly impacts the ease with which developers can adopt and implement these technologies.

#### Alignment with Proving Systems

- **Backbone Proving System Compatibility:** A key consideration is how well a zkML framework aligns with its underlying proving system. This includes examining if the framework leverages the full capabilities of the proving system or if any specific optimizations improve performance.
- **Framework-Specific Optimizations:** Any optimizations or modifications made to the proving systems to enhance their suitability for ML tasks are critically evaluated. This includes adaptations for handling complex operations typical in ML models, like matrix multiplications or activation functions.

#### Fairness in Comparative Analysis
Given the diversity in the underlying mechanics of these proving systems, a direct comparison across different systems may not always present a fair assessment. Each proving system is designed with specific goals and trade-offs in mind, making them suitable for different types of tasks and applications.

### Limitations and Future Work

While the methodology is comprehensive, it focuses on the prover side of the zk proof systems, leaving aspects like verifier runtime and proof size for future analysis. Furthermore, the evolving nature of zkML technologies means continuous updates and refinements to the benchmarking process will be necessary.

### Conclusion

These additional considerations ensure that our benchmarking methodology is well-rounded and covers aspects beyond basic performance metrics. By analyzing factors like other benchmark works, special features, community support, and alignment with proving systems, we aim to provide a more nuanced and holistic view of the zkML landscape. This comprehensive approach is crucial for understanding the full spectrum of capabilities and limitations of various zkML frameworks and guiding users in selecting the most appropriate tools for their specific needs.


<!-- 
### ZKML Score

ZK Score, as detailed in [this reference](https://medium.com/@ingonyama/zk-score-zk-hardware-ranking-standard-6bcc76414bc9), focuses on infrastructure benchmarks in the zkML field. The score is based on modular multiplications per Watt, offering a standard for comparing hardware efficiencies in the zkML space.

### ZKML Comparison Method Pitfalls

#### Complexity Theory and PoC Implementation
- Theoretical models don't always account for practical constants that determine performance.
- Choosing appropriate systems, protocols, and applications for comparison can be challenging.

### End-to-End Benchmarking Tools
- Existing tools like EF zkalc, The Pantheon of ZKP, and ZK Bench might not have fully optimized circuits, making it hard to draw meaningful conclusions.




> Discuss how the proving system will infect the prover time, memory size, etc. Then make conclusion that, directly compare these may not be fair cross different proving system. Instead, i will mark the difference, list pros and cons, leave the reader to decide which one is the best fit.

Taken together, we end the paper by examining two real world use-cases for verifiable inference that exemplify distinct axes of scale: model sophistication in the case of Worldcoin’s iris verification model, and cost-effective proof generation in the case of AI Arena’s gameplay verification. These two applications, along with emergent use-cases involving ambitious AI algorithms, motivate our next work in a ZK proving systems tailor-made for highly structured neural network operations — the Remainder proof system -->
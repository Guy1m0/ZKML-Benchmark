# Milestone 2: Design the Benchmark and Evaluation

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
2. **Accuracy:** 
3. **Maximum Prover Memory Usage:** The peak memory usage during the proof generation process, indicating the resource intensity of the system.
4. **Framework Compatibility:** Evaluation of each framework's compatibility with different machine learning model formats and operating systems.

### Planned Benchmarking Tasks
Our benchmarking methodology includes tasks on two computer vision (CV) datasets - MNIST and CIFAR10 - to evaluate the frameworks under different levels of complexity:

1. **MNIST Dataset:**
      - Simplicity of the Task: MNIST, being a dataset of handwritten digits, represents a less complex task, suitable for evaluating the basic capabilities of zkML frameworks.
      - Framework Assessment: We will observe how each framework handles relatively simple image data and whether it can maintain accuracy and efficiency.
2. **CIFAR10 Dataset:**
    - Increased Complexity: CIFAR10, with its more complex image data (like animals, vehicles, etc.), increases the challenge for zkML frameworks.
    - Parameter Variation: We will test the frameworks on this dataset with an increasing number of parameters and layers, pushing the boundaries of each framework's capacity.
    - Accuracy Loss Measurement: If applicable, we will measure and compare the accuracy loss across different frameworks, providing insight into their robustness and fidelity in more complex tasks.

### Benchmarking Process

The benchmarking process for evaluating zkML frameworks is structured to provide a comprehensive analysis of their performance across various metrics. Here’s an expanded breakdown of the steps involved:

1. **Circuit Design for MLPs:** For each zkML framework, we design circuits representing Multi-Layer Perceptrons (MLPs), focusing on structures typically used in machine learning models. These circuits are tailored to each framework's specifications and capabilities.

2. **Uniform Testing Conditions:** We employ the same benchmark suites across all frameworks to ensure consistent and fair testing conditions. These suites include MLP architectures with varying parameters, FLOPs, and number of layers, providing a uniform basis for comparison.
3. **Encoding MLPs:** Each framework's unique approach to encoding MLPs is critically examined. This involves understanding how each proof system translates machine learning operations into its respective arithmetization scheme.
4. **Exclusion of Pre-Processing Steps:** Our measurements concentrate exclusively on the proof generation phase, deliberately omitting pre-processing or witness generation steps to maintain a focused evaluation of proof generation efficiency.
5. **Modifying MLP Structures:** To assess scalability and robustness, we systematically modify the MLP structures within each framework by increasing the number of parameters and layers. This alteration simulates more complex ML models and highlights each framework’s adaptability and performance under escalated complexity.
6. **Comparative Performance Analysis:** We compare the performance of each framework under modified conditions. This includes analyzing how changes in MLP structures impact key metrics like proof generation time and memory usage.
7. **Highlight Differences:** Clear distinctions in performance and capabilities across various proving systems are marked. This includes detailed information on how each system responds to different benchmarking tasks and conditions, including those with modified MLP structures.
8. **List Pros and Cons:** A comprehensive list of the advantages and disadvantages for each framework and its associated proving system is provided. This helps in understanding the suitability of each framework for specific types of ML models, particularly as complexity increases.

9. **Framework-Specific Feature Evaluation:** Any unique or specialized features of each framework, such as specialized circuit designs or optimizations for particular types of ML models, are evaluated. This helps in understanding how these features contribute to the overall performance and utility of the framework.

Through this detailed benchmarking process, we aim to provide a nuanced understanding of each zkML framework’s capabilities, especially in handling increasingly complex machine learning models. This approach will guide users in selecting the most suitable framework for their specific requirements in the realm of zkML.

# Commit

> it seems v3.9.3 way much better than v7.1.4 in terms of witness gen 
* EZKL (release): v7.1.4 
* ddkang (commit hash): 43789582671f16148dd02cbac5654f462d4fe3e4
* Orion (release): v0.2.0
* Keras2circom (commit hash): 6e00b2366896f780d9a66b3483c1a988d9538bd5
* opml (commit hash): d1e0527983bdc97fa9f923121e6547925d5df31e


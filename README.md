# Constraint-Aware Neural Network Systems

This repository documents my UROP 1100 research work on hardware-efficient neural networks and signal classification.

The project is organized into two connected parts:

- **LutNet**: hardware-aware pruning and packing for LUT-based neural networks targeting FPGA efficiency

- **RadioML**: CNN-based modulation classification experiments for studying representation learning under noisy conditions

Although the two parts use different datasets and models, they are connected by a common question:

> How can model structure and feature representation be designed to improve both learning behavior and hardware efficiency?

---

1. LutNet: Hardware-Aware Pruning and FPGA Efficiency

The LutNet/ directory focuses on analyzing and debugging a LUT-based neural network training pipeline.

Key topics include:

* sensitivity-based global pruning
* threshold tie problems
* pruning consistency
* structural packing
* FPGA-oriented resource reduction

This part of the project is centered on understanding why naive pruning can fail in practice, especially when many sensitivity values are tied at the threshold, and how improved tie-breaking logic leads to more stable sparsity control and better hardware mapping behavior.

See LutNet/README.md￼ for details.

⸻

2. RadioML: CNN-Based Modulation Classification

The RadioML/ directory focuses on modulation classification using the RadioML dataset.

This part of the project investigates:

* baseline 1D CNN performance on raw IQ signals
* class-wise and SNR-wise failure patterns
* frequency-aware feature engineering
* branch-based architectures using IQ and instantaneous frequency (IF)

A major goal here is not just improving accuracy, but also understanding why the model fails under certain conditions, especially:

* low-SNR collapse
* AM-SSB sink behavior
* WBFM misclassification

See RadioML/README.md￼ for details.

⸻

My Main Contributions

Across the repository, my work includes:

* analyzing and reverse-engineering research code pipelines
* writing debugging and analysis scripts
* comparing experiment runs and summarizing trade-offs
* studying structured failure modes using plots, summaries, and confusion-style analysis
* reorganizing experiments into clearer model / data / training / analysis workflows

⸻

Notes

* This repository emphasizes analysis, interpretation, and experiment structure
* It does not include the full original collaborative research codebase
* Some code has been simplified or reorganized to highlight my direct contributions more clearly

⸻

Directory Guide

LutNet/

Use this folder if you want to understand:

* hardware-aware pruning logic
* threshold tie problems
* FPGA efficiency trade-offs

RadioML/

Use this folder if you want to understand:

* CNN baselines for modulation classification
* how different feature representations affect performance
* how failure modes change across classes and SNR levels

⸻

Summary

This repository reflects a progression from:

* hardware-aware pruning and structural efficiency
    to
* representation-aware learning and model behavior analysis

Together, these experiments helped me understand both:

* how models are optimized for hardware constraints
* how feature and architecture choices affect learning performance

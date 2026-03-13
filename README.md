# CVPR'26

# CoFiDA-M: Concept-Aware Feature Modulation for Cross-Domain Adaptation with Image-Only Inference 
<p align="center">
  <img width="612" alt="Model architecture" src="Screenshot 2026-03-10 at 10.55.45.png" />
</p>

## Abstract
Models for AI-based skin cancer screening suffer a severe performance drop when shifting from expert dermoscopic (source) images to consumer-grade clinical (target) photos, hindering real-world deployment. Existing domain adaptation methods often ignore crucial semantic invariants, such as clinical concepts. While new foundation models like MONET can provide this semantic information as dense, probabilistic scores, this metadata is unavailable at test time, creating a deployment paradox for practical image-only screening tools. We address this gap by proposing CoFiDA-M, a privileged information framework that learns from concepts at training time but deploys as an image-only model. Our method trains a teacher network that uses MONET concept probabilities to guide a FiLM modulator, transforming visual features into a semantically ``edited" feature space. A lightweight, image-only student is then trained to reproduce this edited representation, not just the teacher's final predictions. This distillation "bakes" the clinical reasoning into the student's weights. On a challenging multi-dataset benchmark, our image-only student significantly outperforms state-of-the-art approaches, especially in melanoma recall. Our work provides a practical and generalizable framework for leveraging noisy, probabilistic metadata as privileged information, demonstrating strong cross-dataset robustness and potential for real-world deployment beyond dermatology. The code will be made publicly available upon acceptance.


Code: Coming Soon...

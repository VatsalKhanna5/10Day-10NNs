# 10 Days - 10 Deep Learning Projects

![Deep Learning Journey](https://img.shields.io/badge/Deep%20Learning-Journey-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive deep learning curriculum featuring 10 hands-on projects, advancing from basic neural networks to cutting-edge architectures. Each day builds upon previous knowledge, providing a structured learning path for deep learning enthusiasts.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Projects](#projects)
  - [Day 1: Forward Neural Network](#day-1-forward-neural-network)
  - [Day 2: Convolutional Neural Networks](#day-2-convolutional-neural-networks)
  - [Day 3: Recurrent Neural Networks](#day-3-recurrent-neural-networks)
  - [Day 4: Generative Adversarial Networks](#day-4-generative-adversarial-networks)
  - [Day 5: Transformer Models](#day-5-transformer-models)
  - [Day 6: Object Detection](#day-6-object-detection)
  - [Day 7: Image Segmentation](#day-7-image-segmentation)
  - [Day 8: Reinforcement Learning](#day-8-reinforcement-learning)
  - [Day 9: AutoEncoders & VAEs](#day-9-autoencoders--vaes)
  - [Day 10: Advanced Project](#day-10-advanced-project)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Appendix](#appendix)

## Overview

This repository contains 10 deep learning projects designed to be completed over 10 days. Each project builds upon knowledge gained from previous days, starting with fundamental neural networks and progressing to advanced architectures and applications. The projects use both TensorFlow and PyTorch to provide exposure to the two most popular deep learning frameworks.

## Prerequisites

- Python 3.8+
- Basic understanding of machine learning concepts
- Familiarity with Python programming
- Knowledge of NumPy and Pandas
- A computer with GPU (recommended for later projects)

## Projects

### Day 1: Forward Neural Network

**Description:** Build your first fully connected neural network (also known as multilayer perceptron) using TensorFlow. This project introduces the fundamental concepts of neural networks, including layers, activation functions, backpropagation, and optimization.

**Key Concepts:**
- Neural Network Architecture
- Forward and Backward Propagation
- Activation Functions
- Loss Functions
- Optimization Algorithms

**Implementation:**
1. TensorFlow Implementation
   


**Dataset:** MNIST for handwritten digit classification

**Expected Outcomes:**
- Working neural network with >95% accuracy on MNIST
- Understanding of neural network components
- Ability to implement networks in both frameworks


**Further Learning Resources:**
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [PyTorch Documentation](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen

### Day 2: Convolutional Neural Networks

**Description:** Learn and implement Convolutional Neural Networks (CNNs) for image classification tasks. This project covers the essential building blocks of CNNs and their application in computer vision.

**Key Concepts:**
- Convolutional Layers
- Pooling Operations
- Feature Maps
- CNN Architectures (LeNet, AlexNet)
- Image Classification

**Dataset:** CIFAR-10 for object recognition

**Expected Outcomes:**
- Working CNN with >80% accuracy on CIFAR-10
- Visualization of learned filters
- Understanding of CNN operations

### Day 3: Recurrent Neural Networks

**Description:** Dive into sequence modeling with Recurrent Neural Networks (RNNs), including LSTM and GRU variants for tasks like text generation and time series prediction.

**Key Concepts:**
- Sequence Modeling
- RNN Architecture
- LSTM and GRU Cells
- Sequence Generation
- Time Series Forecasting

**Dataset:** Shakespeare text corpus for language modeling

**Expected Outcomes:**
- Working RNN for text generation
- Understanding of sequence data handling
- Comparison of different RNN cell types

### Day 4: Generative Adversarial Networks

**Description:** Explore the fascinating world of GANs by building models that can generate realistic images from random noise.

**Key Concepts:**
- Generator and Discriminator Networks
- Adversarial Training
- Mode Collapse
- Image Generation
- GAN Variants (DCGAN)

**Dataset:** CelebA or Fashion-MNIST

**Expected Outcomes:**
- Working GAN generating realistic images
- Understanding of adversarial training dynamics
- Visualization of the generation process

### Day 5: Transformer Models

**Description:** Implement attention mechanisms and transformer architecture for natural language processing tasks.

**Key Concepts:**
- Self-Attention Mechanism
- Multi-Head Attention
- Positional Encoding
- Encoder-Decoder Architecture
- Transfer Learning with Pre-trained Models

**Dataset:** English-French translation dataset

**Expected Outcomes:**
- Working transformer for machine translation
- Understanding of attention mechanisms
- Visualization of attention weights

### Day 6: Object Detection

**Description:** Build object detection systems using techniques like YOLO or SSD to identify and locate multiple objects in images.

**Key Concepts:**
- Bounding Box Prediction
- Anchor Boxes
- Non-Maximum Suppression
- IOU (Intersection Over Union)
- Object Detection Architectures

**Dataset:** Pascal VOC or COCO subset

**Expected Outcomes:**
- Working object detector
- Real-time detection capabilities
- Understanding of detection metrics

### Day 7: Image Segmentation

**Description:** Implement semantic segmentation to classify each pixel in an image, enabling detailed scene understanding.

**Key Concepts:**
- Pixel-wise Classification
- U-Net Architecture
- Encoder-Decoder Segmentation
- Evaluation Metrics (IoU, Dice)
- Transfer Learning for Segmentation

**Dataset:** Cityscapes or PASCAL VOC Segmentation

**Expected Outcomes:**
- Working segmentation model
- Pixel-accurate scene understanding
- Visualization of segmentation masks

### Day 8: Reinforcement Learning

**Description:** Apply deep reinforcement learning to train agents that learn to perform tasks through interaction with an environment.

**Key Concepts:**
- Agent-Environment Interaction
- Policy Gradients
- Q-Learning and DQN
- Experience Replay
- Reward Engineering

**Environment:** OpenAI Gym (CartPole, LunarLander)

**Expected Outcomes:**
- Working RL agent solving a game
- Understanding of RL principles
- Learning curves visualization

### Day 9: AutoEncoders & VAEs

**Description:** Build autoencoders and variational autoencoders for dimensionality reduction, denoising, and generative modeling.

**Key Concepts:**
- Latent Space Representation
- Encoder-Decoder Structure
- Reconstruction Loss
- Variational Inference
- Generative Modeling

**Dataset:** MNIST or Fashion-MNIST

**Expected Outcomes:**
- Working autoencoder for image reconstruction
- Latent space visualization
- Image generation capabilities

### Day 10: Advanced Project

**Description:** Combine multiple techniques learned throughout the course to build an advanced deep learning application of your choice.

**Project Options:**
- Style Transfer Implementation
- Image-to-Image Translation
- Multi-modal Deep Learning
- Custom GAN Application
- Reinforcement Learning for Complex Environment

**Expected Outcomes:**
- End-to-end advanced deep learning project
- Integration of multiple techniques
- Deployment-ready application

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/10-days-dl-projects.git
cd 10-days-dl-projects
pip install -r requirements.txt
```

For GPU support (highly recommended for later projects):

```bash
pip install tensorflow-gpu
# or for PyTorch
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Usage

Each day's project is contained in its own directory. Navigate to the specific day and run the main script:

```bash
cd day1
python main.py
```

Alternatively, open and run the Jupyter notebooks:

```bash
jupyter notebook day1/notebook.ipynb
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- TensorFlow and PyTorch communities
- Dataset providers
- Deep learning researchers whose papers informed these implementations
- All contributors and users of this educational resource

## Appendix
### Mathematical Foundations

- [Linear Algebra](#linear-algebra)
- [Calculus](#calculus)
- [Probability and Statistics](#probability-and-statistics)
- [Neural Network Mathematics](#neural-network-mathematics)
- [Optimization Theory](#optimization-theory)

### Linear Algebra

Essential linear algebra concepts used throughout the projects:

#### Vectors and Matrices

- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ - Column vector

- $\mathbf{A} \in \mathbb{R}^{m \times n}$ - Matrix with $m$ rows and $n$ columns

 $\mathbf{A} = \left(\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{array}\right)$

#### Matrix Operations

Matrix multiplication: $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}$

Transpose: $\mathbf{A}^T_{ij} = \mathbf{A}_{ji}$

Trace: $\text{Tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$

Determinant: $\det(\mathbf{A})$ or $|\mathbf{A}|$

#### Eigenvalues and Eigenvectors

$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$

Where $\lambda$ is an eigenvalue and $\mathbf{v}$ is the corresponding eigenvector.

### Calculus

Key calculus concepts used in deep learning:

#### Derivatives

Scalar derivative: $\frac{df(x)}{dx}$

Partial derivative: $\frac{\partial f(\mathbf{x})}{\partial x_i}$

Gradient: $\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$

#### Chain Rule

$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$

For multivariate functions: $\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$

### Probability and Statistics

Core probability concepts in deep learning:

#### Probability Distributions

Gaussian distribution: $p(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

Bernoulli distribution: $p(x|p) = p^x (1-p)^{1-x}$

#### Information Theory

Entropy: $H(X) = -\sum_i p(x_i) \log p(x_i)$

Cross-entropy: $H(p,q) = -\sum_i p(x_i) \log q(x_i)$

KL Divergence: $D_{KL}(p||q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$

### Neural Network Mathematics

Mathematical formulations for neural networks:

#### Forward Pass

For a single neuron: $y = \sigma\left(\sum_{i=1}^n w_i x_i + b\right) = \sigma(\mathbf{w}^T\mathbf{x} + b)$

For a layer with activation function $\sigma$: $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$

#### Common Activation Functions

Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$

ReLU: $f(x) = \max(0, x)$

Softmax: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$

Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

#### Backpropagation

Chain rule application: $\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l+1)}} \cdot \frac{\partial a_j^{(l+1)}}{\partial z_j^{(l+1)}} \cdot \frac{\partial z_j^{(l+1)}}{\partial w_{ij}^{(l)}}$

### Optimization Theory

Mathematical concepts behind training algorithms:

#### Gradient Descent

Update rule: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$

#### Stochastic Gradient Descent (SGD)

Update rule: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})$

#### Momentum

Update rules:
$\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \alpha \nabla_\theta J(\theta_t)$
$\theta_{t+1} = \theta_t - \mathbf{v}_t$

#### Adam Optimizer

Update rules:

$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta_t)$
$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta_t))^2$
$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$
$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$
$\theta_{t+1} = \theta_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$

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

**Description:** Build your first fully connected neural network (also known as multilayer perceptron) using both TensorFlow and PyTorch. This project introduces the fundamental concepts of neural networks, including layers, activation functions, backpropagation, and optimization.

**Key Concepts:**
- Neural Network Architecture
- Forward and Backward Propagation
- Activation Functions
- Loss Functions
- Optimization Algorithms

**Implementation:**
1. TensorFlow Implementation
   - Sequential API
   - Custom Model Subclassing
   - Training and Evaluation
   
2. PyTorch Implementation
   - nn.Module and nn.Sequential
   - Custom Network Implementation
   - Training Loop from Scratch

**Dataset:** MNIST for handwritten digit classification

**Expected Outcomes:**
- Working neural network with >95% accuracy on MNIST
- Understanding of neural network components
- Ability to implement networks in both frameworks

**Code Structure:**
```
day1/
├── tensorflow_implementation/
│   ├── sequential_model.py
│   └── custom_model.py
├── pytorch_implementation/
│   ├── sequential_model.py
│   └── custom_model.py
└── README.md
```

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

# 🧠 Neural Network From Scratch in Python

This repository contains the full implementation of a **simple neural network built from scratch** in Python with Pytorch, as demonstrated in my [Medium article](https://medium.com/@jkosla/introduction-to-neural-networks-building-a-neural-network-from-scratch-in-python-with-d7b84b2b64b7).

---

## 📖 Overview

In this project, we build a **fully connected neural network** from scratch, using Pytorch only to perform faster matrices calculations on GPU. The implementation includes:

- **Forward Propagation**
- **Backpropagation**
- **Training with Gradient Descent**
- **Evaluation & Accuracy Calculation**

The code is designed to be simple and educational, demonstrating the core concepts of neural networks. Perfect for beginners who want to understand how neural networks work under the hood!

---

## 📂 Files

- `simple_nn.py` — Implementation of the neural network.
- `train.py` — Training script with evaluation functions.
- `dataset.py` — Helper functions for data processing and evaluation.
---

## 💡 Getting Started

### Requirements
- Python 3.x
- NumPy
- PyTorch

Install dependencies:
```bash
pip install numpy
pip install matplotlib
pip install torch
```
## Usage:

```python
%matplotlib inline
from dataset import XORDataset, visualize_samples, visualize_classification
import matplotlib.pyplot as plt
import torch.utils.data as data
from simple_nn import SimpleClassifier, GradientDescent
import torch
from train_nn import train_model, eval_model
```


```python

num_inputs = 2
num_hidden = 4
num_outputs = 1

train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, drop_last=False
)


model = SimpleClassifier(num_inputs, num_hidden, num_outputs)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device
optimizer = GradientDescent(lr=0.01)
```


```python
_ = visualize_classification(model, test_dataset.data, test_dataset.label)
```


    
![png](output_2_0.png)
    



```python
train_model(model, train_data_loader)
eval_model(model, test_data_loader)
_ = visualize_classification(model, test_dataset.data, test_dataset.label)
```

    Accuracy of the model: 100.00%



    
![png](output_3_1.png)
    


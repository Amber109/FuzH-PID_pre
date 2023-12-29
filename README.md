# FuzH-PID: Highly controllable and stable DNN for COVID-19 detection via improved Stochastic Optimization

## Introduction

Welcome to the "FuzH-PID: Highly controllable and stable DNN for COVID-19 detection via improved Stochastic Optimization" repository. This project focuses on the application of deep learning techniques to automatically diagnose Coronavirus Disease (COVID-19) using chest Computed Tomography (CT) images. By leveraging the power of machine learning and medical imaging, our goal is to assist healthcare professionals in the early detection and diagnosis of COVID-19 cases.


## Programming Environment

To set up the programming environment for this project, follow these steps:

### Prerequisites

- MATLAB (version 2019.B)
- Required Matlab Toolbox:
  - Image Processing Toolbox
  - Deep Learning Toolbox
  - Computer Vision Toolbox
  - Parallel Computing Toolbox
  - Optimization Toolbox
  - Fuzzy Logic Toolbox
- Python (version 3.7.10)
- Required Python packages:
  - TensorFlow
  - PyTorch
  - scikit-learn
  - NumPy
  - Pandas
  - OpenCV
  - Pillow (PIL)

### Installation

1. **MATLAB Setup**:
   - Download and install MATLAB version 2019.B from the official MathWorks website.
   - Ensure that you have the required MATLAB toolboxes.

2. **Python Setup**:
   - Install Python version 3.7.10 on your system from the official Python website.
   - Use a virtual environment to manage your project-specific dependencies. You can create one using `virtualenv` or `conda`.
   - Install the necessary Python packages using pip:
     ```shell
     pip install tensorflow torch scikit-learn numpy pandas opencv-python pillow
     ```

## Usage

In this repository, you will find two versions of the FuzH-PID model dedicated to the automatic diagnosis of Coronavirus Disease (COVID-19) using chest CT images. These models are designed to support healthcare professionals and researchers by providing robust diagnostic tools in the fight against this global health challenge.

### Python Version

   - FuzH-PID.py: For Python enthusiasts, we offer a Python implementation of the model, which is perfect for those who are accustomed to Python's ecosystem for deep learning applications. This python file runs the FuzH-PID model, which combines fuzzy logic and neural networks for COVID-19 diagnosis based on chest CT images. It encapsulates the core functionality of the FuzH-PID model with potential hereditary and serves as a key component for diagnostic tasks. To run it, load your pre-trained model, provide the image path, and specify the class index. Comprehensive documentation is available in the Python directory's `README.md`, detailing the setup process, usage instructions, and ways to tailor the model to your specific needs.###

### Matlab Version

  - FuzH-PID_CNN_covid.m: The Matlab version of our model is tailored for users who prefer a Matlab environment for their medical image processing and deep learning tasks. This MATLAB file defines the neural network structure and main functions. To use it, specify the training and testing data paths, the number of epochs, and the learning rate. This MATLAB file runs the FuzH-PID_CNN model, which combines fuzzy logic and neural networks for COVID-19 diagnosis based on chest CT images. It encapsulates the core functionality of the FuzH-PID model with potential hereditary and serves as a key component for diagnostic tasks.
   - PID_DRAW.m: The "PID_DRAW.m" MATLAB file is responsible for generating curves of training loss and validation loss. These curves help visualize the control actions and decision-making processes of the FuzH-PID model, offering insights into how the model makes diagnostic decisions.
   - Trainer.m: "Trainer.m" is a MATLAB script designed for setting up the training process of the FuzH-PID model. It provides the configuration and initialization for training, allowing you to customize training parameters according to your specific needs.
   - trainNetworkFuzzy.m: This MATLAB script, "trainNetworkFuzzy.m," can serve as a valuable tool for neural network training tasks, not only on COVID-19 diagnosis but also on image data such as the MNIST dataset.
The model's functions and dependencies are all contained within this directory. For a detailed explanation of each function and how to customize the model to your dataset, refer to the `README.md` within the Matlab directory.

Both versions of the model are meticulously documented and come with example datasets for testing purposes. We encourage you to explore both and choose the one that best fits your computational environment and personal preference.

## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   
@article{,
  title={FuzH-PID: Highly controllable and stable DNN for COVID-19 detection via improved Stochastic Optimization},
  author={Xujing Yao, Cheng Kang, Xin Zhang, Shuihua Wang, Yudong Zhang},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

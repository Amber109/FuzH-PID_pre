# Automatic Diagnosis of Coronavirus Disease on Chest CT Images Utilizing Deep Learning

## Introduction

Welcome to the "Automatic Diagnosis of Coronavirus Disease on Chest CT Images Utilizing Deep Learning" repository. This project focuses on the application of deep learning techniques to automatically diagnose Coronavirus Disease (COVID-19) using chest Computed Tomography (CT) images. By leveraging the power of machine learning and medical imaging, our goal is to assist healthcare professionals in the early detection and diagnosis of COVID-19 cases.
<div align="center">
  <img src="UI.jpg"/>
</div><br/>


## Programming Environment

To set up the programming environment for this project, follow these steps:

### Prerequisites

- MATLAB (version 2021.B)
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
   - Download and install MATLAB version 2019.A from the official MathWorks website.
   - Ensure that you have the required MATLAB toolboxes.

2. **Python Setup**:
   - Install Python version 3.7.10 on your system from the official Python website.
   - Use a virtual environment to manage your project-specific dependencies. You can create one using `virtualenv` or `conda`.
   - Install the necessary Python packages using pip:
     ```shell
     pip install tensorflow torch scikit-learn numpy pandas opencv-python pillow
     ```

## Usage
In this repository, you will find three distinct deep learning models tailored for the automatic diagnosis of Coronavirus Disease (COVID-19) based on chest CT images. Each model offers a unique approach to this critical task, empowering healthcare professionals and researchers with powerful diagnostic tools.

1. **GBBNet (Model1)**:
   - COVIDEXTRACT_Fea_v5.m: This MATLAB file is used for feature extraction. To run it, provide the input image folder and output feature folder as arguments.
   - COVID_Main.m: COVID_Main.m: This MATLAB file defines the neural network structure and main functions. To use it, specify the training and testing data paths, the number of epochs, and the learning rate.
   - Gradcam.m: This MATLAB file is for generating Grad-CAM maps. To run it, load your pre-trained model, provide the image path, and specify the class index.
     
2. **AdaD-FNN (Model2)**:
   - U2MNet-sample.ipynb: This Jupyter Notebook file is designed for running the U2-net model to perform segmentation of CT images. 
   - adadfnn-sample.ipynb: This Jupyter Notebook file is used to run the AdaD-FNN ensemble learning model for classification. 
2. **FuzzyPID (Model3)**:
   - FuzzyPID_CNN_covid.m: This MATLAB file runs the Fuzzy PID CNN model, which combines fuzzy logic and neural networks for COVID-19 diagnosis based on chest CT images. It encapsulates the core functionality of the Fuzzy PID CNN model and serves as a key component for diagnostic tasks.
   - PID_DRAW.m: The "PID_DRAW.m" MATLAB file is responsible for generating PID (Proportional-Integral-Derivative) curves. These curves help visualize the control actions and decision-making processes of the Fuzzy PID CNN model, offering insights into how the model makes diagnostic decisions.
   - Trainer.m: "Trainer.m" is a MATLAB script designed for setting up the training process of the Fuzzy PID CNN model. It provides the configuration and initialization for training, allowing you to customize training parameters according to your specific needs.
   - trainNetworkFuzzy.m: This MATLAB script, "trainNetworkFuzzy.m," is tailored for training a neural network using the MNIST dataset. While not specific to COVID-19 diagnosis, it can serve as a valuable tool for neural network training tasks, particularly on image data such as the MNIST dataset.

## Citation
If you find this project useful for your research, please consider citing: 
```bibtex   
@article{yao2021csgbbnet,
  title={CSGBBNet: An explainable deep learning framework for COVID-19 detection},
  author={Yao, Xu-Jing and Zhu, Zi-Quan and Wang, Shui-Hua and Zhang, Yu-Dong},
  journal={Diagnostics},
  volume={11},
  number={9},
  pages={1712},
  year={2021},
  publisher={MDPI}
}
@article{yao2022adad,
  title={AdaD-FNN for chest CT-based COVID-19 diagnosis},
  author={Yao, Xujing and Zhu, Ziquan and Kang, Cheng and Wang, Shui-Hua and Gorriz, Juan Manuel and Zhang, Yu-Dong},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  volume={7},
  number={1},
  pages={5--14},
  year={2022},
  publisher={IEEE}
}
```

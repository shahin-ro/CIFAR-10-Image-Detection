# CIFAR-10 Image Classification with TensorFlow 🚀

Welcome to the **CIFAR-10 Image Classifier** project! This repository
contains a simple yet effective Convolutional Neural Network (CNN) built
using TensorFlow to classify images from the CIFAR-10 dataset into 10
categories (e.g., airplanes, cars, birds). 🖼️

## 📋 Project Overview

This project implements a CNN to classify 32x32 RGB images from the
CIFAR-10 dataset. It includes:

-   **Data Preparation**: Loading and normalizing CIFAR-10 data. 📊
-   **Model Architecture**: A sequential CNN with Conv2D,
    BatchNormalization, MaxPooling, and Dense layers. 🧠
-   **Training**: 30 epochs with Adam optimizer and categorical
    cross-entropy loss. ⚙️
-   **Evaluation**: Accuracy/loss plots and ROC curves for performance
    analysis. 📈
-   **Prediction**: Upload custom images for real-time classification.
    🔍

Perfect for machine learning beginners and enthusiasts! 🌟

## 📁 Repository Structure

-   `CIFAR_10_Image_Detection.ipynb`: Jupyter Notebook with the full
    code.
-   `README.md`: You're reading it! 😄

## 🛠️ Requirements

-   Python 3.x
-   TensorFlow
-   NumPy
-   Matplotlib
-   Scikit-learn
-   OpenCV (cv2)
-   Jupyter Notebook

Install dependencies:

``` bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

## 🚀 Getting Started

1.  **Clone the repo**:

    ``` bash
    git clone https://github.com/shahin-ro/CIFAR-10-Image-Detection.git
    ```

2.  **Run the notebook**: Open `CIFAR_10_Image_Detection.ipynb` in
    Jupyter Notebook or Google Colab.

3.  **Train the model**: Execute the cells to load data, train the CNN,
    and evaluate performance.

4.  **Test with your images**: Upload images to classify them using the
    trained model. 🖼️

## 📊 Model Architecture

The CNN consists of:

-   2 Conv2D layers (32 filters) + BatchNorm + MaxPooling
-   2 Conv2D layers (64 filters) + BatchNorm + MaxPooling
-   GlobalAveragePooling + Dense layer (10 classes, softmax)

Total parameters: \~66K. Lightweight and efficient! ⚡

## 📈 Results

-   **Training Accuracy**: \~95% after 30 epochs.
-   **Validation Accuracy**: \~70-73% (room for improvement with
    tuning).
-   Visualizations: Loss/accuracy plots and ROC curves included. 📉

## 🖼️ Test Your Images

Upload any 32x32 RGB image to see the model's prediction. The notebook
displays the image with its predicted class. 🎨

## 💡 Future Improvements

-   Add data augmentation to improve generalization. 🔄
-   Experiment with deeper architectures or dropout. 🧪
-   Fine-tune hyperparameters for better accuracy. 🎯

## 📜 License

This project is licensed under the MIT License. Feel free to use and
modify! 📄

## 🙌 Contributing

Contributions are welcome! Fork the repo, make changes, and submit a
pull request. Let's improve this together! 🤝

## 📬 Contact

Have questions? Reach out via GitHub Issues or email. 📧

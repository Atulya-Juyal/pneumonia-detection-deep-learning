<p align="center">
</p>
<h1 align="center">Pneumonia Detection using Deep Learning</h1>
<p align="center">
  A deep learning model to classify chest X-ray images for the diagnosis of pneumonia.
</p>

<p align="center">
  <a href="#"><img alt="Domain" src="https://img.shields.io/badge/Domain-Medical%20Imaging-orange?style=for-the-badge"></a>
  <a href="#"><img alt="Model" src="https://img.shields.io/badge/Model-CNN-blue?style=for-the-badge"></a>
  <a href="#"><img alt="Framework" src="https://img.shields.io/badge/Framework-TensorFlow%2FKeras-red?style=for-the-badge"></a>
  <a href="#"><img alt="Language" src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge"></a>
</p>

---

## 🚀 About The Project

Pneumonia is a life-threatening lung infection that causes millions of deaths annually. Early and accurate diagnosis is critical for effective treatment, and medical imaging like chest X-rays plays a vital role. However, manual diagnosis can be time-consuming and prone to error.

This project implements a **Convolutional Neural Network (CNN)**, a deep learning model, to automate the detection of pneumonia from chest X-ray images. The model is trained to classify images as either **'Normal'** or **'Pneumonia'**, serving as a powerful tool to assist medical professionals.

---

## 💾 Dataset

This project utilizes the **"Chest X-Ray Images (Pneumonia)"** dataset, which is a popular, publicly available collection on Kaggle.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Content:** The dataset contains 5,863 chest X-ray images (JPEG) organized into two categories: `PNEUMONIA` and `NORMAL`.
* **Details:** The images are from retrospective cohorts of pediatric patients of one to five years old.

<p align="center">
  <img src="https://github.com/Atulya-Juyal/pneumonia-detection-deep-learning/blob/main/test_cases/n1.jpeg" width="600">
</p>

---

## 🤖 Model Architecture

The core of this project is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture is designed to effectively learn features from the X-ray images.

* The model consists of multiple **convolutional layers** (with `ReLU` activation) to extract features like edges and textures.
* **Max-pooling layers** are used to downsample the feature maps, reducing computational complexity and making the model more robust.
* **Data augmentation** techniques (e.g., rotation, zoom, flips) were applied during training to prevent overfitting.
* The final layers consist of **dense (fully connected) layers** that perform the final classification.

---

## 📈 Results & Performance

The model was trained and evaluated, achieving high performance metrics in distinguishing between healthy and infected lungs.

* **Training Accuracy:** ~97%
* **Validation Accuracy:** ~95%
* **Key Metrics:** High Precision and Recall, indicating a low rate of false positives and false negatives.


---

## 📦 Local Installation & Setup

To run this project on your local machine, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Atulya-Juyal/pneumonia-detection-deep-learning.git](https://github.com/Atulya-Juyal/pneumonia-detection-deep-learning.git)
    cd pneumonia-detection-deep-learning
    ```
2.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    *(First, ensure you have a `requirements.txt` file by running `pip freeze > requirements.txt` in your original project environment.)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset:** Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the project directory as per the paths used in the code.

5.  **Run the notebook:** Launch Jupyter Notebook and open the `.ipynb` file.
    ```bash
    jupyter notebook
    ```

---

> Made by Atulya Juyal
>
> Check out my linkedin profile : https://www.linkedin.com/in/atulya-juyal-86a1a528a/

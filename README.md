# 🧠 Computer Vision Lab 4 — Feature-Based Image Classification & Mean Shift Segmentation

## 📌 Overview

This repository presents a classical computer vision pipeline implementing:

- **Feature-based Image Classification using SIFT**
- **Unsupervised Image Segmentation using Mean Shift Clustering**

The project demonstrates traditional handcrafted feature extraction techniques and density-based clustering approaches, providing a foundational understanding of image representation and segmentation prior to deep learning-based pipelines.

The implementation is designed for reproducible experimentation using the **CIFAR-10 dataset**, executed within **Google Colab / Python environments**.

---

## 🎯 Research Motivation

Before the dominance of deep neural networks, feature engineering methods such as SIFT were widely used for visual recognition tasks. Understanding these approaches is critical for:

- Interpreting low-level visual features
- Understanding classical computer vision workflows
- Building explainable and lightweight vision systems
- Developing intuition about feature descriptors and clustering

This project revisits these foundational methods in a structured experimental setting.

---

## 🔬 Experimental Objectives

1. Extract invariant image descriptors using **Scale Invariant Feature Transform (SIFT)**.
2. Perform supervised image classification using feature vectors and a KNN classifier.
3. Implement **Mean Shift Clustering** for unsupervised image segmentation.
4. Evaluate classification performance and visualize segmentation outputs.

---

## 🧪 Methodology

### 🔹 Feature-Based Image Classification Pipeline

1. Load CIFAR-10 dataset.
2. Convert RGB images to grayscale.
3. Detect keypoints and extract SIFT descriptors.
4. Normalize feature representation.
5. Train K-Nearest Neighbor classifier.
6. Evaluate model performance using classification accuracy.

---

### 🔹 Mean Shift Image Segmentation Pipeline

1. Select representative input image.
2. Transform pixel space into feature vectors.
3. Apply Mean Shift clustering.
4. Assign labels to pixel regions.
5. Generate segmented visualization output.

---

## 📊 Dataset

**CIFAR-10 Dataset**

- Total Images: 60,000
- Classes: 10 object categories
- Image Size: 32×32
- Automatically downloaded during execution

Dataset Source: TensorFlow/Keras Dataset Loader

---

## 🛠️ Technology Stack

| Category              | Tools                 |
|-----------------------|-----------------------|
| Programming Language  | Python                |
| Computer Vision       | OpenCV                |
| Machine Learning      | Scikit-Learn          |
| Feature Extraction    | SIFT                  |
| Segmentation          | Mean Shift Clustering |
| Numerical Processing  | NumPy                 |
| Visualization         | Matplotlib            |
| Execution Environment | Google Colab          |

---

## 📂 Repository Structure

```
CV-Lab4/
│
├── notebooks/
│   └── CV_Lab4_Colab.ipynb
│
├── src/
│   ├── sift_classification.py
│   └── mean_shift_segmentation.py
│
├── results/
│   ├── classification_output.png
│   └── segmentation_output.png
│
├── report/
│   └── experiment_report.md
│
├── requirements.txt
└── README.md
```

```
CV-Lab4/
│
├── CV_LAB4.ipynb
│ 
│
├── CV_Lab_Report
│  
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Execution

### Run in Google Colab

1. Open notebook from `notebooks/`
2. Execute cells sequentially
3. Dataset downloads automatically

### Run Locally

```bash
python src/sift_classification.py
python src/mean_shift_segmentation.py
```

---

## 📈 Results & Observations

- SIFT descriptors successfully capture invariant local features.
- KNN classification demonstrates effectiveness of handcrafted feature representations.
- Mean Shift clustering produces density-based segmentation without predefined cluster count.
- Performance depends on descriptor richness and dataset complexity.

(Add quantitative accuracy values and screenshots in results folder.)

---

## 🧠 Learning Outcomes

- Understanding invariant feature extraction
- Classical machine learning pipelines for vision
- Density-based clustering in pixel space
- Bridging traditional CV and modern AI systems

---

## 🚀 Potential Extensions

- Replace KNN with SVM or Random Forest classifiers
- Compare SIFT with ORB or SURF descriptors
- Evaluate segmentation using color-space transformations
- Integrate deep CNN feature extraction for comparison
- Benchmark against deep learning baselines

---

## 👤 Author

Name: APARAJITA VAISH
Roll No: 253100101
Program: M.Tech. (ECE)  
Institute: IIITNR 

---

## 📜 License

This project is licensed under the MIT License.

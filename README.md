# Fingerprint Matching Script using Python & OpenCV

## Introduction
This project implements a **fingerprint matching system** using Python and OpenCV. It processes fingerprint images from the **SOCOFing dataset**, extracts key features, and compares them using feature matching techniques.

---

## 1️⃣ The SOCOFing Dataset
The **SOCOFing dataset** (Sokoto Coventry Fingerprint Dataset) is a publicly available dataset containing thousands of fingerprint images. It includes:
- **Real, Altered, and Obfuscated fingerprints**, allowing for robust testing.
- **Variations in orientation, pressure, and distortion**, making it a challenging dataset for fingerprint recognition tasks.
- A structured dataset with labels indicating finger type (e.g., left index, right thumb, etc.).

This dataset was selected because it provides **real-world variations** that allowed me to train and test the fingerprint-matching algorithm effectively.

---

## 2️⃣ The OpenCV Library
[OpenCV](https://opencv.org/) (Open Source Computer Vision Library) is a widely used library for image processing and computer vision. In this project, we used OpenCV to:

- **Read and preprocess images** (convert to grayscale, resize, and apply histogram equalization).
- **Detect keypoints** using the **SIFT (Scale-Invariant Feature Transform) algorithm**.
- **Compute descriptors** that represent unique fingerprint features.
- **Match fingerprints** using the **FLANN (Fast Library for Approximate Nearest Neighbors) matcher**.
- **Visualize results** by drawing matches between keypoints.

These OpenCV functions enabled me to develop an efficient and scalable fingerprint-matching pipeline.

---

## 3️⃣ Understanding Keypoints & match_points

### **Keypoints:**
- Keypoints represent **unique feature points** detected in an image.
- In the project, keypoints are extracted using the **SIFT algorithm**.
- Each fingerprint image contains hundreds of keypoints, which are used for comparison.

### **match_points:**
- `match_points` refers to the **filtered set of matches** that pass a quality threshold.
- After extracting keypoints from two images, I used **feature matching algorithms** to compare them.
- **filter matches** by distance to ensure **only the strongest feature correspondences are considered**.
- The quality of `match_points` determines the final fingerprint-matching score.

---

## 4️⃣ Code Refactoring & Optimizations

### **Key Improvements:**
1. **Strong Match Points Filtering:**
   - Initially, all keypoint matches were considered, leading to false positives.
   - I **filtered matches based on distance thresholds** to retain only strong matches.

2. **Precision & Recall Analysis:**
   - I implemented **precision and recall metrics** to evaluate match quality.
   - **Precision** measures the percentage of correct matches out of the total strong matches.
   - **Recall** measures how many true keypoints were successfully matched.
   - By tuning these parameters, I improved fingerprint recognition accuracy.

3. **High Scores Achieved:**
   - With our optimizations, I obtained **high matching scores** (e.g., **1330.0** for a left ring finger match).
   - Adjusting the **match distance threshold** and **strong match retention percentage** significantly improved results.

These refinements resulted in a **more accurate and efficient fingerprint-matching system**.

---

## 🚀 Conclusion
This project demonstrates the power of **OpenCV, feature extraction, and machine learning-based fingerprint matching**. By utilizing the **SOCOFing dataset** and optimizing **keypoint matching**, I developed an **accurate fingerprint recognition pipeline**. Further improvements could include **deep learning-based fingerprint classification** and **multi-fingerprint comparison methods**.

---

## 📌 How to Run the Program
1. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. View results, including **matched fingerprints** and **precision-recall analysis**.

---

## 🛠 Future Enhancements
- **Enhancing matching with deep learning** (CNN-based fingerprint recognition).
- **Speed optimizations** for large-scale fingerprint databases.
- **Better preprocessing techniques** to handle noisy or incomplete fingerprints.

---

## 📜 Credits
- **Dataset:** SOCOFing (Sokoto Coventry Fingerprint Dataset)
- **Libraries Used:** OpenCV, NumPy
- **Developed by:** Barra Harrison


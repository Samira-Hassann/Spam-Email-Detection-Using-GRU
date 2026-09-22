# 📩 Spam Message Detection using Word2Vec + GRU

A deep learning-based NLP project for classifying text messages as **Spam** or **Ham (Not Spam)** using a **Word2Vec-based embedding layer and GRU neural network**.

## 🚀 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, or other potentially harmful content.

This project builds an end-to-end spam classification pipeline:

**Text Preprocessing → Tokenization → Word2Vec Embeddings → GRU → Binary Classification**

The final model was selected after comparing it with a traditional **TF-IDF + Logistic Regression baseline**.

---
## 🚀 Project Links

| Resource | Link |
|----------|------|
| 📓 Kaggle Notebook | [View Notebook](https://www.kaggle.com/code/samoura/spam-detection) |
| 🌐 Live Demo | [Try the App](https://spam-email-detection-gru.streamlit.app/) |
---

## 🧠 Model Architecture

The final model consists of:

* **Embedding Layer**

  * Initialized using Word2Vec embeddings
  * Embedding dimension: `100`
  * Trainable during model training
  * `mask_zero=True`

* **GRU Layers**

  * GRU(128)
  * GRU(64)
  * GRU(32)
  * GRU(16)
  * GRU(8)

* **Fully Connected Layers**

  * Dense(64, ReLU)
  * Dropout(0.3)
  * Dense(1, Sigmoid)

The model uses **Binary Cross-Entropy** as the loss function and Adam as the optimizer.

---

## 📊 Dataset

The dataset contains labeled text messages belonging to two classes:

| Label | Meaning |
| ----- | ------- |
| `0`   | Ham     |
| `1`   | Spam    |

The data was split using a stratified train-test split:

* **80% Training**
* **20% Testing**
* `random_state = 42`
* Stratification was used to preserve the class distribution.

---

## ⚙️ Text Processing

The text processing pipeline includes:

1. Text cleaning
2. Tokenization
3. Conversion of words into integer IDs
4. Sequence padding
5. Word2Vec-based embedding representation
6. GRU sequence modeling

The maximum sequence length used by the final model is:

```text
MAX_LEN = 250
```

---

## ⚖️ Class Imbalance

The dataset contains considerably more Ham messages than Spam messages.

To reduce the effect of class imbalance during training, **class weights** were applied.

The calculated class weights were approximately:

```python
{
    0: 0.571,
    1: 4.020
}
```

This gives more importance to errors on Spam messages during training.

---

## 📈 Model Performance

### Baseline — TF-IDF + Logistic Regression

```text
Train Accuracy: 98.69%
Test Accuracy: 97.87%
```

Spam class:

| Metric    | Score |
| --------- | ----: |
| Precision |   93% |
| Recall    |   90% |
| F1-score  |   91% |

Confusion Matrix:

```text
[[895   9]
 [ 13 115]]
```

---

### 🏆 Final Model — Word2Vec + GRU

```text
Train Accuracy: 99.59%
Test Accuracy: 98.74%
```

Spam class:

| Metric    | Score |
| --------- | ----: |
| Precision |   99% |
| Recall    |   91% |
| F1-score  |   95% |

Confusion Matrix:

```text
[[903   1]
 [ 12 116]]
```

### 📌 Comparison

| Metric          | Baseline | Word2Vec + GRU |
| --------------- | -------: | -------------: |
| Train Accuracy  |   98.69% |     **99.59%** |
| Test Accuracy   |   97.87% |     **98.74%** |
| Spam Precision  |      93% |        **99%** |
| Spam Recall     |      90% |        **91%** |
| Spam F1         |      91% |        **95%** |
| False Positives |        9 |          **1** |
| False Negatives |       13 |         **12** |

The Word2Vec + GRU model was selected as the **final model** based on its performance on the unseen test set.

---

## 🎚️ Threshold Tuning

The default classification threshold of `0.5` was further investigated using the validation set.

Several thresholds were evaluated to study the trade-off between Precision and Recall.

The threshold that achieved the highest validation F1-score was:

```text
Best Threshold = 0.16
Best Validation F1 = 0.9528
```

Therefore, the deployed application uses:

```python
THRESHOLD = 0.16
```

The test set was kept separate during threshold selection and was used only for final model evaluation.

---

## 🌐 Deployment

The model is deployed using **Streamlit**.

The application allows users to:

1. Enter a text message.
2. Process the message using the same tokenizer used during training.
3. Generate a spam probability.
4. Classify the message as:

   * 🟢 **HAM**
   * 🔴 **SPAM**

### Example

```text
Message:
Congratulations! You have won a free prize. Call now!

Prediction:
🚨 SPAM

Spam Probability:
98.7%
```

---

## 📁 Repository Structure

```text
spam-gru-classifier/
│
├── app.py
├── model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File               | Description                      |
| ------------------ | -------------------------------- |
| `app.py`           | Streamlit deployment application |
| `model.keras`      | Trained Word2Vec + GRU model     |
| `tokenizer.pkl`    | Fitted text tokenizer            |
| `requirements.txt` | Python dependencies              |
| `.gitignore`       | Files excluded from Git          |

---

## 🛠️ Technologies

* Python
* TensorFlow / Keras
* Gensim
* NumPy
* Scikit-learn
* Streamlit
* NLP
* Word2Vec
* GRU

---

## ▶️ Run Locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 Project Goals

This project demonstrates an end-to-end NLP workflow including:

* Text preprocessing
* Tokenization
* Word embeddings
* Handling class imbalance
* Sequence modeling with GRU
* Model evaluation
* Threshold optimization
* Model deployment with Streamlit

---

## 👩‍💻 Author

**Samira Hassan**

Junior ML Engineer & Data Scientist

Interested in Machine Learning, Deep Learning, NLP, and Computer Vision.

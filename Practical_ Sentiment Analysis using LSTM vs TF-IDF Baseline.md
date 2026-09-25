# Practical: Sentiment Analysis using LSTM and Comparison with TF-IDF Baseline

## 1. Objective

Build a **sentiment analysis model** using an **LSTM (Long Short-Term Memory)** network and compare its performance with a traditional machine-learning baseline using **TF-IDF features**.

We will use the **IMDB Movie Reviews dataset**, where each review is classified as:

- `0` → Negative
- `1` → Positive

### Models to Compare

| Model | Feature Representation | Classifier |
|---|---|---|
| Baseline | TF-IDF | Logistic Regression |
| Deep Learning | Word sequences | LSTM |

---

# 2. Learning Outcomes

After completing this practical, you should be able to:

1. Load and preprocess a text sentiment dataset.
2. Convert text into TF-IDF features.
3. Build a traditional sentiment classifier.
4. Convert text into padded word sequences.
5. Build an LSTM-based sentiment classifier.
6. Evaluate both models using accuracy and classification metrics.
7. Compare traditional NLP approaches with sequence-based deep learning.

---

# 3. Dataset

We will use the **IMDB Movie Reviews dataset**.

The dataset contains:

- 50,000 movie reviews
- 25,000 training reviews
- 25,000 testing reviews
- Binary sentiment labels

### Example

```text
Review:
"This movie was absolutely fantastic. The acting was brilliant."

Label:
Positive → 1
```

```text
Review:
"The story was boring and the movie was a complete waste of time."

Label:
Negative → 0
```

---

# 4. Install and Import Libraries

### Code

```python
!pip install -q tensorflow scikit-learn pandas numpy matplotlib seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("TensorFlow version:", tf.__version__)
```

---

# 5. Load the IMDB Dataset

TensorFlow provides the IMDB dataset directly.

```python
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))
```

Expected output:

```text
Training samples: 25000
Testing samples: 25000
```

Here, the reviews are already represented as sequences of integer IDs.

For example:

```text
[1, 14, 22, 16, 43, 530, ...]
```

Each integer represents a word in the IMDB vocabulary.

---

# 6. Understand the Dataset

Let's examine the first review.

```python
print("First review:")
print(x_train[0])

print("\nFirst label:")
print(y_train[0])
```

Check the review length:

```python
print("Length of first review:", len(x_train[0]))
```

Let's calculate the average review length.

```python
train_lengths = [len(review) for review in x_train]

print("Average review length:",
      np.mean(train_lengths))

print("Minimum length:",
      np.min(train_lengths))

print("Maximum length:",
      np.max(train_lengths))
```

---

# 7. Convert IMDB Sequences Back to Text

For the TF-IDF model, we need the actual text.

TensorFlow provides the word-to-index dictionary.

```python
word_index = imdb.get_word_index()

reverse_word_index = {
    value: key
    for key, value in word_index.items()
}

def decode_review(review):
    return " ".join(
        reverse_word_index.get(i - 3, "?")
        for i in review
    )
```

Decode the first review:

```python
print(decode_review(x_train[0]))
```

The output will look something like:

```text
this film was just brilliant...
```

---

# 8. Prepare Text Data for the Baseline Model

Convert all reviews from integer sequences into text.

```python
train_text = [decode_review(review) for review in x_train]
test_text = [decode_review(review) for review in x_test]

print(train_text[0])
```

---

# PART A — Baseline Model

# 9. TF-IDF Representation

Before using LSTM, we will create a traditional machine-learning baseline.

### What is TF-IDF?

TF-IDF represents a document using numerical features based on how important each word is.

The basic idea is:

\[
TF-IDF(t,d) = TF(t,d) \times IDF(t)
\]

where:

- `TF` = Term Frequency
- `IDF` = Inverse Document Frequency

Common words occurring in many documents receive lower importance, while informative words receive higher importance.

---

# 10. Create TF-IDF Features

We will limit the vocabulary to 10,000 features.

```python
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(train_text)

X_test_tfidf = tfidf.transform(test_text)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)
```

Expected shape:

```text
Training feature shape: (25000, 10000)
Testing feature shape: (25000, 10000)
```

---

# 11. Train Logistic Regression

We will use Logistic Regression as our baseline classifier.

```python
baseline_model = LogisticRegression(
    max_iter=1000
)

baseline_model.fit(
    X_train_tfidf,
    y_train
)
```

---

# 12. Make Predictions

```python
y_pred_baseline = baseline_model.predict(
    X_test_tfidf
)
```

Calculate accuracy:

```python
baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

print("TF-IDF + Logistic Regression Accuracy:",
      baseline_accuracy)
```

---

# 13. Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_baseline,
        target_names=["Negative", "Positive"]
    )
)
```

Observe:

- Precision
- Recall
- F1-score
- Accuracy

---

# 14. Confusion Matrix

```python
cm_baseline = confusion_matrix(
    y_test,
    y_pred_baseline
)

sns.heatmap(
    cm_baseline,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TF-IDF + Logistic Regression")
plt.show()
```

---

# PART B — LSTM MODEL

# 15. Why LSTM?

TF-IDF treats a review mainly as a collection of important words.

For example:

```text
"The movie was not good"
```

The words are represented individually.

However, the order matters.

Compare:

```text
"The movie was good."
```

and

```text
"The movie was not good."
```

The word `good` occurs in both sentences, but the word `not` changes the meaning.

LSTM processes the sequence of words and can learn relationships between words across the sequence.

---

# 16. Prepare Data for LSTM

We will use:

```text
Vocabulary size = 10,000
Maximum sequence length = 200
```

```python
num_words = 10000
max_length = 200
```

The IMDB dataset already contains integer sequences, so we only need to pad them.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train_lstm = pad_sequences(
    x_train,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

X_test_lstm = pad_sequences(
    x_test,
    maxlen=max_length,
    padding='post',
    truncating='post'
)

print("Training shape:", X_train_lstm.shape)
print("Testing shape:", X_test_lstm.shape)
```

Expected:

```text
Training shape: (25000, 200)
Testing shape: (25000, 200)
```

---

# 17. Understand Padding

Suppose the maximum sequence length is 8.

Original review:

```text
[12, 25, 78, 43]
```

After padding:

```text
[12, 25, 78, 43, 0, 0, 0, 0]
```

All reviews therefore have the same length.

This is necessary because neural networks process batches of tensors with compatible dimensions.

---

# 18. Build the LSTM Architecture

Our architecture will be:

```text
Input
  ↓
Embedding
  ↓
LSTM
  ↓
Dense
  ↓
Sigmoid
  ↓
Sentiment
```

### Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

```python
lstm_model = Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=128,
        input_length=max_length
    ),

    LSTM(128),

    Dense(1, activation='sigmoid')
])
```

Display the architecture:

```python
lstm_model.summary()
```

---

# 19. Understand the Architecture

## Embedding Layer

The embedding layer converts each word ID into a dense vector.

For example:

```text
word ID = 25
```

may become:

```text
[0.12, -0.43, 0.78, ...]
```

with 128 dimensions.

Therefore:

```text
Word ID
   ↓
128-dimensional vector
```

---

## LSTM Layer

The LSTM receives the word vectors sequentially.

Conceptually:

```text
Word 1 → LSTM
           ↓
Word 2 → LSTM
           ↓
Word 3 → LSTM
           ↓
...
           ↓
Final representation
```

The LSTM maintains information using its internal gates and hidden state.

---

## Dense Layer

The final Dense layer produces one value:

```text
0 → Negative
1 → Positive
```

Because we use sigmoid:

\[
P(y=1)=\sigma(z)
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

---

# 20. Compile the Model

```python
lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Why binary cross-entropy?

Because this is a binary classification problem.

\[
L =
-[y\log(\hat{y})+(1-y)\log(1-\hat{y})]
\]

---

# 21. Train the LSTM

```python
history = lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```

Observe the following during training:

```text
loss
accuracy
val_loss
val_accuracy
```

---

# 22. Plot Training Performance

## Accuracy

```python
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Training vs Validation Accuracy")
plt.legend()

plt.show()
```

---

## Loss

```python
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training vs Validation Loss")
plt.legend()

plt.show()
```

### Observe

If training accuracy continues increasing while validation accuracy stops improving or decreases, the model may be **overfitting**.

---

# 23. Evaluate LSTM

```python
lstm_loss, lstm_accuracy = lstm_model.evaluate(
    X_test_lstm,
    y_test,
    verbose=0
)

print("LSTM Test Accuracy:", lstm_accuracy)
```

---

# 24. Generate LSTM Predictions

```python
y_prob_lstm = lstm_model.predict(
    X_test_lstm
)

y_pred_lstm = (
    y_prob_lstm >= 0.5
).astype(int).flatten()
```

---

# 25. LSTM Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_lstm,
        target_names=["Negative", "Positive"]
    )
)
```

---

# 26. LSTM Confusion Matrix

```python
cm_lstm = confusion_matrix(
    y_test,
    y_pred_lstm
)

sns.heatmap(
    cm_lstm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Confusion Matrix")

plt.show()
```

---

# PART C — MODEL COMPARISON

# 27. Compare Both Models

```python
results = pd.DataFrame({
    "Model": [
        "TF-IDF + Logistic Regression",
        "Embedding + LSTM"
    ],
    "Accuracy": [
        baseline_accuracy,
        lstm_accuracy
    ]
})

results
```

Visualize:

```python
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Sentiment Analysis Model Comparison")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.show()
```

---

# 28. Compare the Models

Fill the following table using your actual experimental results.

| Property | TF-IDF + Logistic Regression | Embedding + LSTM |
|---|---|---|
| Feature representation | TF-IDF | Word Embeddings |
| Uses word order | Limited | Yes |
| Sequence information | No | Yes |
| Training time | Lower | Higher |
| Computational requirement | Lower | Higher |
| Model type | Traditional ML | Deep Learning |
| Accuracy | ______ | ______ |
| F1-score | ______ | ______ |

---

# 29. Test the Model on Your Own Reviews

Create some reviews manually.

```python
custom_reviews = [
    "This movie was absolutely fantastic and I loved every moment.",
    "The movie was boring and extremely disappointing.",
    "The acting was good but the story was terrible.",
    "I really enjoyed this movie. It was entertaining and emotional."
]
```

---

# 30. Convert Custom Reviews to Sequences

We need to convert words into IMDB word IDs.

```python
def encode_custom_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        encoded.append(
            word_index.get(word, 2)
        )

    return encoded
```

Convert the reviews:

```python
custom_sequences = [
    encode_custom_review(review)
    for review in custom_reviews
]
```

Pad them:

```python
custom_padded = pad_sequences(
    custom_sequences,
    maxlen=max_length,
    padding='post',
    truncating='post'
)
```

---

# 31. Predict Sentiment

```python
predictions = lstm_model.predict(
    custom_padded
)
```

Display results:

```python
for review, prediction in zip(
    custom_reviews,
    predictions
):

    sentiment = (
        "Positive"
        if prediction[0] >= 0.5
        else "Negative"
    )

    print("Review:", review)
    print("Predicted sentiment:", sentiment)
    print("Probability:", float(prediction[0]))
    print("-" * 70)
```

---

# 32. Important Experiment — Effect of Sequence Length

Now investigate whether sequence length affects performance.

Train/evaluate models using:

```text
max_length = 100
max_length = 200
max_length = 300
```

Create a table:

| Maximum Sequence Length | Accuracy |
|---:|---:|
| 100 | ______ |
| 200 | ______ |
| 300 | ______ |

### Question

Why might increasing sequence length improve performance initially but eventually increase computational cost?

---

# 33. Important Experiment — Effect of LSTM Units

Try:

```python
LSTM(64)
```

Then:

```python
LSTM(128)
```

Then:

```python
LSTM(256)
```

Record the results.

| LSTM Units | Accuracy | Training Time |
|---:|---:|---:|
| 64 | ______ | ______ |
| 128 | ______ | ______ |
| 256 | ______ | ______ |

### Question

Does increasing the number of LSTM units always guarantee better performance?

---

# 34. Viva Questions

### Q1. What is sentiment analysis?

Sentiment analysis is the task of determining the emotional polarity of text, such as positive or negative.

---

### Q2. Why is TF-IDF used as a baseline?

TF-IDF provides a simple and strong traditional representation of text that can be combined with classifiers such as Logistic Regression.

---

### Q3. What is the major limitation of TF-IDF?

TF-IDF does not naturally represent the sequential order of words.

For example:

```text
"I like this movie"
```

and

```text
"I do not like this movie"
```

contain overlapping words, but their meanings differ.

---

### Q4. Why do we use an Embedding layer?

The Embedding layer converts discrete word IDs into dense numerical vectors that can be learned during training.

---

### Q5. Why is LSTM suitable for sentiment analysis?

LSTM is designed to process sequential data and maintain information over multiple time steps.

---

### Q6. What is the role of the forget gate?

The forget gate determines which information from the previous cell state should be discarded.

\[
f_t =
\sigma(W_f[h_{t-1},x_t]+b_f)
\]

---

### Q7. Why do we use sigmoid in the final layer?

Because this is a binary classification problem.

The sigmoid produces a value between 0 and 1:

\[
0 \leq \hat{y} \leq 1
\]

---

### Q8. Why do we use binary cross-entropy?

Because the target contains two classes:

```text
0 → Negative
1 → Positive
```

---

### Q9. What does padding do?

Padding makes sequences have a common length so that they can be processed together in batches.

---

### Q10. What is overfitting?

Overfitting occurs when the model performs very well on training data but performs relatively poorly on unseen data.

---

# 35. Discussion Questions

Answer the following after completing the experiment.

### Question 1

Which model achieved higher test accuracy in your experiment?

```text
Answer:
____________________________________________________
____________________________________________________
```

### Question 2

Why might the LSTM model be able to capture information that TF-IDF cannot?

```text
Answer:
____________________________________________________
____________________________________________________
```

### Question 3

What happens if the maximum sequence length is too small?

```text
Answer:
____________________________________________________
____________________________________________________
```

### Question 4

What happens if the maximum sequence length is extremely large?

```text
Answer:
____________________________________________________
____________________________________________________
```

### Question 5

Does higher accuracy necessarily mean that a model is always better for every application? Explain.

```text
Answer:
____________________________________________________
____________________________________________________
```

---

# 36. Extension Activity

Modify the LSTM model by adding:

```text
Embedding
    ↓
LSTM
    ↓
Dropout
    ↓
Dense
    ↓
Sigmoid
```

Example:

```python
from tensorflow.keras.layers import Dropout

improved_model = Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=128,
        input_length=max_length
    ),

    LSTM(128),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])
```

Compile and train the model.

```python
improved_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Compare it with the original LSTM.

---

# 37. Final Submission

Your practical submission should contain:

1. Dataset loading
2. Data exploration
3. TF-IDF feature extraction
4. Logistic Regression baseline
5. Baseline evaluation
6. LSTM preprocessing
7. LSTM architecture
8. LSTM training
9. LSTM evaluation
10. Confusion matrices
11. Accuracy comparison
12. Custom review predictions
13. Experimental observations
14. Answers to discussion questions

---

# 38. Expected Workflow

```text
                 IMDB Dataset
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       TF-IDF Features     Word Sequences
             │                 │
             ▼                 ▼
     Logistic Regression    Padding
             │                 │
             ▼                 ▼
       Predictions          Embedding
             │                 │
             │                 ▼
             │                LSTM
             │                 │
             │                 ▼
             │               Dense
             │                 │
             └────────┬────────┘
                      ▼
              Model Comparison
                      │
                      ▼
             Accuracy / F1-score
```

---

# 39. Key Takeaway

Traditional NLP:

```text
Text
 ↓
TF-IDF
 ↓
Machine Learning Classifier
 ↓
Prediction
```

Sequence-based Deep Learning:

```text
Text
 ↓
Tokenization
 ↓
Embedding
 ↓
LSTM
 ↓
Prediction
```

The important conceptual difference is:

> **TF-IDF represents the importance of words, while LSTM processes the sequence of words and can learn information from their context and order.**

The objective of this practical is therefore not simply to find which model has the highest accuracy, but to understand how the **representation of text influences the type of information a model can learn**.
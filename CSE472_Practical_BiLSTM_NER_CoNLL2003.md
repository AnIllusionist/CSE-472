# CSE472 Practical: Bidirectional LSTM for Named Entity Recognition

## Practical Title
**Implement a Bidirectional LSTM model for Named Entity Recognition (NER) using the CoNLL-2003 dataset**

### Goal
Train a **BiLSTM sequence-labeling model** to assign one tag to every token in a sentence. For this practical, the target entity types are:

- **Person**
- **Location**
- **Organization**
- **Other**

CoNLL-2003 contains BIO-style labels. We map:

- `PER` → Person
- `LOC` → Location
- `ORG` → Organization
- `MISC` → Other
- `O` → Other

We keep the BIO prefixes during training (`B-Person`, `I-Person`, etc.) because entity boundaries matter.

---

## 1. Learning Objectives

By the end of the practical, students should be able to:

1. Explain NER as a **sequence-labeling** problem.
2. Load and inspect the CoNLL-2003 dataset.
3. Convert tokens and NER tags into integer sequences.
4. Pad variable-length sequences.
5. Build a **Bidirectional LSTM** with Keras.
6. Generate one prediction for every token.
7. Evaluate token-level and entity-level performance.
8. Compare a unidirectional LSTM with a BiLSTM.

---

## 2. Why BiLSTM for NER?

Consider:

> `Steve Jobs founded Apple in California`

NER labels might be:

```text
Steve       B-Person
Jobs        I-Person
founded     O
Apple       B-Organization
in          O
California  B-Location
```

A token can depend on context on **both sides**. A BiLSTM runs one LSTM from left-to-right and another from right-to-left, then combines their hidden states.

```text
Forward:   token1 → token2 → token3 → ... → tokenT
Backward:  token1 ← token2 ← token3 ← ... ← tokenT
                         ↓
                contextual representation
                         ↓
                     NER tag
```

---

## 3. Architecture

```text
Token IDs
   ↓
Embedding
   ↓
Bidirectional LSTM
   ↓
Dense + Softmax
   ↓
Tag for every token
```

For tokens `x1 ... xT`:

```text
x1, x2, x3, ..., xT
        ↓
   BiLSTM outputs
h1, h2, h3, ..., hT
        ↓
Softmax for each position
        ↓
y1, y2, y3, ..., yT
```

This is a **many-to-many sequence labeling** problem.

---

## 4. Install and Import Libraries

### Colab cell

```python
!pip -q install datasets seqeval
```

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import load_dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)
```

---

## 5. Load CoNLL-2003

```python
dataset = load_dataset("conll2003")
print(dataset)
```

Inspect the splits:

```python
print(dataset["train"])
print(dataset["validation"])
print(dataset["test"])
```

Inspect one sample:

```python
example = dataset["train"][0]
print(example)
```

The important fields are:

- `tokens`
- `ner_tags`

---

## 6. Inspect the Original NER Labels

```python
label_names = dataset["train"].features["ner_tags"].feature.names

print(label_names)
```

Create mappings:

```python
id_to_tag = {i: tag for i, tag in enumerate(label_names)}
tag_to_id = {tag: i for i, tag in enumerate(label_names)}

print(id_to_tag)
```

Typical tags are:

```text
O
B-PER   I-PER
B-LOC   I-LOC
B-ORG   I-ORG
B-MISC  I-MISC
```

---

## 7. Use a Smaller Subset for the Classroom

Start with a smaller subset so students can iterate quickly.

```python
TRAIN_SIZE = 5000
VAL_SIZE = 1000
TEST_SIZE = 1000

train_data = dataset["train"].select(range(TRAIN_SIZE))
val_data = dataset["validation"].select(range(VAL_SIZE))
test_data = dataset["test"].select(range(TEST_SIZE))

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))
```

Later, remove `.select(...)` to train on the full dataset.

---

## 8. Map CoNLL Labels to the Required Classes

```python
def simplify_tag(tag):
    if tag == "O":
        return "O"

    prefix, entity = tag.split("-", 1)

    if entity == "PER":
        return prefix + "-Person"
    elif entity == "LOC":
        return prefix + "-Location"
    elif entity == "ORG":
        return prefix + "-Organization"
    else:
        return "O"
```

Test it:

```python
test_tags = [
    "O", "B-PER", "I-PER", "B-LOC", "I-LOC",
    "B-ORG", "I-ORG", "B-MISC", "I-MISC"
]

for tag in test_tags:
    print(tag, "->", simplify_tag(tag))
```

Expected idea:

```text
B-PER   -> B-Person
I-PER   -> I-Person
B-LOC   -> B-Location
I-LOC   -> I-Location
B-ORG   -> B-Organization
I-ORG   -> I-Organization
B-MISC  -> O
I-MISC  -> O
```

---

## 9. Create the Tag Vocabulary

`PAD` is used only for padded positions.

```python
all_tags = [
    "PAD",
    "O",
    "B-Person", "I-Person",
    "B-Location", "I-Location",
    "B-Organization", "I-Organization"
]

tag_to_index = {tag: i for i, tag in enumerate(all_tags)}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}

print(tag_to_index)
```

---

## 10. Build the Word Vocabulary

```python
word_to_index = {
    "<PAD>": 0,
    "<UNK>": 1
}

for example in train_data:
    for token in example["tokens"]:
        token = token.lower()
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)

index_to_word = {i: w for w, i in word_to_index.items()}

print("Vocabulary size:", len(word_to_index))
```

`<UNK>` handles words that occur in validation/test but are not in the training vocabulary.

---

## 11. Encode One Example

```python
def encode_example(example):
    tokens = [token.lower() for token in example["tokens"]]

    original_tags = [id_to_tag[i] for i in example["ner_tags"]]
    tags = [simplify_tag(tag) for tag in original_tags]

    x = [word_to_index.get(token, word_to_index["<UNK>"]) for token in tokens]
    y = [tag_to_index[tag] for tag in tags]

    return x, y, tokens, tags
```

Check one sentence:

```python
x, y, tokens, tags = encode_example(train_data[0])

for token, tag in zip(tokens, tags):
    print(f"{token:20s} {tag}")
```

---

## 12. Encode Complete Splits

```python
def encode_dataset(data):
    X, Y = [], []
    token_lists, tag_lists = [], []

    for example in data:
        x, y, tokens, tags = encode_example(example)
        X.append(x)
        Y.append(y)
        token_lists.append(tokens)
        tag_lists.append(tags)

    return X, Y, token_lists, tag_lists
```

```python
X_train, y_train, train_tokens, train_tags = encode_dataset(train_data)
X_val, y_val, val_tokens, val_tags = encode_dataset(val_data)
X_test, y_test, test_tokens, test_tags = encode_dataset(test_data)

print(len(X_train), len(X_val), len(X_test))
```

---

## 13. Padding

Sentences have different lengths, so we use a fixed maximum length.

```python
lengths = [len(x) for x in X_train]
print("Maximum sentence length:", max(lengths))
print("Average sentence length:", np.mean(lengths))
```

For a classroom run:

```python
MAX_LEN = 100
```

Pad inputs:

```python
X_train_pad = pad_sequences(
    X_train, maxlen=MAX_LEN, padding="post", truncating="post", value=0
)

X_val_pad = pad_sequences(
    X_val, maxlen=MAX_LEN, padding="post", truncating="post", value=0
)

X_test_pad = pad_sequences(
    X_test, maxlen=MAX_LEN, padding="post", truncating="post", value=0
)
```

Pad targets:

```python
PAD_TAG = tag_to_index["PAD"]


y_train_pad = pad_sequences(
    y_train, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_TAG
)

y_val_pad = pad_sequences(
    y_val, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_TAG
)

y_test_pad = pad_sequences(
    y_test, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_TAG
)

print("X_train:", X_train_pad.shape)
print("y_train:", y_train_pad.shape)
```

---

## 14. Ignore Padding During Training

Padding must not contribute to the loss.

```python
sample_weight_train = (y_train_pad != PAD_TAG).astype("float32")
sample_weight_val = (y_val_pad != PAD_TAG).astype("float32")
```

The mask has:

- `1` for real tokens
- `0` for padding

---

## 15. Build the BiLSTM Model

```python
VOCAB_SIZE = len(word_to_index)
NUM_TAGS = len(tag_to_index)
EMBEDDING_DIM = 100
LSTM_UNITS = 64

inputs = Input(shape=(MAX_LEN,))

embedding = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM,
    mask_zero=True
)(inputs)

bilstm = Bidirectional(
    LSTM(LSTM_UNITS, return_sequences=True)
)(embedding)

outputs = Dense(
    NUM_TAGS,
    activation="softmax"
)(bilstm)

model = Model(inputs, outputs)
model.summary()
```

### Why `return_sequences=True`?

NER needs **one output per token**.

```text
Sentence
   ↓
BiLSTM
   ↓
h1 h2 h3 ... hT
   ↓
Tag1 Tag2 Tag3 ... TagT
```

If `return_sequences=False` were used, the LSTM would return only one final representation for the whole sentence.

---

## 16. Compile the Model

```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
```

---

## 17. Train the Model

```python
EPOCHS = 5
BATCH_SIZE = 32

history = model.fit(
    X_train_pad,
    y_train_pad,
    validation_data=(X_val_pad, y_val_pad, sample_weight_val),
    sample_weight=sample_weight_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)
```

---

## 18. Plot Training Curves

### Accuracy

```python
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("BiLSTM NER Accuracy")
plt.legend()
plt.show()
```

### Loss

```python
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BiLSTM NER Loss")
plt.legend()
plt.show()
```

---

## 19. Predict on the Test Set

```python
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=-1)

print(y_pred.shape)
```

---

## 20. Token-Level Evaluation

Flatten only the real tokens; ignore `PAD`.

```python
y_true_flat = []
y_pred_flat = []

for i in range(len(y_test_pad)):
    for j in range(MAX_LEN):
        true_id = y_test_pad[i, j]

        if true_id == PAD_TAG:
            continue

        y_true_flat.append(index_to_tag[true_id])
        y_pred_flat.append(index_to_tag[y_pred[i, j]])
```

```python
labels_for_report = [
    "O",
    "B-Person", "I-Person",
    "B-Location", "I-Location",
    "B-Organization", "I-Organization"
]

print(classification_report(
    y_true_flat,
    y_pred_flat,
    labels=labels_for_report,
    zero_division=0
))
```

---

## 21. Entity-Level F1 with Seqeval

Token accuracy alone is not enough for NER because entity boundaries matter.

```python
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import precision_score, recall_score, f1_score
```

Create sentence-level sequences:

```python
true_sequences = []
pred_sequences = []

for i in range(len(y_test_pad)):
    true_sentence = []
    pred_sentence = []

    for j in range(MAX_LEN):
        true_id = y_test_pad[i, j]

        if true_id == PAD_TAG:
            continue

        true_sentence.append(index_to_tag[true_id])
        pred_sentence.append(index_to_tag[y_pred[i, j]])

    true_sequences.append(true_sentence)
    pred_sequences.append(pred_sentence)
```

Evaluate:

```python
print(seqeval_report(true_sequences, pred_sequences))

print("Entity Precision:", precision_score(true_sequences, pred_sequences))
print("Entity Recall:", recall_score(true_sequences, pred_sequences))
print("Entity F1:", f1_score(true_sequences, pred_sequences))
```

### Important distinction

**Token-level metrics:** Was each individual token labeled correctly?

**Entity-level F1:** Was the complete entity span identified correctly?

---

## 22. Predict Entities for a New Sentence

```python
def predict_entities(sentence):
    tokens = sentence.split()

    token_ids = [
        word_to_index.get(token.lower(), word_to_index["<UNK>"])
        for token in tokens
    ]

    padded = pad_sequences(
        [token_ids],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
        value=0
    )

    probs = model.predict(padded, verbose=0)
    pred_ids = np.argmax(probs, axis=-1)[0]

    results = []
    for token, pred_id in zip(tokens[:MAX_LEN], pred_ids[:len(tokens)]):
        results.append((token, index_to_tag[pred_id]))

    return results
```

Try:

```python
sentence = "Barack Obama visited New York and met Microsoft executives"

for token, tag in predict_entities(sentence):
    print(f"{token:15s} -> {tag}")
```

---

## 23. Display Predictions as a Table

```python
results = predict_entities(sentence)

results_df = pd.DataFrame(
    results,
    columns=["Token", "Predicted Tag"]
)

display(results_df)
```

---

## 24. Compare Actual vs Predicted on a Test Sentence

```python
sample_id = 5

sentence = " ".join(test_tokens[sample_id])
predictions = predict_entities(sentence)
predicted_tags = [tag for _, tag in predictions]

comparison = pd.DataFrame({
    "Token": test_tokens[sample_id][:len(predicted_tags)],
    "Actual": test_tags[sample_id][:len(predicted_tags)],
    "Predicted": predicted_tags
})

display(comparison)
```

Discuss:

- Which entities were correctly identified?
- Which tokens were misclassified?
- Were entity boundaries correct?

---

## 25. Mini Experiment: LSTM vs BiLSTM

Replace:

```python
bilstm = Bidirectional(
    LSTM(LSTM_UNITS, return_sequences=True)
)(embedding)
```

with:

```python
bilstm = LSTM(
    LSTM_UNITS,
    return_sequences=True
)(embedding)
```

Train again and compare:

| Model | Validation Accuracy | Entity F1 | Training Time |
|---|---:|---:|---:|
| LSTM | — | — | — |
| BiLSTM | — | — | — |

### Discussion

Why might the BiLSTM help?

Because each token representation can use:

```text
Left context + Token + Right context
```

rather than only left context.

---

## 26. Mini Experiment: Change Hidden Size

Try:

```python
LSTM_UNITS = 32
```

and then:

```python
LSTM_UNITS = 64
```

Compare:

- Training accuracy
- Validation accuracy
- Entity F1
- Training time

Question:

> Does increasing hidden size always improve generalization?

---

## 27. Mini Experiment: Change Maximum Sequence Length

Try:

```python
MAX_LEN = 50
```

and then:

```python
MAX_LEN = 100
```

Discuss the trade-offs:

- Truncation can remove tokens.
- Larger sequences require more computation.
- More padding increases wasted computation.

---

## 28. Common Errors

### Error 1 — Shape mismatch

Check:

```python
print(X_train_pad.shape)
print(y_train_pad.shape)
```

Both should be shaped approximately as:

```text
(number_of_sentences, MAX_LEN)
```

### Error 2 — `return_sequences=False`

For NER this is incorrect because we need one output per token.

### Error 3 — Counting padding in evaluation

Always ignore positions where:

```python
true_id == PAD_TAG
```

### Error 4 — Treating NER as sentence classification

NER is:

```text
Token1 → Label1
Token2 → Label2
...
TokenT → LabelT
```

not:

```text
Sentence → One label
```

---

## 29. Viva / Discussion Questions

1. What is Named Entity Recognition?
2. Why is NER a sequence-labeling problem?
3. What does `B-PER` mean?
4. What does `I-ORG` mean?
5. Why do we need word embeddings?
6. What does bidirectional processing add?
7. Why is `return_sequences=True` required?
8. Why is softmax used at every time step?
9. Why must padding be excluded from the loss/evaluation?
10. What is the difference between token accuracy and entity-level F1?
11. Why might BiLSTM outperform a unidirectional LSTM for offline NER?
12. Why is a BiLSTM less suitable for strict real-time streaming?
13. What happens when an unseen word appears at test time?
14. What is the purpose of the `<UNK>` token?
15. What would change if you used a BiGRU instead of a BiLSTM?

---

## 30. Final Student Challenge

Test the trained model on a sentence containing at least:

- one person
- one location
- one organization

Example:

```text
Elon Musk visited London and met Microsoft executives.
```

Generate token-level predictions and discuss:

1. Which entities were detected correctly?
2. Which entities were missed?
3. Were BIO boundaries correct?
4. Why might the model have made those mistakes?

---

# Final Takeaway

The complete practical pipeline is:

```text
CoNLL-2003
    ↓
Tokens + BIO Tags
    ↓
Vocabulary + Integer Encoding
    ↓
Padding
    ↓
Embedding
    ↓
Bidirectional LSTM
    ↓
Dense + Softmax
    ↓
One NER prediction per token
    ↓
Token-level + Entity-level Evaluation
```

> **Key idea:** A BiLSTM reads the complete sequence in both directions and creates a context-rich representation for each token, making it useful for sequence-labeling tasks such as Named Entity Recognition.

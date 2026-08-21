# NLP Practical Class: Word Embeddings and Vector Representations

**Course:** CSE472 — Deep Learning for Natural Language Processing  
**Duration:** Approximately 2 hours

## 1. Practical Overview

This practical moves from the previous text representations (BoW and TF-IDF) to dense word embeddings.

```text
Words
  ↓
Pretrained Word Embeddings
  ↓
Dense Vectors
  ↓
Cosine Similarity
  ↓
Word Relationships
  ↓
PCA / t-SNE
  ↓
2-D Visualization
```

The central question is:

> If TF-IDF represents words numerically, how can we represent relationships between words such as `car` and `automobile`?

## 2. Learning Outcomes

Students will be able to:

1. Explain sparse vs dense representations.
2. Load pretrained Word2Vec or GloVe embeddings.
3. Extract word vectors and identify embedding dimensionality.
4. Calculate cosine similarity.
5. Find similar words.
6. Explore word analogies.
7. Visualize embeddings using PCA.
8. Optionally visualize embeddings using t-SNE.
9. Interpret semantic neighborhoods.
10. Discuss limitations of static word embeddings.

## 3. Prerequisites

Students should know:

- Python
- NumPy
- Matplotlib
- Basic linear algebra
- BoW and TF-IDF
- Dot product and vector norm

## 4. Installation

```bash
pip install numpy pandas matplotlib scikit-learn gensim
```

---

# Part A — Why Word Embeddings?

Consider:

```text
car
automobile
banana
```

A one-hot representation might be:

```text
car        → [1, 0, 0]
automobile → [0, 1, 0]
banana     → [0, 0, 1]
```

Ask:

> Which two words should have similar representations?

Expected:

`car` and `automobile`.

But one-hot vectors do not naturally express that relationship.

### Discussion

Why might a learned dense representation be more useful?

---

# Part B — Load a Pretrained Word2Vec Model

## Task 1: Load Word2Vec

If the instructor provides a pretrained model:

```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)
```

Check:

```python
print("Vocabulary size:", len(model.key_to_index))
print("Embedding dimension:", model.vector_size)
```

> The pretrained model file should be supplied separately because it can be very large.

If the instructor provides GloVe vectors, use the provided loading method.

---

# Part C — Inspect Word Vectors

## Task 2

```python
words = [
    "cat", "dog", "kitten", "puppy",
    "car", "automobile", "banana", "computer"
]

for word in words:
    if word in model:
        print(word, "→ found")
    else:
        print(word, "→ not found")
```

Extract vectors:

```python
for word in words:
    if word in model:
        vector = model[word]
        print("\nWord:", word)
        print("Dimension:", len(vector))
        print("First 10 values:", vector[:10])
```

### Observe

A vector may look like:

```text
cat →
[0.12, -0.41, 0.73, 0.18, ...]
```

Unlike one-hot vectors, embeddings are dense.

### Question

> Does dimension 1 necessarily mean "animal" and dimension 2 necessarily mean "size"?

No. Individual dimensions are generally not directly interpretable.

---

# Part D — Cosine Similarity

## Task 3

Cosine similarity:

\[
\cos(	heta)=rac{A\cdot B}{||A||||B||}
\]

Code:

```python
from sklearn.metrics.pairwise import cosine_similarity

word1 = "cat"
word2 = "kitten"

v1 = model[word1].reshape(1, -1)
v2 = model[word2].reshape(1, -1)

score = cosine_similarity(v1, v2)[0][0]

print(word1, word2)
print("Cosine similarity:", score)
```

## Compare multiple pairs

```python
pairs = [
    ("cat", "kitten"),
    ("dog", "puppy"),
    ("car", "automobile"),
    ("cat", "banana"),
    ("computer", "banana")
]

for w1, w2 in pairs:
    if w1 in model and w2 in model:
        v1 = model[w1].reshape(1, -1)
        v2 = model[w2].reshape(1, -1)
        score = cosine_similarity(v1, v2)[0][0]
        print(f"{w1:12s} {w2:12s} → {score:.4f}")
```

### Questions

1. Which pair is most similar?
2. Is `car` close to `automobile`?
3. Is `cat` close to `banana`?
4. Does high similarity necessarily mean synonymy?
5. Can two words be related without being synonyms?

---

# Part E — Find Similar Words

## Task 4

```python
for target in ["cat", "dog", "car", "computer"]:
    print("\nWords similar to:", target)
    for word, score in model.most_similar(target, topn=10):
        print(f"{word:20s} {score:.4f}")
```

Classify the results as:

- synonym
- related concept
- same topic
- same context
- surprising

### Discussion

> Does `most_similar()` mean "find synonyms"?

No. Similarity can reflect several types of relatedness.

---

# Part F — Word Analogies

## Task 5

Try:

```python
result = model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)

for word, score in result:
    print(word, score)
```

Conceptually:

\[
king-man+womanpprox queen
\]

### Questions

1. What relationship is being represented?
2. Was the model explicitly taught the definition of `queen`?
3. Why can vector arithmetic produce relationships?
4. Does this prove human-like understanding?

Important:

> **Statistical pattern ≠ human-like understanding.**

---

# Part G — Your Own Analogy

Try another relationship:

```python
model.most_similar(
    positive=["paris", "italy"],
    negative=["france"],
    topn=5
)
```

Try your own example.

Record:

- analogy
- top result
- similarity score
- whether it makes sense
- possible reason for success/failure

Do not assume every analogy will work.

---

# Part H — Prepare Embeddings for Visualization

## Task 6

```python
selected_words = [
    "cat", "dog", "kitten", "puppy",
    "car", "automobile", "vehicle", "truck",
    "computer", "laptop", "software", "internet",
    "banana", "apple", "orange"
]

valid_words = [
    word for word in selected_words
    if word in model
]
```

Extract vectors:

```python
import numpy as np

vectors = np.array([
    model[word]
    for word in valid_words
])

print("Number of words:", len(valid_words))
print("Embedding matrix shape:", vectors.shape)
```

If there are 15 words and 300-dimensional embeddings:

```text
(15, 300)
```

---

# Part I — PCA Visualization

## Task 7

Reduce the vectors to two dimensions:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

vectors_2d = pca.fit_transform(vectors)

print("Original shape:", vectors.shape)
print("Reduced shape:", vectors_2d.shape)
```

Conceptually:

```text
300-D
 ↓
PCA
 ↓
2-D
```

Plot:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.scatter(
    vectors_2d[:, 0],
    vectors_2d[:, 1]
)

for i, word in enumerate(valid_words):
    plt.annotate(
        word,
        (vectors_2d[i, 0], vectors_2d[i, 1])
    )

plt.title("Word Embeddings Visualized Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
```

### Analyze

1. Are animal words close?
2. Are `car`, `automobile`, and `vehicle` close?
3. Are technology words close?
4. Which word placement surprises you?
5. Do all words from a category form a perfect cluster?

---

# Part J — Optional t-SNE

## Task 8

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

vectors_tsne = tsne.fit_transform(vectors)
```

Plot:

```python
plt.figure(figsize=(12, 8))

plt.scatter(
    vectors_tsne[:, 0],
    vectors_tsne[:, 1]
)

for i, word in enumerate(valid_words):
    plt.annotate(
        word,
        (vectors_tsne[i, 0], vectors_tsne[i, 1])
    )

plt.title("Word Embeddings Visualized Using t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
```

## PCA vs t-SNE

Discuss:

1. Are the plots identical?
2. Why might they differ?
3. Does t-SNE preserve all original distances?
4. Does a beautiful cluster prove that the embedding is perfect?

Important:

> A 2-D PCA/t-SNE plot is an approximation of the original high-dimensional embedding space.

---

# Part K — Polysemy

Consider:

> I deposited money in the **bank**.

and:

> The boat reached the **bank**.

Ask:

> Does traditional Word2Vec give two different vectors for `bank`?

No.

A static embedding provides one vector:

```text
bank → one vector
```

This demonstrates why static embeddings struggle with multiple meanings.

---

# Part L — Critical Analysis

Answer:

### 1. Bias

If the training corpus contains social or cultural biases, can embeddings inherit those biases?

### 2. Rare words

What happens when a word appears only a few times in the training corpus?

### 3. Corpus dependence

Would an embedding trained mostly on computer-science documents represent `Python` exactly the same way as one trained on nature/biology documents?

Why?

### 4. Similarity

Why does high cosine similarity not necessarily mean synonymy?

### 5. Understanding

Does:

\[
king-man+womanpprox queen
\]

prove that the model understands gender or monarchy like a human?

Explain.

---

# Part M — Mini Investigation

Create at least three semantic groups.

### Animals

```text
cat
dog
lion
tiger
```

### Technology

```text
computer
laptop
software
internet
```

### Transport

```text
car
automobile
vehicle
truck
```

### Food

```text
apple
banana
orange
fruit
```

Extract embeddings and visualize them using PCA.

Report:

1. Which groups cluster?
2. Which words are closest?
3. Which results are surprising?
4. Which results are unexpected?
5. What might explain them?

---

# Part N — Final Comparison

Complete:

| Property | One-Hot | BoW / TF-IDF | Word Embeddings |
|---|---|---|---|
| Representation | | | |
| Sparse/Dense | | | |
| Learned? | | | |
| Semantic relationships | | | |
| Context dependent? | | | |
| Typical use | | | |
| Main limitation | | | |

---

# Part O — Viva Questions

1. What is a word embedding?
2. Why are embeddings dense?
3. What is the distributional hypothesis?
4. What is Word2Vec?
5. What is CBOW?
6. What is Skip-Gram?
7. What is a context window?
8. What is an embedding matrix?
9. What does embedding dimension mean?
10. What is cosine similarity?
11. Why is cosine similarity useful?
12. What does `most_similar()` return?
13. What is a word analogy?
14. Why can analogy experiments fail?
15. What is GloVe?
16. How does GloVe differ conceptually from Word2Vec?
17. Why is PCA used?
18. Why is t-SNE used?
19. Why should a PCA/t-SNE plot not be treated as the original embedding space?
20. What is a static word embedding?
21. What is polysemy?
22. How can embeddings inherit bias?
23. Why does the training corpus matter?
24. Does high cosine similarity necessarily mean synonymy?
25. Do word embeddings prove that a model understands language?

---

# Part P — Submission Requirements

Submit:

## 1. Python/Jupyter Notebook

Include:

- pretrained model loading
- word-vector extraction
- embedding dimensionality
- cosine similarity
- similar-word experiments
- analogy experiment
- PCA visualization
- t-SNE visualization if assigned
- observations
- critical analysis

## 2. Figures

At minimum:

- PCA visualization
- t-SNE visualization if assigned

## 3. Short Report

Maximum 2–3 pages:

1. Objective
2. Model used
3. Embedding dimensionality
4. Similarity results
5. Analogy results
6. PCA interpretation
7. t-SNE interpretation
8. Critical observations
9. Limitations
10. Conclusion

---

# Part Q — Challenge Task

Choose a new word not used in the practical.

For example:

```text
music
```

Run:

```python
model.most_similar("music", topn=15)
```

Then:

1. Select 10 related words.
2. Extract their embeddings.
3. Calculate pairwise cosine similarities.
4. Visualize them using PCA.
5. Visualize them using t-SNE.
6. Identify clusters.
7. Explain one surprising result.
8. Explain whether the result demonstrates semantic understanding.

---

# Final Reflection

Answer in 4–6 sentences:

> Why are word embeddings a major improvement over Bag-of-Words and TF-IDF, and what problems do they still have?

A strong answer should mention:

- dense representations
- learned representations
- contextual/distributional information
- semantic relationships
- cosine similarity
- bias
- polysemy
- corpus dependence
- static representation limitations
- lack of proof of human-like understanding

# Final Conceptual Journey

```text
ONE-HOT
   ↓
Words are separate IDs

BoW
   ↓
Which words occur?

TF-IDF
   ↓
Which words distinguish documents?

WORD EMBEDDINGS
   ↓
Which words have similar learned representations?

COSINE SIMILARITY
   ↓
How similar are their vectors?

VECTOR ARITHMETIC
   ↓
Can relationships emerge geometrically?

PCA / t-SNE
   ↓
Can we visualize the learned space?

CRITICAL ANALYSIS
   ↓
What does the model actually know?
```

## Final Discussion Question

> **If two words are close in an embedding space, does that mean the model truly understands their meaning?**

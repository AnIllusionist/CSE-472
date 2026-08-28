# CSE472 — Practical: Word2Vec / GloVe Embeddings, Similarity and Visualization

**Environment:** Google Colab  
**Duration:** ~2 hours  
**Course:** Deep Learning for Natural Language Processing

## Objective

Load pretrained Word2Vec or GloVe embeddings, extract word vectors, compare words using cosine similarity, explore nearest neighbours and vector analogies, and visualize selected embeddings in 2-D using PCA or t-SNE.

The practical reinforces:

- Dense word embeddings
- Distributional hypothesis
- Context windows
- Word2Vec
- CBOW and Skip-Gram concepts
- Cosine similarity
- Semantic similarity
- Vector analogies
- GloVe/co-occurrence intuition
- PCA and t-SNE
- Static embedding limitations

## Learning Outcomes

By the end, students should be able to:

1. Load a pretrained Word2Vec/GloVe model in Colab.
2. Inspect vocabulary size and embedding dimension.
3. Extract dense vectors.
4. Compute cosine similarity.
5. Find nearest neighbours.
6. Run analogy experiments.
7. Visualize embeddings with PCA.
8. Visualize embeddings with t-SNE.
9. Interpret semantic neighbourhoods.
10. Critically discuss bias, polysemy, rare words and corpus dependence.

---

# Part A — Colab Setup

## 1. Install libraries

```python
!pip -q install gensim scikit-learn matplotlib numpy pandas
```

## 2. Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models import KeyedVectors
```

---

# Part B — Load a Pretrained Word2Vec Model

## 3. Upload the model

For a local model file supplied by the instructor:

```python
from google.colab import files

uploaded = files.upload()
```

Then load it:

```python
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)
```

> The large Google News model should be supplied/downloaded separately because of its size.

## 4. Inspect the model

```python
print("Vocabulary size:", len(model.key_to_index))
print("Embedding dimension:", model.vector_size)
```

### Questions

1. What is the vocabulary size?
2. What is the embedding dimension?
3. If the dimension is 300, what does each word receive?
4. How is this different from one-hot encoding?

---

# Part C — Extract Word Embeddings

## 5. Select words

```python
words = [
    "cat",
    "dog",
    "kitten",
    "puppy",
    "car",
    "automobile",
    "vehicle",
    "banana",
    "computer",
    "laptop"
]
```

Check availability:

```python
for word in words:
    print(word, "→", word in model)
```

## 6. Extract a vector

```python
word = "cat"

vector = model[word]

print("Word:", word)
print("Vector shape:", vector.shape)
print("First 10 values:", vector[:10])
```

### Observe

A word may look like:

```text
cat →
[0.12, -0.41, 0.73, 0.18, ...]
```

This is a dense vector.

---

# Part D — Cosine Similarity

## 7. Calculate similarity

Recall:

\[
\cos(	heta)=rac{A\cdot B}{||A||||B||}
\]

```python
w1 = "cat"
w2 = "kitten"

v1 = model[w1].reshape(1, -1)
v2 = model[w2].reshape(1, -1)

score = cosine_similarity(v1, v2)[0][0]

print(f"{w1} vs {w2}: {score:.4f}")
```

## 8. Compare several pairs

```python
pairs = [
    ("cat", "kitten"),
    ("dog", "puppy"),
    ("car", "automobile"),
    ("computer", "laptop"),
    ("cat", "banana")
]

rows = []

for w1, w2 in pairs:
    if w1 in model and w2 in model:
        v1 = model[w1].reshape(1, -1)
        v2 = model[w2].reshape(1, -1)

        score = cosine_similarity(v1, v2)[0][0]

        rows.append([w1, w2, score])

similarity_df = pd.DataFrame(
    rows,
    columns=["Word 1", "Word 2", "Cosine Similarity"]
)

similarity_df
```

### Questions

1. Which pair has the highest similarity?
2. Is `car` close to `automobile`?
3. Is `cat` close to `banana`?
4. Does high similarity necessarily mean synonymy?
5. Give an example of two related words that are not synonyms.

---

# Part E — Nearest Neighbours

## 9. Find similar words

```python
for target in ["cat", "car", "computer", "king"]:

    if target in model:

        print("
Similar to:", target)

        results = model.most_similar(
            target,
            topn=10
        )

        for word, score in results:
            print(f"{word:20s} {score:.4f}")
```

### Classification activity

For each result, label it:

- synonym
- related concept
- same topic
- contextual association
- surprising

### Question

> Does `most_similar()` mean “find synonyms”?

Explain why the answer is **no**.

---

# Part F — Odd-One-Out

## 10. Use embedding geometry

```python
group = [
    "king",
    "queen",
    "prince",
    "banana"
]

odd = model.doesnt_match(group)

print("Odd one out:", odd)
```

### Discuss

Why is the odd word likely to be `banana`?

What does this suggest about the geometry of the embedding space?

---

# Part G — Vector Analogies

## 11. Classic analogy

```python
result = model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=10
)

for word, score in result:
    print(f"{word:20s} {score:.4f}")
```

Conceptually:

\[
king-man+womanpprox queen
\]

### Questions

1. What relationship is being represented?
2. Why might vector arithmetic reveal relationships?
3. Was the model explicitly given the definition of `queen`?
4. Does this prove human-like understanding?

Important:

> Statistical pattern encoding is not the same as human-like understanding.

---

# Part H — Your Own Analogy

## 12. Try at least two new analogies

Example:

```python
model.most_similar(
    positive=["paris", "italy"],
    negative=["france"],
    topn=5
)
```

Create:

| Analogy | Top result | Score | Worked? | Explanation |
|---|---|---:|---|---|
| king - man + woman | | | | |
| Paris - France + Italy | | | | |
| Your example 1 | | | | |
| Your example 2 | | | | |

Do not assume every analogy succeeds.

A failure is also an observation.

---

# Part I — Prepare Embeddings for Visualization

## 13. Select semantic groups

```python
animals = [
    "cat",
    "dog",
    "kitten",
    "puppy",
    "lion",
    "tiger"
]

technology = [
    "computer",
    "laptop",
    "software",
    "internet",
    "keyboard",
    "program"
]

transport = [
    "car",
    "automobile",
    "vehicle",
    "truck",
    "bus",
    "train"
]

selected_words = animals + technology + transport
```

Keep only available terms:

```python
valid_words = [
    word for word in selected_words
    if word in model
]

print("Valid words:", valid_words)
```

## 14. Create embedding matrix

```python
vectors = np.array([
    model[word]
    for word in valid_words
])

print("Number of words:", len(valid_words))
print("Embedding shape:", vectors.shape)
```

For example:

```text
(18, 300)
```

means:

- 18 words
- 300 dimensions per word

---

# Part J — PCA

## 15. Reduce dimensions

```python
pca = PCA(n_components=2)

vectors_pca = pca.fit_transform(vectors)

print("Original:", vectors.shape)
print("Reduced:", vectors_pca.shape)
```

Concept:

```text
300-D
  ↓
 PCA
  ↓
 2-D
```

## 16. Plot the PCA result

```python
plt.figure(figsize=(12, 8))

plt.scatter(
    vectors_pca[:, 0],
    vectors_pca[:, 1]
)

for i, word in enumerate(valid_words):
    plt.annotate(
        word,
        (vectors_pca[i, 0], vectors_pca[i, 1])
    )

plt.title("Word Embeddings Visualized Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
```

### Questions

1. Which words appear close?
2. Do animal words form a neighbourhood?
3. Are `car`, `automobile` and `vehicle` close?
4. Which word placement is surprising?
5. Does PCA preserve every detail of the original 300-D space?

---

# Part K — t-SNE

## 17. Reduce to 2-D

```python
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

vectors_tsne = tsne.fit_transform(vectors)

print("Original:", vectors.shape)
print("Reduced:", vectors_tsne.shape)
```

## 18. Plot t-SNE

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
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()
```

---

# Part L — PCA vs t-SNE

Compare the two plots.

Answer:

1. Are the plots identical?
2. Which neighbourhoods are stable?
3. Which relationships change?
4. Why might t-SNE produce different arrangements?
5. Does t-SNE preserve all original distances?
6. Does a good-looking cluster prove semantic correctness?

Important:

> PCA/t-SNE provides a lower-dimensional visualization of high-dimensional embeddings. It is not the original embedding space.

---

# Part M — Polysemy Experiment

## 19. Examine “bank”

Consider:

```text
I deposited money in the bank.
The boat reached the bank.
```

Traditional static Word2Vec gives:

```text
bank → one vector
```

### Discussion

1. Why is one vector insufficient for multiple meanings?
2. What role does surrounding context play?
3. What kind of representation would be better?

This motivates contextual embeddings in later lectures.

---

# Part N — Corpus Dependence

## 20. Explore “Python”

```python
model.most_similar(
    "Python",
    topn=10
)
```

Discuss:

- programming language
- snake
- software
- domain context

Ask:

> Would a model trained on computer-science text and one trained on general/nature text necessarily produce the same neighbourhood for `Python`?

No.

Embedding quality depends on the training corpus.

---

# Part O — Bias and Critical Analysis

## 21. Discuss bias

Embedding learning pipeline:

```text
Human-generated text
        ↓
Training corpus
        ↓
Statistical patterns
        ↓
Learned embeddings
        ↓
Possible bias
```

Answer:

1. Can embeddings contain bias?
2. Can similarity reflect stereotypes in training data?
3. Should embedding similarity be treated as objective truth?

---

# Part P — Integrated Investigation

## 22. Build Your Own Semantic Space

Choose at least three categories.

Example:

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

Then:

1. Extract vectors.
2. Calculate selected pairwise cosine similarities.
3. Find nearest neighbours.
4. Visualize with PCA.
5. Visualize with t-SNE.
6. Compare the two visualizations.
7. Identify one surprising result.
8. Explain a possible reason.

---

# Part Q — Final Comparison

Complete:

| Property | One-Hot | BoW / TF-IDF | Word2Vec / GloVe |
|---|---|---|---|
| Sparse/Dense | | | |
| Learned? | | | |
| Semantic relationships | | | |
| Uses context to learn representation? | | | |
| One vector per word? | | | |
| Supports similarity search? | | | |
| Main limitation | | | |

---

# Part R — Viva Questions

1. What is a word embedding?
2. Why are word embeddings dense?
3. What is the distributional hypothesis?
4. What is Word2Vec?
5. What is CBOW?
6. What is Skip-Gram?
7. What is a context window?
8. What is an embedding dimension?
9. What is an embedding matrix?
10. What is cosine similarity?
11. Why is cosine similarity useful?
12. What does `most_similar()` calculate?
13. Does high similarity imply synonymy?
14. What is a word analogy?
15. Why can analogy experiments fail?
16. What is GloVe?
17. How is GloVe conceptually different from Word2Vec?
18. Why is PCA used?
19. Why is t-SNE used?
20. Why should 2-D visualization not be treated as the original space?
21. What is a static word embedding?
22. What is polysemy?
23. How can corpus choice affect embeddings?
24. How can embeddings inherit bias?
25. Do embeddings prove human-like language understanding?

---

# Part S — Submission Requirements

Submit **one Google Colab notebook** containing:

1. Setup
2. Pretrained model loading
3. Model/vocabulary information
4. Selected word vectors
5. Cosine similarity
6. Nearest-neighbour experiments
7. Odd-one-out experiment
8. Analogy experiment
9. PCA visualization
10. t-SNE visualization
11. Polysemy analysis
12. Corpus-dependence analysis
13. Bias/critical analysis
14. Final comparison
15. Reflection

### Minimum required outputs

- 3 cosine-similarity comparisons
- 1 nearest-neighbour experiment
- 1 odd-one-out experiment
- 1 analogy experiment
- 1 PCA plot
- 1 t-SNE plot
- critical analysis of at least 2 limitations

---

# Part T — Optional GloVe Extension

If a pretrained GloVe model is supplied, repeat:

```text
GloVe
 ↓
word vectors
 ↓
cosine similarity
 ↓
nearest neighbours
 ↓
PCA
 ↓
t-SNE
```

Compare the same words under Word2Vec and GloVe.

Record:

| Word | Word2Vec neighbour | GloVe neighbour | Observation |
|---|---|---|---|
| cat | | | |
| car | | | |
| computer | | | |

Discuss why two pretrained models may produce different neighbourhoods.

---

# Final Reflection

Answer in 5–7 sentences:

> If Word2Vec can place words such as `cat` and `kitten` close together without being explicitly told that they are similar, what has the model actually learned?

Distinguish between:

- statistical patterns of language use
- semantic relationships that emerge in the vector space
- and genuine human-like understanding of meaning

## Final Conceptual Journey

```text
Word
 ↓
Pretrained Vector
 ↓
Dense Representation
 ↓
Cosine Similarity
 ↓
Semantic Neighbourhood
 ↓
Analogy
 ↓
PCA / t-SNE
 ↓
Visualization
 ↓
Critical Analysis
```

> **Final question:** If two words are close in an embedding space, does that mean the model truly understands their meaning?

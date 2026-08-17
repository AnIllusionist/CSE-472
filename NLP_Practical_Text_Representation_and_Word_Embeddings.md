# Practical: Text Representation and Word Embedding Visualization

## Course
**CSE472 — Deep Learning for Natural Language Processing**

## Practical Title
**From Text to Semantic Vector Space: BoW, TF-IDF, Word2Vec/GloVe and PCA/t-SNE**

## Duration
**Approximately 2 hours**

## 1. Practical Overview

This practical builds the complete workflow:

```text
Processed Text
      ↓
Bag-of-Words
      ↓
TF-IDF
      ↓
Pretrained Word Embeddings
      ↓
Selected Word Vectors
      ↓
Cosine Similarity
      ↓
PCA / t-SNE
      ↓
2-D Semantic Visualization
```

The practical connects the Week 2 lectures with implementation.

## 2. Learning Outcomes

By the end of this practical, students should be able to:

1. Convert processed text into a Bag-of-Words representation.
2. Convert text into a TF-IDF representation.
3. Load pretrained Word2Vec or GloVe embeddings.
4. Extract embeddings for selected words.
5. Compute similarity between word embeddings.
6. Reduce high-dimensional embeddings to 2-D using PCA.
7. Optionally visualize embeddings using t-SNE.
8. Interpret clusters and relationships in an embedding space.
9. Explain the difference between sparse TF-IDF representations and dense word embeddings.

# Part A — Text → Numerical Representations

## Task 1: Create a Small Text Corpus

```python
documents = [
    "The cat drinks milk and likes fish.",
    "The dog drinks milk and likes meat.",
    "The kitten likes fish and plays with toys.",
    "The puppy plays with toys and likes meat.",
    "Cars and automobiles are useful vehicles.",
    "The car is fast and the automobile is comfortable."
]
```

### Before running the code, discuss

1. Which words do you expect to occur frequently?
2. Which words might distinguish documents?
3. Will `car` and `automobile` become the same feature?
4. Can a word-count representation know that they are semantically related?

## Task 2: Bag-of-Words

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_bow = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nBoW Matrix:")
print(X_bow.toarray())
```

### Questions

1. What does a row represent?
2. What does a column represent?
3. What does `0` mean?
4. What does a value greater than `0` mean?
5. What happens to dimensionality as vocabulary size increases?

## Task 3: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(documents)

print("Vocabulary:")
print(tfidf.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(X_tfidf.toarray())
```

### Questions

1. How is TF-IDF different from BoW?
2. Why are the values no longer simple integer counts?
3. Which terms receive relatively higher weights?
4. Why might a common term receive a lower weight?
5. Are `car` and `automobile` still separate features?

### Key observation

> **TF-IDF does not inherently know that `car` and `automobile` are semantically related.**

This motivates dense word embeddings.

# Part B — Pretrained Word Embeddings

## Task 4: Load a Pretrained Word2Vec Model

If the instructor provides a Word2Vec model, load it with:

```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)
```

The pretrained model file should be provided/downloaded separately because it is large.

If a pretrained GloVe model is provided, use the instructor-provided loading procedure.

## Task 5: Inspect Embeddings

```python
words = [
    "cat",
    "dog",
    "kitten",
    "puppy",
    "car",
    "automobile",
    "banana",
    "computer"
]

for word in words:
    if word in model:
        print(word, "found")
    else:
        print(word, "not found")
```

Extract vectors:

```python
for word in words:
    if word in model:
        vector = model[word]
        print(word)
        print("Embedding dimension:", len(vector))
        print("First 10 values:", vector[:10])
        print()
```

### Questions

1. What is the embedding dimension?
2. Why is an embedding dense?
3. How is it different from a one-hot vector?
4. Does each dimension have an obvious human-readable meaning?

# Part C — Semantic Similarity

## Task 6: Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

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
2. Are `car` and `automobile` close?
3. Are `cat` and `banana` close?
4. Does high similarity necessarily mean synonymy?
5. Why can two related but non-synonymous words have high similarity?

## Task 7: Find Similar Words

```python
for target in ["cat", "car", "computer", "king"]:
    print("\nSimilar to:", target)
    for word, score in model.most_similar(target, topn=10):
        print(f"{word:20s} {score:.4f}")
```

Discuss whether the returned words are:

- synonyms
- related concepts
- contextually similar
- topically similar

# Part D — Word Analogies

## Task 8: Vector Arithmetic

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

```text
king - man + woman ≈ queen
```

### Discussion

1. What relationship is being captured?
2. Was the model explicitly taught the definition of `queen`?
3. Does successful analogy completion prove human-like understanding?
4. Can analogy results fail?

Key idea:

> **Statistical pattern ≠ human-like understanding.**

# Part E — PCA Visualization

## Task 9: Select Words

```python
selected_words = [
    "cat", "dog", "kitten", "puppy",
    "car", "automobile", "vehicle",
    "banana", "apple", "orange",
    "computer", "laptop"
]

valid_words = [
    word for word in selected_words
    if word in model
]
```

Extract embeddings:

```python
import numpy as np

vectors = np.array([
    model[word]
    for word in valid_words
])

print("Number of words:", len(valid_words))
print("Embedding shape:", vectors.shape)
```

## Task 10: Reduce Using PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

vectors_2d = pca.fit_transform(vectors)

print("Original shape:", vectors.shape)
print("Reduced shape:", vectors_2d.shape)
```

Conceptually:

```text
300D
 ↓
PCA
 ↓
2D
```

## Task 11: Plot PCA

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

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

# Part F — Optional t-SNE

## Task 12: t-SNE

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
plt.figure(figsize=(10, 7))

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

## PCA vs t-SNE Discussion

Answer:

1. Which words appear close together?
2. Do animal words form a meaningful neighborhood?
3. Are `car`, `automobile`, and `vehicle` close?
4. Are PCA and t-SNE plots identical?
5. Why might the two plots differ?
6. Does a 2-D plot preserve all information from the original embedding?

Important:

> **PCA/t-SNE is a visualization of the embedding space, not the original high-dimensional space.**

# Part G — Mini Investigation

Create at least three semantic groups.

Example:

### Animals

```text
cat, dog, lion, tiger
```

### Technology

```text
computer, laptop, software, internet
```

### Transport

```text
car, automobile, vehicle, truck
```

Extract their embeddings and visualize them.

### Investigation Questions

1. Do words from the same category tend to be close?
2. Which word is the most surprising?
3. Which words are unexpectedly far apart?
4. Does the visualization support your expectations?

# Part H — Critical Analysis

Answer the following:

1. If `car` and `automobile` are close, what does that tell us?
2. Does high cosine similarity necessarily mean two words are synonyms?
3. What happens if the training corpus contains social or cultural biases?
4. Consider:
   - “I deposited money in the bank.”
   - “The boat reached the bank.”

   Would traditional static Word2Vec give `bank` two different vectors?
5. Why might rare words have poorer embeddings?
6. Why are individual embedding dimensions generally not directly interpretable?
7. Why are embeddings dependent on their training corpus?

# Part I — Final Comparison

Complete:

| Property | BoW | TF-IDF | Word2Vec/GloVe |
|---|---|---|---|
| Representation | | | |
| Sparse/Dense | | | |
| Learned? | | | |
| Captures word order? | | | |
| Semantic similarity? | | | |
| Typical dimensionality | | | |
| Main limitation | | | |

# Part J — Viva Questions

1. What is Bag-of-Words?
2. What is TF-IDF?
3. Why is TF-IDF sparse?
4. What is a word embedding?
5. What is the distributional hypothesis?
6. How does Word2Vec learn representations?
7. What is CBOW?
8. What is Skip-Gram?
9. What is a context window?
10. What is an embedding matrix?
11. Why are embeddings dense?
12. What is cosine similarity?
13. Why is cosine similarity useful?
14. What is a pretrained embedding?
15. What is GloVe?
16. How does GloVe differ from Word2Vec?
17. What is PCA?
18. Why reduce embeddings to 2-D?
19. What is t-SNE?
20. Why can t-SNE plots be misleading?
21. What is polysemy?
22. How can embeddings contain bias?
23. Why can static embeddings struggle with multiple meanings?
24. What is the difference between static and contextual embeddings?

# Part K — Submission Requirements

Submit:

### 1. Python Notebook

Include:

- corpus preparation
- BoW
- TF-IDF
- pretrained embedding loading
- selected word embeddings
- cosine similarity
- similar-word results
- analogy experiment
- PCA visualization
- t-SNE visualization
- observations

### 2. Figures

At minimum:

- PCA visualization
- t-SNE visualization

### 3. Short Report

Maximum **2–3 pages**:

1. Objective
2. Methodology
3. BoW observations
4. TF-IDF observations
5. Embedding similarity results
6. PCA interpretation
7. t-SNE interpretation
8. PCA vs t-SNE comparison
9. Critical analysis
10. Conclusion

# Part L — Challenge Task

Choose a new word that was not used in the examples.

For example:

```text
music
```

Then:

1. Find its top 15 similar words.
2. Select 10 related words.
3. Extract their embeddings.
4. Calculate pairwise cosine similarities.
5. Visualize them using PCA.
6. Visualize them using t-SNE.
7. Identify meaningful clusters.
8. Explain at least one surprising result.

# Final Question

> **If TF-IDF already converts text into numbers, why do we need word embeddings?**

A strong answer should explain that TF-IDF primarily represents word importance based on occurrence/frequency, whereas word embeddings learn dense representations from language usage/context and can encode useful relationships between words. However, embeddings are not perfect: they can inherit bias, struggle with polysemy, depend on their training corpus, and do not prove human-like language understanding.

## Expected Conceptual Journey

```text
BoW
  ↓
"Which words occur?"

TF-IDF
  ↓
"Which words are important?"

Word2Vec / GloVe
  ↓
"What words occur in similar contexts?"

Embedding Space
  ↓
"Which words have similar representations?"

Cosine Similarity
  ↓
"How similar are these representations?"

PCA / t-SNE
  ↓
"Can we visualize these relationships?"

Critical Analysis
  ↓
"Are these representations actually understanding language?"
```

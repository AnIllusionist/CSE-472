# CSE472 — NLP Practical Class
## CBOW, Skip-Gram, GloVe Embeddings, PCA and t-SNE

**Environment:** Google Colab  
**Recommended duration:** 2 hours  
**Level:** Beginner-friendly hands-on practical

---

## 1. Practical Aim

In this practical, you will:

1. Train a **CBOW Word2Vec** model.
2. Train a **Skip-Gram Word2Vec** model.
3. Find similar words.
4. Try word analogies.
5. Compare CBOW and Skip-Gram.
6. Load/use pretrained **Word2Vec or GloVe** embeddings.
7. Select words from different semantic categories.
8. Reduce high-dimensional vectors to 2-D using **PCA**.
9. Optionally use **t-SNE**.
10. Visualize and interpret semantic clusters.

The complete flow is:

```text
Sample Text
    ↓
Tokenization
    ↓
CBOW / Skip-Gram
    ↓
Learned Word Vectors
    ↓
Similarity + Analogy
    ↓
Pretrained Embeddings
    ↓
PCA / t-SNE
    ↓
2-D Visualization
```

---

# 2. Learning Outcomes

After this practical, you should be able to:

- explain CBOW and Skip-Gram in simple words;
- train both models using Gensim;
- change basic Word2Vec parameters;
- find similar words;
- perform simple analogy experiments;
- compare CBOW and Skip-Gram;
- load pretrained embeddings;
- use PCA/t-SNE for visualization;
- interpret semantic clusters;
- discuss limitations of static embeddings.

---

# 3. Google Colab Setup

```python
!pip -q install gensim scikit-learn matplotlib numpy pandas
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

---

# Part A — Prepare a Sample Corpus

## Task 1: Create the Corpus

Use a small corpus so that model behavior is easy to inspect.

```python
sentences = [
    "The cat drinks milk",
    "The cat likes fish",
    "The dog drinks milk",
    "The dog likes meat",
    "The kitten drinks milk",
    "The kitten likes fish",
    "The puppy likes meat",
    "The puppy plays with toys",
    "The car is fast",
    "The automobile is fast",
    "The vehicle moves quickly",
    "The computer is useful",
    "The laptop is useful",
    "The computer runs software"
]
```

### Before running the model, predict:

1. Which words should be related?
2. Which words have similar contexts?
3. Should `cat` and `kitten` be similar?
4. Should `car` and `automobile` be similar?
5. Should `cat` and `banana` be similar?

---

## Task 2: Tokenize

```python
tokenized_sentences = [
    sentence.lower().split()
    for sentence in sentences
]

for sentence in tokenized_sentences:
    print(sentence)
```

Example:

```text
"The cat drinks milk"
```

becomes:

```text
["the", "cat", "drinks", "milk"]
```

---

# Part B — Train CBOW

## Task 3: Train a CBOW Model

```python
cbow_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=2,
    sg=0
)
```

### Important

```text
sg=0 → CBOW
```

Basic parameters:

```text
vector_size → embedding dimension
window      → context window
min_count   → minimum word frequency
workers     → training threads
sg          → training architecture
```

---

## Task 4: What Does CBOW Do?

CBOW means:

> **Context → Target**

For:

```text
The cat drinks milk
```

the model uses surrounding words to predict a target word.

Conceptually:

```text
cat + milk
    ↓
  CBOW
    ↓
 drinks
```

---

## Task 5: Inspect a CBOW Vector

```python
vector = cbow_model.wv["cat"]

print("Shape:", vector.shape)
print("Embedding dimension:", cbow_model.vector_size)
print("First 10 values:", vector[:10])
```

If:

```text
vector_size = 50
```

then:

```text
cat → 50 numerical values
```

---

# Part C — Similar Words with CBOW

## Task 6

```python
for word in ["cat", "dog", "car", "computer"]:

    print("\nSimilar to:", word)

    try:
        print(cbow_model.wv.most_similar(word, topn=5))
    except KeyError:
        print("Word not found.")
```

### Questions

1. Which words are returned for `cat`?
2. Are they synonyms or simply related?
3. Is `car` close to `automobile`?
4. Which result surprises you?
5. Why might a tiny corpus produce unexpected results?

---

# Part D — Train Skip-Gram

## Task 7

```python
skipgram_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=2,
    sg=1
)
```

### Important

```text
sg=1 → Skip-Gram
```

---

## Task 8: Understand Skip-Gram

Skip-Gram reverses the task:

> **Target → Context**

For:

```text
The cat drinks milk
```

target:

```text
cat
```

predict:

```text
the
drinks
```

Conceptually:

```text
          cat
         /           ↓     ↓
       the  drinks
```

---

# Part E — Compare CBOW and Skip-Gram

## Task 9

```python
test_words = [
    "cat",
    "dog",
    "car",
    "computer"
]

for word in test_words:

    print("
======================")
    print("WORD:", word)

    print("CBOW:")
    print(cbow_model.wv.most_similar(word, topn=5))

    print("Skip-Gram:")
    print(skipgram_model.wv.most_similar(word, topn=5))
```

Record:

| Word | CBOW neighbours | Skip-Gram neighbours | Observation |
|---|---|---|---|
| cat | | | |
| dog | | | |
| car | | | |
| computer | | | |

### Discussion

1. Are the results identical?
2. Which gives more useful neighbours?
3. Why might the results differ?
4. Can you conclude that CBOW is always better?
5. Can you conclude that Skip-Gram is always better?

Expected conclusion:

> No. Performance depends on the corpus, vocabulary, context window, training settings, and task.

---

# Part F — Context Window Experiment

## Task 10

Train another CBOW model with a smaller window:

```python
cbow_w1 = Word2Vec(
    tokenized_sentences,
    vector_size=50,
    window=1,
    min_count=1,
    workers=2,
    sg=0
)
```

And a larger window:

```python
cbow_w4 = Word2Vec(
    tokenized_sentences,
    vector_size=50,
    window=4,
    min_count=1,
    workers=2,
    sg=0
)
```

Compare:

```python
print("Window = 1")
print(cbow_w1.wv.most_similar("cat", topn=5))

print("\nWindow = 4")
print(cbow_w4.wv.most_similar("cat", topn=5))
```

### Questions

1. Did the neighbours change?
2. Why?
3. What does a larger window capture?
4. What is the trade-off of a larger context window?

---

# Part G — Cosine Similarity

## Task 11

Use:

```python
pairs = [
    ("cat", "kitten"),
    ("dog", "puppy"),
    ("car", "automobile"),
    ("car", "banana")
]

for w1, w2 in pairs:

    try:
        score = cbow_model.wv.similarity(w1, w2)
        print(f"{w1:12s} {w2:12s} → {score:.4f}")
    except KeyError:
        print(w1, w2, "→ unavailable")
```

### Questions

1. Which pair is most similar?
2. Why might similarity be unreliable on a tiny corpus?
3. Does high similarity mean synonymy?
4. Give one pair of related but non-synonymous words.

---

# Part H — Word Analogy

## Task 12

A small classroom corpus may not learn meaningful analogies. Use a pretrained model for the classic analogy experiment.

Conceptually:

\[
king-man+womanpprox queen
\]

Later, run it with the pretrained model:

```python
result = pretrained_model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)

for word, score in result:
    print(f"{word:20s} {score:.4f}")
```

### Discussion

1. What relationship is being represented?
2. Why can vector arithmetic capture relationships?
3. Does this prove human-like understanding?
4. Why can analogy experiments fail?

Important:

> **A learned statistical relationship is not the same as human-like understanding.**

---

# Part I — Load a Pretrained Word2Vec Model

## Task 13

If your instructor provides a pretrained model:

```python
pretrained_model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)
```

Inspect:

```python
print("Vocabulary size:",
      len(pretrained_model.key_to_index))

print("Embedding dimension:",
      pretrained_model.vector_size)
```

> The large model file should be provided separately because of its size.

---

# Part J — Explore Pretrained Embeddings

## Task 14

```python
for word in ["king", "happy", "computer", "car"]:

    if word in pretrained_model:

        print("\nSimilar to:", word)

        for w, score in pretrained_model.most_similar(
            word,
            topn=5
        ):
            print(f"{w:20s} {score:.4f}")
```

### Compare

How are these results different from the tiny model trained in Part C?

Discuss:

- corpus size
- training quality
- vocabulary
- domain
- training time

---

# Part K — Odd-One-Out

## Task 15

```python
group = [
    "king",
    "queen",
    "prince",
    "banana"
]

print(
    "Odd one out:",
    pretrained_model.doesnt_match(group)
)
```

### Question

Why does the embedding space identify the odd word?

---

# Part L — Word Analogy

## Task 16

```python
result = pretrained_model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=10
)

for word, score in result:
    print(f"{word:20s} {score:.4f}")
```

Interpret:

\[
king-man+womanpprox queen
\]

---

# Part M — Select Semantic Categories

## Task 17

Create word groups.

### Animals

```python
animals = [
    "cat",
    "dog",
    "kitten",
    "puppy",
    "lion",
    "tiger"
]
```

### Technology

```python
technology = [
    "computer",
    "laptop",
    "software",
    "internet",
    "keyboard"
]
```

### Transport

```python
transport = [
    "car",
    "automobile",
    "vehicle",
    "truck",
    "bus"
]
```

Combine:

```python
selected_words = animals + technology + transport
```

Keep only words in the model:

```python
valid_words = [
    word for word in selected_words
    if word in pretrained_model
]

print(valid_words)
```

---

# Part N — Create the Embedding Matrix

## Task 18

```python
vectors = np.array([
    pretrained_model[word]
    for word in valid_words
])

print("Number of words:", len(valid_words))
print("Shape:", vectors.shape)
```

For example:

```text
(16, 300)
```

means:

```text
16 words
300 dimensions per word
```

---

# Part O — PCA Visualization

## Task 19

Reduce the vectors from high-dimensional space to 2-D:

```python
pca = PCA(n_components=2)

vectors_pca = pca.fit_transform(vectors)

print("Original shape:", vectors.shape)
print("Reduced shape:", vectors_pca.shape)
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

plt.title("Word Embeddings — PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()
```

### Analyze

1. Do animal words appear near each other?
2. Are `car`, `automobile`, and `vehicle` close?
3. Do technology words form a neighbourhood?
4. Which result surprises you?
5. Does PCA preserve every detail of the original space?

---

# Part P — t-SNE Visualization

## Task 20

```python
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

vectors_tsne = tsne.fit_transform(vectors)

print("Original shape:", vectors.shape)
print("Reduced shape:", vectors_tsne.shape)
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

plt.title("Word Embeddings — t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)

plt.show()
```

---

# Part Q — PCA vs t-SNE

Compare the two visualizations.

Answer:

1. Are they identical?
2. Which semantic neighbourhoods remain visible?
3. Which positions change?
4. Why might they differ?
5. Does t-SNE preserve exact global distances?
6. Does a beautiful cluster prove that the embedding is correct?

Important:

> PCA and t-SNE provide lower-dimensional views of high-dimensional vectors. They do not reproduce the original space perfectly.

---

# Part R — GloVe Extension

If your instructor provides pretrained GloVe vectors, repeat the following workflow:

```text
GloVe
 ↓
Word vectors
 ↓
Cosine similarity
 ↓
Nearest neighbours
 ↓
Analogy
 ↓
PCA
 ↓
t-SNE
```

Compare selected words under Word2Vec and GloVe.

Record:

| Word | Word2Vec neighbour | GloVe neighbour | Observation |
|---|---|---|---|
| cat | | | |
| car | | | |
| computer | | | |

Discuss:

> Why can two pretrained embedding models produce different neighbourhoods?

---

# Part S — Critical Analysis

## Task 21 — Polysemy

Consider:

> I deposited money in the **bank**.

and:

> The boat reached the **bank**.

Traditional static embeddings give:

```text
bank → one vector
```

Question:

> Why can one static vector struggle with multiple meanings?

---

## Task 22 — Corpus Dependence

Consider:

```text
Python
```

Possible meanings:

- programming language
- snake

Ask:

> Would a model trained mostly on computer-science text necessarily represent Python in the same way as a model trained on nature text?

Explain.

---

## Task 23 — Bias

Discuss:

```text
Human-generated text
        ↓
Training corpus
        ↓
Statistical patterns
        ↓
Embedding
        ↓
Possible bias
```

Questions:

1. Can embeddings contain bias?
2. Why?
3. Should embedding similarity be treated as objective truth?

---

# Part T — Final Comparison

Complete:

| Property | CBOW | Skip-Gram | GloVe |
|---|---|---|---|
| Basic idea | | | |
| Input | | | |
| Output | | | |
| Training signal | | | |
| Local/global information | | | |
| Main strength | | | |
| Main limitation | | | |

---

# Part U — Final Investigation

## Task 24 — Build Your Own Semantic Space

Choose at least three categories.

Example:

```text
Animals:
cat, dog, lion, tiger

Technology:
computer, laptop, software, internet

Transport:
car, automobile, vehicle, truck
```

Then:

1. Extract embeddings.
2. Calculate selected cosine similarities.
3. Find nearest neighbours.
4. Perform PCA.
5. Perform t-SNE.
6. Compare the plots.
7. Identify one surprising result.
8. Explain a possible reason.

---

# Part V — Viva Questions

1. What is Word2Vec?
2. What is CBOW?
3. What is Skip-Gram?
4. What is the difference between CBOW and Skip-Gram?
5. What is a context window?
6. What does `sg=0` mean in Gensim?
7. What does `sg=1` mean?
8. What is an embedding dimension?
9. What is a pretrained embedding?
10. Why are word embeddings dense?
11. What is cosine similarity?
12. What does `most_similar()` return?
13. What is an analogy?
14. Why can analogies fail?
15. What is GloVe?
16. How does GloVe differ from Word2Vec?
17. Why is PCA used?
18. Why is t-SNE used?
19. Why can PCA and t-SNE plots look different?
20. What is a static embedding?
21. What is polysemy?
22. How can embeddings inherit bias?
23. Why does corpus choice matter?
24. Does similarity always mean synonymy?
25. Do word embeddings prove human-like understanding?

---

# Part W — Submission Requirements

Submit **one Google Colab notebook** containing:

1. Corpus creation
2. Tokenization
3. CBOW training
4. Skip-Gram training
5. Similar-word results
6. CBOW vs Skip-Gram comparison
7. Context-window experiment
8. Pretrained Word2Vec/GloVe loading
9. Cosine-similarity experiments
10. Odd-one-out experiment
11. At least one analogy experiment
12. PCA visualization
13. t-SNE visualization
14. Critical analysis
15. Final comparison table

### Minimum required outputs

- 3 similarity comparisons
- 1 odd-one-out experiment
- 1 analogy experiment
- 1 CBOW vs Skip-Gram comparison
- 1 PCA plot
- 1 t-SNE plot
- 2 critical observations

---

# Part X — Final Reflection

Answer in 5–7 sentences:

> How do CBOW, Skip-Gram and GloVe learn useful numerical representations of words, and what are the main limitations of these static embeddings?

Include:

- context/distributional information
- dense vectors
- semantic similarity
- analogy relationships
- corpus dependence
- polysemy
- bias
- static-vector limitation

---

# Final Conceptual Journey

```text
CBOW
Context → Target

Skip-Gram
Target → Context

        ↓

Word2Vec
        ↓
Learned Dense Vectors
        ↓
Similarity
        ↓
Analogies

GloVe
Global Co-occurrence
        ↓
Dense Vectors

        ↓
PCA / t-SNE
        ↓
2-D Visualization
        ↓
Interpret Relationships
        ↓
Critical Analysis
```

## Final Question

> **If two words are close together in an embedding space, what can we conclude about their relationship — and what can we NOT conclude about their meaning?**

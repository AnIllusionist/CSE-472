# CSE472 — Lecture Notes
## GloVe Embeddings, Semantic Similarity, PCA and t-SNE

**Course:** Deep Learning for Natural Language Processing  
**Topic:** GloVe Embeddings and Visualization  
**Level:** Simple, beginner-friendly explanation

---

## 1. Where Are We?

So far:

```text
Text
 ↓
Tokens
 ↓
BoW
 ↓
TF-IDF
 ↓
Vector Space Models
 ↓
Dense Word Embeddings
 ↓
Word2Vec
```

Today we ask:

> **Can we learn word vectors using global word co-occurrence information?**

This leads to **GloVe**.

---

## 2. What Is GloVe?

**GloVe** means **Global Vectors for Word Representation**.

Simple idea:

> GloVe uses how words occur together across a large collection of text to learn dense word vectors.

Remember:

```text
Global co-occurrence
        ↓
Learned vectors
```

---

## 3. What Is Co-Occurrence?

Two words co-occur when they appear together within a chosen context.

Example:

> The cat drinks milk.

Possible relationships include:

```text
cat ↔ drinks
cat ↔ milk
drinks ↔ milk
```

The exact count depends on the corpus and the context/window definition.

---

## 4. Simple Example

Suppose a corpus contains:

> The cat drinks milk.  
> The dog drinks milk.  
> The cat likes fish.  
> The dog likes meat.

We can observe:

```text
cat → milk, fish
dog → milk, meat
```

Words that repeatedly occur with related words can develop related representations.

---

## 5. Co-Occurrence Matrix

A simplified matrix might be:

|       | cat | dog | milk | fish | meat |
|------|----:|----:|----:|----:|----:|
| cat  | – | 0 | 2 | 1 | 0 |
| dog  | 0 | – | 2 | 0 | 1 |
| milk | 2 | 2 | – | 0 | 0 |
| fish | 1 | 0 | 0 | – | 0 |
| meat | 0 | 1 | 0 | 0 | – |

For example:

```text
cat-milk = 2
```

means the two words co-occurred twice according to the chosen counting rule.

---

## 6. Why Not Use This Matrix Directly?

A vocabulary of 50,000 words could produce:

```text
50,000 × 50,000
```

features.

That is huge and mostly sparse.

So we want:

> **A smaller, dense representation that keeps useful information.**

This is one reason learned embeddings are useful.

---

## 7. Word2Vec vs GloVe

You already learned Word2Vec.

### Word2Vec

```text
Context
   ↓
Prediction
   ↓
Learned vector
```

### GloVe

```text
Global co-occurrence
        ↓
Statistical structure
        ↓
Learned vector
```

Both aim to learn useful dense word representations.

---

## 8. Why Global Information?

Word2Vec often looks at local context windows.

GloVe uses information gathered from the corpus as a whole.

Imagine millions of sentences.

We can ask:

> Across the entire corpus, which words repeatedly occur together?

That provides a global view of word usage.

---

## 9. A Useful GloVe Idea: Ratios

Consider:

```text
ice
steam
```

and contexts:

```text
solid
gas
water
```

Suppose, as an illustration:

```text
P(solid | ice)   = 0.40
P(solid | steam) = 0.05
```

Then:

\[
rac{P(solid|ice)}{P(solid|steam)} = 8
\]

Now suppose:

```text
P(gas | ice)   = 0.02
P(gas | steam) = 0.30
```

Then:

\[
rac{0.02}{0.30} pprox 0.067
\]

### Intuition

The ratio tells us which context is more strongly associated with one target word than another.

These values are illustrative; the important idea is the comparison of co-occurrence patterns.

---

## 10. The Main GloVe Idea

Remember:

> **Word relationships can be reflected in patterns of word-context co-occurrence.**

So:

```text
Global statistics
       ↓
Relationships
       ↓
Dense vectors
```

---

## 11. GloVe Equation

A simplified GloVe relationship is:

\[
w_i^T	ilde{w}_j+b_i+	ilde{b}_j
pprox
\log(X_{ij})
\]

Where:

- \(X_{ij}\) = co-occurrence count
- \(w_i\) = learned word vector
- \(	ilde{w}_j\) = context vector
- \(b_i,	ilde{b}_j\) = bias terms

### In simple words

GloVe learns vectors whose interactions reflect observed co-occurrence statistics.

---

## 12. Why Use log?

Suppose counts are:

```text
1
10
100
10,000
1,000,000
```

Raw counts have a huge range.

A logarithm compresses that scale.

So a very large count does not dominate as strongly.

Remember:

> **log helps compress very large count differences.**

---

# 13. From Vectors to Similarity

After learning, each word has a dense vector.

For example:

```text
cat
→ [0.21, -0.43, 0.72, ...]
```

and:

```text
kitten
→ [0.19, -0.40, 0.70, ...]
```

If their directions are similar, their cosine similarity can be high.

---

## 14. Cosine Similarity

\[
\cos(	heta)=
rac{A\cdot B}
{\|A\|\|B\|}
\]

Simple meaning:

> **How similar are the directions of the two vectors?**

High similarity:

```text
→
  →
```

Low similarity:

```text
→
↑
```

---

## 15. Simple Numerical Example

Let:

\[
A=[1,2]
\]

\[
B=[2,4]
\]

Dot product:

\[
A\cdot B=(1)(2)+(2)(4)=10
\]

Magnitudes:

\[
\|A\|=\sqrt5
\]

\[
\|B\|=\sqrt{20}
\]

Therefore:

\[
\cos(	heta)
=
rac{10}{\sqrt5\sqrt{20}}
=1
\]

### Interpretation

The vectors point in exactly the same direction.

---

# 16. Semantic Similarity

Ask students to predict:

```text
cat — kitten
car — automobile
doctor — hospital
car — banana
```

Which should be most related?

Important:

> **Similarity does not necessarily mean synonymy.**

For example:

```text
doctor ↔ hospital
```

can be strongly related without being synonyms.

---

# 17. Nearest Neighbours

A pretrained embedding can return nearby words.

For example:

```python
model.most_similar("computer")
```

The results are based on vector similarity.

Ask:

> Are all returned words synonyms?

No.

They may be:

- semantically related
- topically related
- contextually related
- associated concepts

---

# 18. Analogy Relationships

A famous example is:

\[
king-man+womanpprox queen
\]

This suggests that some relationships appear as approximate directions in the embedding space.

Conceptually:

```text
man → woman
king → queen
```

---

## 19. Analogy Discussion

Ask:

> Does this prove that the model understands monarchy?

Not necessarily.

It shows that the learned representation contains a useful statistical relationship.

Important:

> **Statistical pattern ≠ human-like understanding**

---

# 20. Why Do We Need PCA?

Suppose each word has a:

```text
300-dimensional vector
```

Can we draw 300 dimensions?

No.

Humans can easily inspect:

```text
1-D
2-D
3-D
```

but not directly visualize 300-D.

So we reduce:

\[
300D ightarrow 2D
\]

for visualization.

---

# 21. PCA in Simple Words

**PCA = Principal Component Analysis**

For this lecture, think of PCA as:

> **A technique that transforms high-dimensional data into a smaller number of new dimensions while trying to preserve the most important variation.**

Example:

```text
300 dimensions
       ↓
      PCA
       ↓
2 principal components
       ↓
2-D plot
```

### Important

PCA does not simply delete some original columns.

It creates new components from the original features.

---

# 22. Real-World PCA Analogy

Imagine a 3-D object:

```text
height
width
depth
```

Take a photograph.

The photo is:

```text
3-D → 2-D
```

You lose some information, but preserve enough structure to see the object.

Similarly:

```text
High-dimensional embedding
          ↓
         PCA
          ↓
       2-D view
```

The 2-D plot is a simplified view.

---

# 23. PCA with Word Embeddings

Suppose:

```text
cat      → 300 values
dog      → 300 values
kitten   → 300 values
car      → 300 values
banana   → 300 values
```

PCA converts each vector into two values:

```text
cat     → [2.3, -0.7]
dog     → [2.1, -0.6]
car     → [-1.8, 0.9]
```

Now we can plot them.

---

# 24. Visualizing the Embedding Space

Conceptual example:

```text
       dog ●
  kitten ●
     cat ●


                         ● car
                    ● automobile


                                  ● banana
```

Words that are close in the learned space may have related usage patterns.

But do not expect perfect clusters.

---

# 25. t-SNE

Another method is:

> **t-SNE**

It is commonly used to visualize high-dimensional data while paying strong attention to local neighbourhoods.

Conceptually:

```text
High-dimensional vectors
          ↓
         t-SNE
          ↓
         2-D
```

---

# 26. PCA vs t-SNE

| PCA | t-SNE |
|---|---|
| Linear | Nonlinear |
| Finds directions of high variance | Emphasizes local neighbourhoods |
| Useful for broad structure | Useful for local cluster visualization |
| Usually easier to interpret | More sensitive to parameters |

The important point:

> Both are tools for inspecting high-dimensional data.

---

# 27. Visualization Warning

A beautiful t-SNE plot does not prove that the embedding is perfect.

Why?

- dimensionality reduction can distort geometry
- t-SNE is parameter-sensitive
- 2-D is only a projection/view of the original space

Remember:

\[
oxed{	ext{2-D visualization} 
eq 	ext{original embedding space}}
\]

---

# 28. Critical Limitation — Polysemy

Consider:

> I deposited money in the bank.

> The boat reached the bank.

Traditional GloVe gives:

```text
bank → one vector
```

It does not automatically give a different vector for each occurrence.

This is a limitation of **static embeddings**.

---

# 29. Other Limitations

### Bias

Embeddings learn from human-generated text and may inherit bias.

### Rare words

Rare words provide less evidence for learning a reliable vector.

### Corpus dependence

Different corpora can produce different neighborhoods.

### Similarity ≠ meaning

A vector relationship is evidence from usage patterns, not a complete definition.

---

# 30. Simple Practical Workflow

```text
Pretrained GloVe / Word2Vec
          ↓
Choose words
          ↓
Extract vectors
          ↓
Cosine similarity
          ↓
Nearest neighbours
          ↓
Analogy experiment
          ↓
PCA / t-SNE
          ↓
2-D visualization
          ↓
Interpret results
          ↓
Discuss limitations
```

---

# 31. Final Comparison

| Representation | Main Question |
|---|---|
| BoW | Which words occur? |
| TF-IDF | Which words distinguish documents? |
| Word2Vec | What can local context tell us? |
| GloVe | What can global co-occurrence tell us? |
| PCA/t-SNE | Can we visualize the learned space? |

---

# 32. Final Discussion

Discuss:

### Question 1
Why is global co-occurrence useful for learning word representations?

### Question 2
How is GloVe different from Word2Vec?

### Question 3
Why can `doctor` and `hospital` be close even though they are not synonyms?

### Question 4
Why does PCA help with visualization?

### Question 5
Why can t-SNE plots be misleading?

### Question 6
Why does a static embedding struggle with `bank`?

### Question 7
Does GloVe truly understand language?

---

# 33. Final Takeaway

> **GloVe learns dense word vectors from global co-occurrence patterns.**

Those vectors can help us study:

- semantic similarity
- related words
- analogy relationships
- embedding-space structure

PCA and t-SNE help us inspect that high-dimensional space in 2-D.

But:

> **A useful embedding is not the same thing as human-like understanding.**

## Final question

> **If two words are close in an embedding space, what exactly has the model learned: meaning, context, or statistical patterns?**

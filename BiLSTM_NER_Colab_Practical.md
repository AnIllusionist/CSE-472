# Practical: Named Entity Recognition Using a BiLSTM

## 1. Aim

Implement and train a **Bidirectional Long Short-Term Memory (BiLSTM)** network for **Named Entity Recognition (NER)**. The model labels each word as one of the following:

- **Person** — a person's name
- **Location** — a place name
- **Organization** — a company, institution, or other organization
- **Other** — a word that is not part of a named entity

## 2. Learning outcomes

After completing this practical, students should be able to:

1. Explain Named Entity Recognition and sequence labelling.
2. Explain why a BiLSTM reads a sentence in both directions.
3. Convert words and labels into numerical IDs.
4. Train a word-level sequence-labelling model in PyTorch.
5. Use the trained model to predict entity labels for a sentence.

## 3. Platform and requirements

This practical is designed for **Google Colab**. It uses PyTorch and a small, template-based annotated dataset created inside the notebook. **No dataset download or file upload is required.** PyTorch is normally available in Colab.

If you use a local Python environment instead, install PyTorch with the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

> **Important note about the dataset:** The built-in dataset is a small educational dataset generated from simple sentence templates. It makes the practical easy to run, but it is not a replacement for the full CoNLL-2003 benchmark. A model trained on this data should not be expected to work reliably on arbitrary real-world text.

## 4. Background theory

### What is Named Entity Recognition?

Named Entity Recognition finds and classifies names in text. For example:

> `Ravi Kumar works at Google in Delhi.`

A correct word-level annotation could be:

| Word | Label | Meaning |
|---|---|---|
| Ravi | B-PER | Beginning of a person's name |
| Kumar | I-PER | Inside a person's name |
| works | O | Other / not an entity |
| at | O | Other / not an entity |
| Google | B-ORG | Beginning of an organization name |
| in | O | Other / not an entity |
| Delhi | B-LOC | Beginning of a location name |
| . | O | Other / not an entity |

### What do the BIO labels mean?

We use the common **BIO tagging scheme** so the model can mark the beginning and continuation of multi-word names:

| Tag | Meaning |
|---|---|
| `B-PER` | Beginning of a Person name |
| `I-PER` | A later word in the same Person name |
| `B-LOC` | Beginning of a Location name |
| `I-LOC` | A later word in the same Location name |
| `B-ORG` | Beginning of an Organization name |
| `I-ORG` | A later word in the same Organization name |
| `O` | Other; not part of a named entity |

The model learns **seven tags**, which are then displayed as the four requested categories: Person, Location, Organization, and Other.

### What is a BiLSTM?

An LSTM is a recurrent neural network that can learn information from sequences. A **Bidirectional LSTM** reads the sentence in two directions:

- Forward: from the first word to the last word.
- Backward: from the last word to the first word.

The two representations are combined to predict a tag for each word. This can help because a word's label often depends on both the words before and after it.

### Model pipeline

```text
Input sentence
      ↓
Tokenize into words
      ↓
Convert words into IDs
      ↓
Word embedding layer
      ↓
Bidirectional LSTM
      ↓
Linear classification layer
      ↓
One BIO tag per word
      ↓
Display Person / Location / Organization / Other
```

## 5. Implementation in Google Colab

Create a new notebook at [Google Colab](https://colab.research.google.com/) and run the following code cells **from top to bottom**.

### Cell 1 — Import libraries and set random seeds

```python
import random
import re

import torch
import torch.nn as nn
import torch.optim as optim

# Make results more repeatable across runs.
random.seed(42)
torch.manual_seed(42)

# Use a GPU if Colab provides one; otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("PyTorch version:", torch.__version__)
```

**Explanation:** `torch` provides tensor operations and neural-network tools. `random` is used for shuffling the examples, and `re` is used to split sentences into words and punctuation. The device selection automatically uses a Colab GPU when available.

### Cell 2 — Create a small annotated dataset

Each sentence is represented by a list of words and a list of tags. The two lists must have the same length: every word must have exactly one tag.

```python
# Small vocabulary of example entities for this classroom exercise.
PERSONS = [
    "Alice Johnson", "Bob Smith", "Maria Garcia", "John Miller",
    "Emma Wilson", "David Brown", "Sara Lee", "Ravi Kumar",
    "Priya Sharma", "Aman Verma", "Neha Singh", "Rahul Mehta"
]

LOCATIONS = [
    "London", "Paris", "New York", "Delhi", "Mumbai", "Tokyo",
    "Rome", "Berlin", "Sydney", "Toronto", "Phagwara", "Gwalior"
]

ORGANIZATIONS = [
    "Google", "Microsoft", "Amazon", "Apple", "OpenAI", "IBM",
    "Infosys", "TCS", "Meta", "Samsung", "LPU", "NASA"
]

# Tokenize words and punctuation separately. For example:
# "Ravi Kumar works at Google." -> ["Ravi", "Kumar", "works", "at", "Google", "."]
def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)


def make_example(parts):
    """Build one annotated sentence from (text, entity_type) parts.

    entity_type is PER, LOC, ORG, or O.
    Named entity words receive B-TYPE for the first word and I-TYPE
    for following words. Ordinary text receives O.
    """
    words = []
    tags = []

    for text, entity_type in parts:
        current_words = tokenize(text)

        if not current_words:
            continue

        if entity_type == "O":
            current_tags = ["O"] * len(current_words)
        else:
            current_tags = [f"B-{entity_type}"]
            current_tags += [f"I-{entity_type}"] * (len(current_words) - 1)

        words.extend(current_words)
        tags.extend(current_tags)

    assert len(words) == len(tags), "Each word must have one tag."
    return words, tags


EXAMPLES = []

# Generate examples with known annotations from simple sentence patterns.
# For instance, the words in a person's name are always tagged PER.
for i, person in enumerate(PERSONS):
    location_1 = LOCATIONS[i % len(LOCATIONS)]
    location_2 = LOCATIONS[(i + 3) % len(LOCATIONS)]
    organization_1 = ORGANIZATIONS[i % len(ORGANIZATIONS)]
    organization_2 = ORGANIZATIONS[(i + 5) % len(ORGANIZATIONS)]

    EXAMPLES.extend([
        make_example([(person, "PER"), ("works at", "O"), (organization_1, "ORG"), (".", "O")]),
        make_example([(person, "PER"), ("lives in", "O"), (location_1, "LOC"), (".", "O")]),
        make_example([(person, "PER"), ("traveled to", "O"), (location_2, "LOC"), ("yesterday .", "O")]),
        make_example([(person, "PER"), ("joined", "O"), (organization_2, "ORG"), ("in", "O"), (location_1, "LOC"), (".", "O")]),
        make_example([(organization_1, "ORG"), ("hired", "O"), (person, "PER"), (".", "O")]),
        make_example([(person, "PER"), ("met representatives of", "O"), (organization_1, "ORG"), ("in", "O"), (location_1, "LOC"), (".", "O")]),
    ])

# Add additional examples so organizations and locations appear in more contexts.
for i, organization in enumerate(ORGANIZATIONS):
    person = PERSONS[i % len(PERSONS)]
    location = LOCATIONS[(i + 2) % len(LOCATIONS)]
    another_organization = ORGANIZATIONS[(i + 1) % len(ORGANIZATIONS)]

    EXAMPLES.extend([
        make_example([(organization, "ORG"), ("opened an office in", "O"), (location, "LOC"), (".", "O")]),
        make_example([(organization, "ORG"), ("announced a partnership with", "O"), (another_organization, "ORG"), (".", "O")]),
        make_example([(organization, "ORG"), ("hired", "O"), (person, "PER"), ("from", "O"), (location, "LOC"), (".", "O")]),
    ])

for i, location in enumerate(LOCATIONS):
    person = PERSONS[(i + 4) % len(PERSONS)]
    organization = ORGANIZATIONS[(i + 3) % len(ORGANIZATIONS)]
    another_location = LOCATIONS[(i + 1) % len(LOCATIONS)]

    EXAMPLES.extend([
        make_example([("the meeting took place in", "O"), (location, "LOC"), (".", "O")]),
        make_example([(person, "PER"), ("traveled from", "O"), (location, "LOC"), ("to", "O"), (another_location, "LOC"), (".", "O")]),
        make_example([("representatives from", "O"), (organization, "ORG"), ("visited", "O"), (location, "LOC"), (".", "O")]),
    ])

# Shuffle examples and reserve some sentences for a simple test set.
random.shuffle(EXAMPLES)
split_index = int(0.80 * len(EXAMPLES))
TRAIN_DATA = EXAMPLES[:split_index]
TEST_DATA = EXAMPLES[split_index:]

print("Total examples:", len(EXAMPLES))
print("Training examples:", len(TRAIN_DATA))
print("Test examples:", len(TEST_DATA))

# Display one example to check that the words and labels align.
print("\nExample sentence:")
print(" ".join(TRAIN_DATA[0][0]))
print(list(zip(TRAIN_DATA[0][0], TRAIN_DATA[0][1])))
```

**Explanation:** `make_example()` assigns `B-` to the first word of an entity and `I-` to the remaining words. For example, `New York` becomes `B-LOC, I-LOC`. The ordinary words and punctuation receive `O`. The examples are split into training and test sets so that we can check the model on sentences it did not train on.

### Cell 3 — Build the word and tag dictionaries

A neural network cannot directly process text labels. We convert each word and each tag into an integer ID.

```python
TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

tag_to_ix = {tag: i for i, tag in enumerate(TAGS)}
ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}

# Build the word vocabulary using the training examples only.
word_to_ix = {"<PAD>": 0, "<UNK>": 1}

for words, _ in TRAIN_DATA:
    for word in words:
        word = word.lower()
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

PAD_IDX = word_to_ix["<PAD>"]
UNK_IDX = word_to_ix["<UNK>"]


def words_to_tensor(words):
    """Convert a list of words into a tensor of integer IDs."""
    ids = [word_to_ix.get(word.lower(), UNK_IDX) for word in words]
    return torch.tensor(ids, dtype=torch.long, device=device)


def tags_to_tensor(tags):
    """Convert a list of BIO tags into a tensor of integer IDs."""
    ids = [tag_to_ix[tag] for tag in tags]
    return torch.tensor(ids, dtype=torch.long, device=device)


print("Vocabulary size:", len(word_to_ix))
print("Number of tags:", len(TAGS))
print("Tag dictionary:", tag_to_ix)
```

**Explanation:** `word_to_ix` maps words to IDs, while `tag_to_ix` maps output labels to IDs. The `<UNK>` ID is used when a prediction sentence contains a word that was not present in the training vocabulary.

### Cell 4 — Define the BiLSTM model

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, number_of_tags):
        super().__init__()

        # Turn every word ID into a trainable vector.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # hidden_dim is the combined size of the forward and backward outputs.
        assert hidden_dim % 2 == 0, "hidden_dim must be an even number."
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, number_of_tags)

    def forward(self, input_ids):
        # input_ids shape: (batch_size, sequence_length)
        embedded_words = self.embedding(input_ids)
        # LSTM output shape: (batch_size, sequence_length, hidden_dim)
        lstm_output, _ = self.lstm(embedded_words)
        lstm_output = self.dropout(lstm_output)
        # Predict one tag score vector for each word.
        tag_scores = self.classifier(lstm_output)
        return tag_scores


EMBEDDING_DIM = 64
HIDDEN_DIM = 128

model = BiLSTMTagger(
    vocab_size=len(word_to_ix),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    number_of_tags=len(TAGS)
).to(device)

print(model)
```

**Explanation:**

- **Embedding layer:** represents each word as a vector of 64 numbers.
- **BiLSTM layer:** reads the sentence from left to right and right to left. Each direction contributes 64 features, giving 128 combined features per word.
- **Dropout:** randomly drops some features during training to reduce overfitting.
- **Linear layer:** converts the 128 features into seven tag scores. The tag with the highest score is the prediction.

### Cell 5 — Train the model

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 40

for epoch in range(EPOCHS):
    model.train()
    random.shuffle(TRAIN_DATA)
    total_loss = 0.0

    for words, tags in TRAIN_DATA:
        input_ids = words_to_tensor(words).unsqueeze(0)  # (1, sequence_length)
        target_ids = tags_to_tensor(tags)                # (sequence_length,)

        optimizer.zero_grad()

        scores = model(input_ids).squeeze(0)             # (sequence_length, number_of_tags)
        loss = criterion(scores, target_ids)

        loss.backward()
        # Avoid unusually large gradients.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(TRAIN_DATA)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {average_loss:.4f}")
```

**Explanation:** One epoch means the model has gone through every training sentence once. `CrossEntropyLoss` compares the predicted tag scores with the correct tags. The optimizer updates the model weights to reduce the loss. The loss will generally decrease as learning progresses, but a low training loss does not guarantee good performance on real-world text.

### Cell 6 — Evaluate token-level accuracy

```python
@torch.no_grad()
def predict_tags_for_words(words):
    """Predict one BIO tag for each supplied word."""
    model.eval()
    input_ids = words_to_tensor(words).unsqueeze(0)
    scores = model(input_ids).squeeze(0)
    predicted_ids = torch.argmax(scores, dim=-1).tolist()
    return [ix_to_tag[index] for index in predicted_ids]


@torch.no_grad()
def evaluate(data):
    model.eval()
    correct = 0
    total = 0

    for words, true_tags in data:
        predicted_tags = predict_tags_for_words(words)
        for predicted, actual in zip(predicted_tags, true_tags):
            correct += int(predicted == actual)
            total += 1

    return correct / total if total else 0.0


test_accuracy = evaluate(TEST_DATA)
print(f"Test token-level accuracy: {test_accuracy * 100:.2f}%")
```

**Explanation:** Token-level accuracy is the percentage of words whose predicted BIO tag exactly matches the correct tag. Since this dataset is small and template-based, the score is only a classroom demonstration and should not be treated as a benchmark result.

### Cell 7 — Predict entities in a new sentence

```python
TAG_TO_CATEGORY = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Organization"
}


def display_predictions(sentence):
    words = tokenize(sentence)
    predicted_tags = predict_tags_for_words(words)

    print(f"\nInput sentence: {sentence}\n")
    print(f"{'Word':<18}{'BIO tag':<12}{'Category'}")
    print("-" * 43)

    for word, tag in zip(words, predicted_tags):
        if tag == "O":
            category = "Other"
        else:
            entity_code = tag.split("-", maxsplit=1)[1]
            category = TAG_TO_CATEGORY[entity_code]

        print(f"{word:<18}{tag:<12}{category}")


display_predictions("Ravi Kumar works at Google in Delhi.")
display_predictions("Alice Johnson traveled to Paris yesterday.")
```

**Explanation:** The function tokenizes the input sentence, gets a predicted tag for every token, and displays both the BIO tag and its simpler category. The exact predictions may differ between runs and depend on what the model learned from the small dataset.

## 6. How to interpret the results

For a sentence such as `Ravi Kumar works at Google in Delhi.`, the desired output is approximately:

| Word | BIO tag | Category |
|---|---|---|
| Ravi | B-PER | Person |
| Kumar | I-PER | Person |
| works | O | Other |
| at | O | Other |
| Google | B-ORG | Organization |
| in | O | Other |
| Delhi | B-LOC | Location |
| . | O | Other |

The table above shows the **expected annotation**, not a guarantee of the model's output. The trained model may make mistakes, especially on words or sentence structures that are uncommon in the training examples.

## 7. Experiments for students

1. Change `EPOCHS` from `40` to `60`. Record the final training loss and test accuracy.
2. Change `EMBEDDING_DIM` from `64` to `32` or `100`. Does the result change?
3. Add a new person's name, location, or organization to the dataset. Add sentences that use it, then run the notebook again.
4. Enter a sentence with an entity that never appeared in the training data. Observe how the model behaves.
5. Compare unidirectional and bidirectional LSTMs by changing `bidirectional=True` to `False`. If you do this, also change the LSTM hidden size to `hidden_size=hidden_dim` so that the linear layer still receives `hidden_dim` features.

## 8. Limitations and extension to CoNLL-2003

This practical focuses on understanding the BiLSTM pipeline. The embedded data is small, repetitive, and generated from templates. Consequently:

- High accuracy on this dataset does not mean the model understands NER in general.
- A word unseen during training maps to `<UNK>`, which limits predictions for new names.
- Real NER data contains ambiguous words, many sentence styles, and more diverse entities.

For a more realistic experiment, replace the mini dataset with the **CoNLL-2003 NER dataset** and convert its PER, LOC, ORG, and miscellaneous labels into the desired tag scheme. CoNLL-2003 commonly uses BIO-style tags and includes an additional miscellaneous category; this practical concentrates on Person, Location, Organization, and Other as requested.

## 9. Conclusion

In this practical, we built a BiLSTM sequence-labelling model with a word embedding layer, a bidirectional LSTM, and a linear classifier. The model learned to assign a BIO tag to every token and converted these tags into Person, Location, Organization, or Other. The same workflow can be extended to larger annotated datasets for more realistic NER experiments.

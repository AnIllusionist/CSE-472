# CSE472 -- NLP Practical 1

# From Human Text to Machine-Ready Text

### A Beginner-Friendly Practical on Text Preprocessing, Rule-Based NLP, and Statistical NLP

> **Teaching goal:** By the end of this practical, a student who has
> never worked with NLP should be able to take a piece of raw human
> text, process it step by step, and understand why modern NLP moved
> from hand-written rules toward learning patterns from data.

------------------------------------------------------------------------

## 1. What are we doing today?

Imagine giving a computer this review:

> **"OMG!!! I absolutely LOVED this movie 😍😍 #BestMovie2026"**

A human immediately understands:

-   Someone is excited.
-   They liked the movie.
-   The emojis strengthen the positive feeling.
-   `OMG`, `!!!`, and the hashtag carry information.

But a computer initially sees only a sequence of characters.

So our practical question is:

> **How do we turn messy human language into something a computer can
> work with?**

This practical follows this journey:

``` text
RAW HUMAN TEXT
      ↓
Normalization
      ↓
Tokenization
      ↓
Punctuation / Stop-word decisions
      ↓
POS Tagging
      ↓
Stemming / Lemmatization
      ↓
Numerical Representation
      ↓
Statistical NLP Model
      ↓
Prediction / Decision
```

The lecture emphasizes that preprocessing is not simply "cleaning text";
it is deciding which information a downstream model should see and which
information can be treated as noise. fileciteturn3file0L747-L800

------------------------------------------------------------------------

# 2. Learning objectives

After completing this practical, students should be able to:

1.  Explain why raw text is difficult for computers.
2.  Install and import basic NLP libraries.
3.  Normalize text.
4.  Tokenize text.
5.  Remove punctuation when appropriate.
6.  Understand stop words.
7.  See why blindly removing stop words can be dangerous.
8.  Perform POS tagging.
9.  Perform stemming.
10. Perform lemmatization.
11. Compare stemming and lemmatization.
12. Understand what "rule-based NLP" means through code.
13. Demonstrate practically why hand-written rules fail.
14. Understand the basic idea of statistical NLP.
15. Convert text into numbers using Bag-of-Words.
16. Train a simple statistical NLP classifier.
17. Compare a hand-written rule with a learned statistical model.
18. Understand why preprocessing depends on the task.

------------------------------------------------------------------------

# 3. Before we start: What does the computer actually see?

Consider:

``` text
OMG!!! I absolutely LOVED this movie 😍😍
```

Humans see:

``` text
Excitement + positive opinion + movie review
```

A program initially sees:

``` text
"O"
"M"
"G"
"!"
"!"
"!"
" "
"I"
...
```

The computer does **not automatically know** that:

-   `LOVED` and `love` are related.
-   `!!!` may indicate strong emotion.
-   😍 may be useful for sentiment.
-   `not good` is different from `good`.
-   `Apple` could mean a company while `apple` could mean a fruit.

The practical is about gradually adding useful structure.

------------------------------------------------------------------------

# 4. Software setup

We will use Python.

## Step 1 -- Check Python

Open a terminal / command prompt and run:

``` bash
python --version
```

or:

``` bash
python3 --version
```

You should see something similar to:

``` text
Python 3.x.x
```

------------------------------------------------------------------------

## Step 2 -- Create a virtual environment

### Windows

``` bash
python -m venv nlp_env
nlp_env\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv nlp_env
source nlp_env/bin/activate
```

You should now see something like:

``` text
(nlp_env)
```

at the beginning of your terminal line.

------------------------------------------------------------------------

## Step 3 -- Install libraries

Run:

``` bash
pip install nltk scikit-learn pandas
```

We will use:

  Library          Why?
  ---------------- -----------------------------------------------
  `nltk`           Basic NLP operations
  `scikit-learn`   Numerical representation and machine learning
  `pandas`         Optional: viewing small datasets

------------------------------------------------------------------------

# 5. Start with the simplest possible program

Create a Python file:

``` text
nlp_practical.py
```

Start with:

``` python
text = "OMG!!! I absolutely LOVED this movie 😍😍"

print(text)
```

Run:

``` bash
python nlp_practical.py
```

Expected output:

``` text
OMG!!! I absolutely LOVED this movie 😍😍
```

### Why did we start here?

Because we should not jump directly into a library.

First understand:

> **Our input is simply raw text.**

Everything else in this practical transforms this text.

------------------------------------------------------------------------

# 6. Practical 1 -- Normalization

## 6.1 What is normalization?

Normalization means making different forms of text more consistent.

For example:

``` text
HELLO
Hello
hello
```

can sometimes be treated as the same word:

``` text
hello
```

The lecture demonstrates this idea using `HELLO`, `Hello`, `hello`, and
`hello!!!`. fileciteturn3file0L404-L430

------------------------------------------------------------------------

## 6.2 Lowercasing

Add:

``` python
text = "OMG!!! I absolutely LOVED this movie 😍😍"

normalized_text = text.lower()

print(normalized_text)
```

Output:

``` text
omg!!! i absolutely loved this movie 😍😍
```

### What happened?

We converted:

``` text
OMG → omg
LOVED → loved
I → i
```

------------------------------------------------------------------------

## 6.3 Why might this be useful?

Suppose our dataset contains:

``` text
Good
good
GOOD
Good!!!
```

A computer may initially treat these as different strings.

Lowercasing can reduce unnecessary variation.

But there is an important warning.

------------------------------------------------------------------------

# 7. ⚠️ Normalization is NOT always safe

Consider:

``` text
Apple
apple
```

Are they always the same?

No.

For example:

``` text
Apple announced a new product.
```

could refer to the company.

While:

``` text
I ate an apple.
```

refers to the fruit.

The lecture explicitly uses this example to show that aggressive
normalization can destroy useful information such as named entities.
fileciteturn3file0L432-L456

### Important principle

> **Never normalize just because you can. Normalize because your task
> requires it.**

------------------------------------------------------------------------

# 8. Practical 2 -- Tokenization

## 8.1 What is tokenization?

Tokenization means breaking text into smaller units called **tokens**.

Example:

``` text
"I love NLP!"
```

can become:

``` text
["I", "love", "NLP", "!"]
```

The lecture defines tokenization as breaking text into smaller
computational units called tokens. fileciteturn3file0L273-L290

------------------------------------------------------------------------

## 8.2 Download the NLTK resources

Add this once:

``` python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

------------------------------------------------------------------------

## 8.3 Word tokenization

``` python
from nltk.tokenize import word_tokenize

text = "I love NLP!"

tokens = word_tokenize(text)

print(tokens)
```

Expected:

``` text
['I', 'love', 'NLP', '!']
```

### Think of it like this

Before:

``` text
"I love NLP!"
```

After:

``` text
["I", "love", "NLP", "!"]
```

Now the computer can work with individual units.

------------------------------------------------------------------------

# 9. Tokenization is harder than `split()`

A beginner may try:

``` python
text.split()
```

For:

``` text
I love NLP!
```

we get:

``` text
['I', 'love', 'NLP!']
```

Notice:

``` text
NLP!
```

is still one item.

A proper tokenizer can separate:

``` text
NLP
!
```

This is why tokenization is not simply "split wherever there is a
space."

The lecture gives difficult examples such as:

``` text
I'm learning.
New Delhi
AI-powered
₹10,000
can't
```

and shows that rule-based splitting can break easily.
fileciteturn3file0L293-L326

------------------------------------------------------------------------

# 10. Student Challenge -- Tokenization Battle ⚔️

Before running code, ask students:

### How should this be tokenized?

``` text
ChatGPT is amazing!!! #AI2026
```

Possible answers:

``` text
A:
["ChatGPT", "is", "amazing", "!", "!", "!", "#AI2026"]

B:
["ChatGPT", "is", "amazing!!!", "#AI2026"]

C:
["ChatGPT", "is", "amazing", "#", "AI", "2026"]
```

There is no universally correct answer.

It depends on:

-   the tokenizer,
-   the vocabulary,
-   the model,
-   and the task.

Modern Transformer systems often use **subword tokenization** rather
than treating every human-visible word as one token.
fileciteturn3file0L344-L366

------------------------------------------------------------------------

# 11. Practical 3 -- Removing punctuation

Suppose:

``` python
text = "I love NLP!!!"
```

We can inspect punctuation using:

``` python
import string

print(string.punctuation)
```

You will see characters such as:

``` text
!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

------------------------------------------------------------------------

## Remove punctuation

``` python
text = "I love NLP!!!"

clean_text = text.translate(
    str.maketrans("", "", string.punctuation)
)

print(clean_text)
```

Output:

``` text
I love NLP
```

------------------------------------------------------------------------

# 12. ⚠️ Should we ALWAYS remove punctuation?

No.

Consider:

``` text
Let's eat, Grandma.
```

versus:

``` text
Let's eat Grandma.
```

The comma changes the interpretation dramatically.

The lecture uses this example to emphasize that punctuation decisions
depend on the downstream task. fileciteturn3file0L389-L402

For sentiment analysis:

``` text
Amazing!!!
```

The `!!!` may carry emotional information.

Therefore:

> **Cleaning is not automatically good.**

------------------------------------------------------------------------

# 13. Practical 4 -- Stop Words

## 13.1 What are stop words?

Stop words are very common words that are sometimes removed because they
contribute little to a particular task.

Examples:

``` text
the
is
a
an
of
in
on
at
```

The lecture illustrates a sentence such as:

``` text
The movie was really very good.
```

being reduced to:

``` text
["movie", "good"]
```

for a task where those connector words may not be useful.
fileciteturn3file0L458-L476

------------------------------------------------------------------------

## 13.2 Download stop-word data

``` python
import nltk

nltk.download("stopwords")
```

------------------------------------------------------------------------

## 13.3 See the stop-word list

``` python
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

print(list(stop_words)[:20])
```

You may see words such as:

``` text
the
is
a
an
of
to
and
```

------------------------------------------------------------------------

## 13.4 Remove stop words

``` python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "The movie was really very good."

tokens = word_tokenize(text)

stop_words = set(stopwords.words("english"))

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

print(filtered_tokens)
```

Possible output:

``` text
['movie', 'really', 'good', '.']
```

The exact output depends on the stop-word list and tokenizer.

------------------------------------------------------------------------

# 14. 🚨 The "NOT" Experiment

This is one of the most important practical demonstrations.

Consider:

``` text
The movie was NOT good.
```

A naive stop-word removal process may remove:

``` text
not
```

and leave:

``` text
movie good
```

What just happened?

The original meaning was:

``` text
NEGATIVE
```

The processed sentence now looks:

``` text
POSITIVE
```

The lecture explicitly demonstrates this failure and warns that
preprocessing must match the downstream task.
fileciteturn3file0L478-L524

------------------------------------------------------------------------

## Try it yourself

``` python
text = "The movie was not good."

tokens = word_tokenize(text)

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("Original:", tokens)
print("After stop-word removal:", filtered_tokens)
```

### Ask students:

> **Should we remove `not` in sentiment analysis?**

Answer:

> Usually, **no**. Negation is extremely important for sentiment.

------------------------------------------------------------------------

# 15. Practical 5 -- POS Tagging

## 15.1 What is POS tagging?

POS = **Part of Speech**

It tells us the grammatical role of a word.

For example:

``` text
The movie was good.
```

can be tagged approximately as:

``` text
The     → DET
movie   → NOUN
was     → VERB
good    → ADJ
```

The practical lecture uses this exact type of example.
fileciteturn3file0L527-L547

------------------------------------------------------------------------

## 15.2 Download the NLTK resources

Depending on your NLTK version:

``` python
import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
```

------------------------------------------------------------------------

## 15.3 Run POS tagging

``` python
from nltk.tokenize import word_tokenize
import nltk

text = "The movie was good."

tokens = word_tokenize(text)

pos_tags = nltk.pos_tag(tokens)

print(pos_tags)
```

Possible output:

``` text
[
    ('The', 'DT'),
    ('movie', 'NN'),
    ('was', 'VBD'),
    ('good', 'JJ'),
    ('.', '.')
]
```

Don't worry if the exact tags look strange initially.

The important idea is:

``` text
WORD → GRAMMATICAL ROLE
```

------------------------------------------------------------------------

# 16. Why does POS tagging matter?

Consider:

``` text
I need a permit.
```

Here:

``` text
permit = NOUN
```

Now:

``` text
They permit students to enter.
```

Here:

``` text
permit = VERB
```

Same spelling.

Different grammatical role.

The lecture uses `permit` to show why POS information gives a model
additional context. fileciteturn3file0L527-L545

------------------------------------------------------------------------

# 17. Practical 6 -- Stemming

## 17.1 The problem

Consider:

``` text
play
plays
played
playing
```

They are different strings but related to the same basic concept.

Stemming tries to reduce related forms to a common **stem**.

------------------------------------------------------------------------

## 17.2 Run a stemmer

``` python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = [
    "play",
    "plays",
    "played",
    "playing",
    "studies",
    "relational"
]

for word in words:
    print(word, "->", stemmer.stem(word))
```

You may get results such as:

``` text
play -> play
plays -> play
played -> play
playing -> play
studies -> studi
relational -> relat
```

------------------------------------------------------------------------

# 18. 🚨 Stemming can produce nonsense

Look at:

``` text
studies → studi
```

Is `studi` a normal English word?

No.

Why did this happen?

Because a stemmer generally uses **chopping rules**.

It is not trying to understand the meaning of the word.

The lecture explicitly demonstrates `studies → studi` and explains that
blind chopping can produce non-dictionary fragments.
fileciteturn3file0L568-L617

### Remember:

> **Stemming is fast, but it can be crude.**

------------------------------------------------------------------------

# 19. Practical 7 -- Lemmatization

Lemmatization is more linguistically informed.

Instead of simply chopping letters, it attempts to return a valid
dictionary base form called the **lemma**.

Examples:

``` text
am → be
is → be
are → be
was → be
```

and:

``` text
ran → run
running → run
runs → run
```

The lecture contrasts lemmatization with stemming and notes that
lemmatization uses linguistic information such as vocabulary and POS.
fileciteturn3file0L619-L674

------------------------------------------------------------------------

## 19.1 Download WordNet

``` python
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
```

------------------------------------------------------------------------

## 19.2 Basic lemmatization

``` python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = [
    "cats",
    "dogs",
    "running",
    "played"
]

for word in words:
    print(word, "->", lemmatizer.lemmatize(word))
```

You may notice that this does **not always give the result you expect**.

For example, without POS information:

``` text
running
```

may remain:

``` text
running
```

Why?

Because the lemmatizer does not automatically know whether the word is a
noun, verb, adjective, etc.

------------------------------------------------------------------------

# 20. Lemmatization with POS

Let's provide grammatical information.

``` python
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running", pos="v"))
print(lemmatizer.lemmatize("ran", pos="v"))
print(lemmatizer.lemmatize("better", pos="a"))
```

Possible output:

``` text
run
run
good
```

This demonstrates an important idea:

> **Context and grammatical role can change the correct base form.**

------------------------------------------------------------------------

# 21. Stemming vs Lemmatization

  -----------------------------------------------------------------------
  Stemming                            Lemmatization
  ----------------------------------- -----------------------------------
  Faster                              Usually slower

  Uses chopping rules                 Uses linguistic information

  Can produce non-words               Usually produces valid words

  Less context-aware                  More context-aware

  Good for some simple tasks          Useful when linguistic correctness
                                      matters
  -----------------------------------------------------------------------

This distinction is directly reflected in the practical lecture.
fileciteturn2file4L777-L802

------------------------------------------------------------------------

# 22. A very important discovery

Now run:

``` python
sentences = [
    "The company is running.",
    "He is running.",
    "The machine is running."
]
```

The word:

``` text
running
```

appears in all three.

But the meaning is different:

``` text
company → operating
person → sprinting
machine → functioning
```

The lecture uses this example to emphasize that preprocessing can
simplify text but does not magically solve meaning.
fileciteturn3file0L678-L693

This is the point where we transition from:

> **Text preprocessing**

to:

> **Learning patterns from data.**

------------------------------------------------------------------------

# 23. NOW: Why did Rule-Based NLP fail?

This is the most important conceptual experiment of today's practical.

Before machine learning became dominant, one major approach was:

> **Write rules ourselves.**

For example:

``` text
IF a word ends with "ed"
THEN classify it as past tense.
```

Or for sentiment:

``` text
IF sentence contains "good"
THEN positive
```

Or:

``` text
IF sentence contains "bad"
THEN negative
```

The lecture describes the historical shift from explicit rules toward
learning patterns from data and identifies too many exceptions,
ambiguity, changing context, slang, and different domains as major
problems for hand-written rules. fileciteturn2file0L108-L146

Let's prove it practically.

------------------------------------------------------------------------

# 24. Practical 8 -- Build a Rule-Based Sentiment Analyzer

Create:

``` python
positive_words = [
    "good",
    "great",
    "amazing",
    "excellent",
    "love",
    "loved"
]

negative_words = [
    "bad",
    "terrible",
    "awful",
    "hate",
    "hated"
]
```

Now create a function:

``` python
def rule_based_sentiment(text):

    words = text.lower().split()

    positive_count = 0
    negative_count = 0

    for word in words:

        if word in positive_words:
            positive_count += 1

        if word in negative_words:
            negative_count += 1

    if positive_count > negative_count:
        return "Positive"

    elif negative_count > positive_count:
        return "Negative"

    else:
        return "Neutral"
```

Test it:

``` python
print(rule_based_sentiment("This movie was amazing"))
print(rule_based_sentiment("This movie was terrible"))
print(rule_based_sentiment("This movie was okay"))
```

Expected:

``` text
Positive
Negative
Neutral
```

So...

### Did our rule-based system work?

Yes!

For simple sentences, it works surprisingly well.

Now let's break it.

------------------------------------------------------------------------

# 25. Rule-Based Failure #1 -- Negation

Try:

``` python
print(rule_based_sentiment("This movie was not good"))
```

What might our system see?

``` text
not
good
```

It finds:

``` text
good → positive
```

So it may predict:

``` text
Positive
```

But a human says:

``` text
Negative
```

### Why did it fail?

Because our rule only looked for words.

It didn't understand:

``` text
NOT + GOOD
```

------------------------------------------------------------------------

# 26. Rule-Based Failure #2 -- Sarcasm

Try:

``` python
print(
    rule_based_sentiment(
        "Great, another broken phone. Just what I needed."
    )
)
```

The word:

``` text
Great
```

is positive.

But the overall meaning is negative/sarcastic.

Our rule may predict:

``` text
Positive
```

A human understands:

``` text
Negative / sarcastic
```

### Why?

The rule doesn't understand tone or context.

This connects directly to the first NLP lecture's discussion of sarcasm
and contextual meaning.

------------------------------------------------------------------------

# 27. Rule-Based Failure #3 -- Slang

Try:

``` python
print(rule_based_sentiment("That movie was sick!"))
```

In some contexts:

``` text
sick = negative
```

But in informal language:

``` text
sick = amazing / impressive
```

Our dictionary doesn't know this.

------------------------------------------------------------------------

# 28. Rule-Based Failure #4 -- Typos

Try:

``` python
print(rule_based_sentiment("This movie was amazng"))
```

Humans can easily recognize:

``` text
amazng ≈ amazing
```

Our rule sees:

``` text
amazng
```

It is not in the dictionary.

Prediction:

``` text
Neutral
```

Human interpretation:

``` text
Positive
```

------------------------------------------------------------------------

# 29. Rule-Based Failure #5 -- New Vocabulary

Imagine students start using a new slang term tomorrow:

``` text
"That movie was fire."
```

Our rules don't know what `fire` means in this context.

Someone must manually update:

``` python
positive_words = [...]
```

This is exactly the scalability problem with hand-written rules.

------------------------------------------------------------------------

# 30. Rule-Based Failure #6 -- Context

Consider:

``` text
The battery is dead.
```

and:

``` text
The battery is not dead.
```

A keyword rule may focus on:

``` text
dead
```

and miss the effect of:

``` text
not
```

Now consider:

``` text
The phone is sick.
```

Is the phone medically ill?

Obviously not.

The meaning depends on context.

------------------------------------------------------------------------

# 31. The Rule Explosion Problem

Suppose we try to fix every failure manually.

We might start writing rules such as:

``` text
IF "not" appears before "good"
THEN negative

IF "not" appears before "great"
THEN negative

IF "not" appears before "excellent"
THEN negative

IF "sick" appears in a movie review
THEN positive

IF "sick" appears in a medical document
THEN negative

IF "fire" appears in youth slang
THEN positive

IF "fire" appears in an emergency report
THEN negative
```

And then we discover:

``` text
"not very good"
"not exactly great"
"hardly good"
"never good"
"great... just great 🙄"
```

The number of rules keeps growing.

### This is the core problem.

> **Human language has too many combinations and exceptions for us to
> manually write every possible rule.**

The lecture summarizes this as too many exceptions, changing context,
new slang, and many domains. fileciteturn2file0L139-L146

------------------------------------------------------------------------

# 32. 🧠 Classroom Experiment: Rule Explosion

Ask students:

> "If I give you 100,000 customer reviews, how many rules would you
> write?"

Then give:

``` text
"I loved this phone."

"I LOVE this phone!!! 😍"

"This phone is fire."

"This phone is sick."

"This phone isn't bad."

"Not exactly amazing."

"Camera is great but battery is terrible."

"Great... another update that broke everything."

"Best phone ever 😂"

"bruh this phone 💀"
```

Ask:

> **"Can you realistically write a rule for every possible
> expression?"**

The expected answer is:

> **No.**

Now introduce:

# Statistical NLP

------------------------------------------------------------------------

# 33. What is Statistical NLP?

Very simply:

> **Instead of telling the computer every rule, we give it examples and
> let it learn patterns from the data.**

Think of two students learning a language.

### Student A -- Rule memorization

The teacher says:

``` text
Memorize these 10,000 grammar rules.
```

### Student B -- Exposure to examples

The teacher gives:

``` text
millions of sentences
```

and asks the student to discover patterns.

Statistical NLP is closer to **Student B**.

The historical NLP progression in the lecture explicitly shows the shift
from rule-based NLP to statistical NLP and then machine learning/deep
learning. fileciteturn2file0L108-L130

------------------------------------------------------------------------

# 34. But what does "statistical" actually mean?

Students often hear:

> "Statistical NLP"

and think:

> "So it is just statistics?"

The basic idea is:

``` text
Look at examples
      ↓
Count patterns
      ↓
Estimate probabilities
      ↓
Use those probabilities to make predictions
```

For example, suppose a dataset contains:

``` text
I love this movie
I loved this movie
This movie is amazing
This movie is excellent
I hate this movie
This movie is terrible
```

A statistical system can discover that:

``` text
love / loved / amazing / excellent
```

often occur in positive examples.

And:

``` text
hate / terrible
```

often occur in negative examples.

Instead of us manually writing:

``` text
IF "amazing" THEN positive
```

the system **learns from examples**.

------------------------------------------------------------------------

# 35. Practical 9 -- Our First Statistical NLP Model

We will build a tiny sentiment classifier.

Our pipeline:

``` text
Training sentences
       ↓
Convert text into numbers
       ↓
Learn patterns
       ↓
Give new sentence
       ↓
Predict sentiment
```

------------------------------------------------------------------------

# 36. Step 1 -- Create training data

``` python
texts = [
    "I love this movie",
    "This movie is amazing",
    "What a great experience",
    "I really enjoyed this film",
    "This is excellent",
    "I hate this movie",
    "This movie is terrible",
    "What an awful experience",
    "I really disliked this film",
    "This is horrible"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative"
]
```

Notice something important.

We are no longer writing a rule saying:

``` text
IF love → positive
```

We are providing **examples**.

------------------------------------------------------------------------

# 37. Step 2 -- Convert words into numbers

A machine-learning model cannot directly perform arithmetic on:

``` text
"I love this movie"
```

We need numbers.

One simple method is:

# Bag-of-Words

Import:

``` python
from sklearn.feature_extraction.text import CountVectorizer
```

Create the vectorizer:

``` python
vectorizer = CountVectorizer()
```

Fit it on our training text:

``` python
X = vectorizer.fit_transform(texts)
```

------------------------------------------------------------------------

# 38. What just happened?

Suppose our vocabulary contains:

``` text
love
movie
amazing
great
hate
terrible
```

A sentence such as:

``` text
I love this movie
```

can become something conceptually like:

``` text
[1, 1, 0, 0, 0, 0]
```

The exact columns depend on the learned vocabulary.

The important idea is:

``` text
TEXT → NUMBERS
```

This is the bridge between language and machine learning.

------------------------------------------------------------------------

# 39. Step 3 -- See the vocabulary

Run:

``` python
print(vectorizer.get_feature_names_out())
```

You will see words learned from the training data.

For example:

``` text
['amazing' 'awful' 'excellent' 'experience' 'film'
 'great' 'hate' 'horrible' 'love' ...]
```

The computer has created a vocabulary.

------------------------------------------------------------------------

# 40. Step 4 -- Inspect the numbers

Run:

``` python
print(X.toarray())
```

You will see a matrix of zeros and ones/counts.

For example:

``` text
[[0 0 0 0 0 0 0 0 ...]
 [1 0 0 0 0 0 0 0 ...]
 ...
]
```

Don't worry about every number.

Remember:

> **The sentence has been converted into numerical features.**

------------------------------------------------------------------------

# 41. Step 5 -- Train a statistical classifier

We will use a simple **Naive Bayes** classifier.

Import:

``` python
from sklearn.naive_bayes import MultinomialNB
```

Create the model:

``` python
model = MultinomialNB()
```

Train it:

``` python
model.fit(X, labels)
```

That's it.

The model has now learned patterns from our examples.

------------------------------------------------------------------------

# 42. Step 6 -- Test a new sentence

Let's give it a sentence it has never seen:

``` python
new_text = ["I really love this film"]

new_X = vectorizer.transform(new_text)

prediction = model.predict(new_X)

print(prediction)
```

Possible output:

``` text
['positive']
```

Notice:

We never wrote:

``` python
if "love" in sentence:
    positive
```

The model learned from examples.

------------------------------------------------------------------------

# 43. Test multiple sentences

Try:

``` python
test_sentences = [
    "This movie was fantastic",
    "I hated this film",
    "What an awful movie",
    "I enjoyed this experience",
    "This was horrible"
]

test_X = vectorizer.transform(test_sentences)

predictions = model.predict(test_X)

for sentence, prediction in zip(test_sentences, predictions):
    print(sentence, "->", prediction)
```

You might get:

``` text
This movie was fantastic -> positive
I hated this film -> negative
What an awful movie -> negative
I enjoyed this experience -> positive
This was horrible -> negative
```

The exact behavior depends on the training data.

------------------------------------------------------------------------

# 44. The Big Difference: Rules vs Statistics

## Rule-Based NLP

We tell the computer:

``` text
IF "good" → positive
IF "great" → positive
IF "bad" → negative
IF "terrible" → negative
```

The human writes the rules.

------------------------------------------------------------------------

## Statistical NLP

We give examples:

``` text
"This movie is great" → positive
"This movie is amazing" → positive
"This movie is terrible" → negative
"This movie is awful" → negative
```

The algorithm learns statistical patterns.

------------------------------------------------------------------------

# 45. Side-by-Side Experiment

Let's compare.

### Rule-based:

``` python
print(rule_based_sentiment(
    "Great, another broken phone. Just what I needed."
))
```

Possible result:

``` text
Positive
```

Human interpretation:

``` text
Negative / sarcastic
```

------------------------------------------------------------------------

### Statistical model:

The model may also fail.

This is important!

> **Statistical NLP is not magic.**

If the training data does not contain enough examples of sarcasm, the
model may also misunderstand it.

------------------------------------------------------------------------

# 46. So why was Statistical NLP better?

Because instead of requiring us to manually specify every pattern:

``` text
good
great
amazing
excellent
...
```

the model can discover associations from many examples.

This makes it:

-   more scalable,
-   more adaptable,
-   less dependent on manually written rules,
-   capable of learning patterns that humans did not explicitly program.

But it still has limitations.

------------------------------------------------------------------------

# 47. Statistical NLP can also fail

Consider:

``` text
"Great... another phone update that broke everything."
```

A simple model may see:

``` text
great
```

and predict:

``` text
positive
```

But the real meaning is negative.

Why?

Because simple statistical models often rely heavily on word-frequency
patterns and may not capture complex context.

This leads naturally to later developments:

``` text
Rule-Based NLP
       ↓
Statistical NLP
       ↓
Machine Learning
       ↓
Deep Learning
       ↓
RNN / LSTM
       ↓
Attention
       ↓
Transformers
       ↓
LLMs
```

The lecture's NLP journey follows this progression toward word
representations, machine learning, RNN/LSTM, attention, Transformers and
LLMs. fileciteturn2file2L644-L659

------------------------------------------------------------------------

# 48. Complete Practical Code

Once students understand each individual step, they can combine
everything.

``` python
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ------------------------------------------------
# STEP 1: DOWNLOAD REQUIRED NLTK RESOURCES
# ------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


# ------------------------------------------------
# STEP 2: RAW TEXT
# ------------------------------------------------

text = "OMG!!! I absolutely LOVED this movie 😍😍"


# ------------------------------------------------
# STEP 3: NORMALIZATION
# ------------------------------------------------

normalized_text = text.lower()

print("\nNORMALIZED TEXT:")
print(normalized_text)


# ------------------------------------------------
# STEP 4: TOKENIZATION
# ------------------------------------------------

tokens = word_tokenize(normalized_text)

print("\nTOKENS:")
print(tokens)


# ------------------------------------------------
# STEP 5: PUNCTUATION REMOVAL
# ------------------------------------------------

tokens_without_punctuation = [
    token
    for token in tokens
    if token not in string.punctuation
]

print("\nWITHOUT PUNCTUATION:")
print(tokens_without_punctuation)


# ------------------------------------------------
# STEP 6: STOP-WORD REMOVAL
# ------------------------------------------------

stop_words = set(stopwords.words("english"))

filtered_tokens = [
    token
    for token in tokens_without_punctuation
    if token not in stop_words
]

print("\nWITHOUT STOP WORDS:")
print(filtered_tokens)


# ------------------------------------------------
# STEP 7: POS TAGGING
# ------------------------------------------------

pos_tags = nltk.pos_tag(tokens_without_punctuation)

print("\nPOS TAGS:")
print(pos_tags)


# ------------------------------------------------
# STEP 8: STEMMING
# ------------------------------------------------

stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(token)
    for token in filtered_tokens
]

print("\nSTEMMED:")
print(stemmed_words)


# ------------------------------------------------
# STEP 9: LEMMATIZATION
# ------------------------------------------------

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(token)
    for token in filtered_tokens
]

print("\nLEMMATIZED:")
print(lemmatized_words)
```

------------------------------------------------------------------------

# 49. Important Warning About the "Complete Pipeline"

Do **not** tell students:

> "Every NLP project must do all these steps."

That is incorrect.

The practical lecture makes this point very clearly:

> **There is no perfect preprocessing pipeline.**

For example:

### Sentiment Analysis

May want to keep:

``` text
not
!!!
emojis
```

### Search Engine

May benefit from:

``` text
lowercasing
stop-word removal
stemming
```

### Modern LLM

Usually uses:

``` text
subword tokenization
```

and does not simply apply an old-fashioned "remove all stop words"
pipeline. fileciteturn3file0L747-L770

Therefore:

> **Preprocessing depends on the task.**

------------------------------------------------------------------------

# 50. Practical Example A -- Sentiment Analysis

Input:

``` text
"I LOVE this phone!!! 😍"
```

Possible processing:

``` text
Raw
↓
Normalize carefully
↓
Tokenize
↓
Keep "LOVE"
↓
Keep "!!!"
↓
Keep emoji
↓
Sentiment model
↓
POSITIVE
```

Why keep `!!!` and 😍?

Because they carry sentiment information.

------------------------------------------------------------------------

# 51. Practical Example B -- Search Engine

Query:

``` text
"best restaurants in Delhi"
```

For some traditional search pipelines, we might simplify:

``` text
best restaurants Delhi
```

Why?

Words such as:

``` text
in
```

may contribute less to keyword matching.

But again, this is task-dependent.

------------------------------------------------------------------------

# 52. Practical Example C -- Spam Detection

Input:

``` text
"URGENT!!! You have WON ₹50,00,000!!! CLICK NOW!!!"
```

Useful information may include:

``` text
URGENT
WON
money
CLICK
!!!
```

Notice something interesting:

For spam detection, punctuation and capitalization may actually be
useful features.

So blindly doing:

``` text
lowercase everything
remove punctuation
remove everything "unimportant"
```

could throw away useful signals.

------------------------------------------------------------------------

# 53. Practical Example D -- Movie Review

Input:

``` text
"The camera is amazing, but the battery life is terrible."
```

A good system should detect:

``` text
Camera → positive
Battery → negative
Overall review → mixed
```

This is more difficult than simply counting:

``` text
amazing
terrible
```

because the system needs to connect sentiment to the correct aspect.

The lecture uses this type of review to demonstrate how NLP can turn
text into customer insights and business decisions.
fileciteturn2file0L205-L227

------------------------------------------------------------------------

# 54. Practical Example E -- Hinglish

Try:

``` text
text = "Yaar movie bahut mast thi!"
```

Ask:

> Will an English-only pipeline necessarily understand this correctly?

Not necessarily.

This introduces:

-   multilingual NLP,
-   code-switching,
-   Hinglish,
-   language-specific tokenization,
-   language-specific morphology.

The lecture specifically notes that English, Hinglish and Hindi cannot
always use identical preprocessing rules, particularly because Indian
languages can be morphologically rich. fileciteturn3file0L695-L720

------------------------------------------------------------------------

# 55. Mini Challenge -- Students Design the Pipeline

Give students:

``` text
"I LOVE this phone!!! The camera is AMAZING 😍,
but the battery isn't good. #Review2026"
```

Give them 5 minutes in pairs.

Ask them:

### Question 1

What will you tokenize?

### Question 2

Will you lowercase everything?

### Question 3

Will you remove `!!!`?

### Question 4

Will you remove 😍?

### Question 5

Will you remove `not` / `isn't`?

### Question 6

Will you stem or lemmatize?

### Question 7

Which information would be dangerous to throw away?

The practical lecture uses essentially this kind of pair activity.
fileciteturn3file0L723-L745

------------------------------------------------------------------------

# 56. A Very Important Mental Model

Students should leave today's practical remembering this:

``` text
                 HUMAN LANGUAGE
                       │
                       ▼
                 RAW TEXT
                       │
                       ▼
                PREPROCESSING
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  Tokenization    Normalization    Linguistic
                                   information
       │
       ▼
   TOKENS / FEATURES
       │
       ▼
   NUMERICAL DATA
       │
       ▼
  MACHINE LEARNING
       │
       ▼
     PREDICTION
```

------------------------------------------------------------------------

# 57. The Three Big Ideas of Today's Practical

## Idea 1 -- Computers don't naturally understand raw text

We need to represent language in a form algorithms can process.

------------------------------------------------------------------------

## Idea 2 -- Preprocessing is not simply cleaning

Every deletion is a decision.

For example:

``` text
Remove "not"?
```

Could completely reverse sentiment.

The practical lecture's central principle is that preprocessing is a
task-dependent information-selection problem.
fileciteturn3file0L228-L248

------------------------------------------------------------------------

## Idea 3 -- Rules don't scale

Rule-based NLP asks:

> "What rule should I write?"

Statistical NLP asks:

> "What patterns can I learn from examples?"

That is one of the most important historical transitions in NLP.

------------------------------------------------------------------------

# 58. Rule-Based vs Statistical NLP -- Final Comparison

  Feature                    Rule-Based NLP           Statistical NLP
  -------------------------- ------------------------ ------------------------------------
  Main idea                  Humans write rules       Model learns patterns
  Data requirement           Low/optional             Training examples/data
  Flexibility                Low                      Higher
  Handling new expressions   Poor                     Can improve with data
  Maintenance                Rules must be updated    Model can be retrained
  Context                    Usually limited          Can learn statistical associations
  Scalability                Difficult                Better
  Example                    `IF "good" → positive`   Learn from labeled reviews

------------------------------------------------------------------------

# 59. But Statistical NLP is not the end

The important story is:

``` text
RULES
  ↓
STATISTICS
  ↓
MACHINE LEARNING
  ↓
DEEP LEARNING
  ↓
TRANSFORMERS
  ↓
LLMs
```

Each generation tried to solve limitations of the previous generation.

So when students later learn:

-   Word embeddings
-   RNNs
-   LSTMs
-   Attention
-   Transformers
-   BERT
-   GPT-style models
-   LLMs

they should remember:

> **We are continuing the same journey: finding better ways to represent
> and learn human language.**

------------------------------------------------------------------------

# 60. Debugging Checklist for Beginners

If code does not run, do not panic.

### Error: `ModuleNotFoundError`

Run:

``` bash
pip install nltk scikit-learn pandas
```

------------------------------------------------------------------------

### Error: NLTK resource missing

Run the relevant download:

``` python
nltk.download("punkt")
```

or:

``` python
nltk.download("stopwords")
```

or:

``` python
nltk.download("wordnet")
```

------------------------------------------------------------------------

### Nothing happens

Check that you actually ran the file:

``` bash
python nlp_practical.py
```

------------------------------------------------------------------------

### Output looks different from the teacher's

That's okay.

Tokenizers, stop-word lists, library versions and preprocessing choices
can produce slightly different outputs.

Focus first on:

> **What transformation happened?**

rather than:

> **Does my output look character-for-character identical?**

------------------------------------------------------------------------

# 61. Viva Questions

### Basic

1.  What is NLP?
2.  What is text preprocessing?
3.  What is a token?
4.  Why do we tokenize text?
5.  What are stop words?
6.  What is normalization?
7.  What is POS tagging?
8.  What is stemming?
9.  What is lemmatization?

### Understanding

10. Why can removing `not` be dangerous?
11. Why can punctuation contain useful information?
12. Why can lowercasing sometimes destroy information?
13. Why can stemming create non-words?
14. Why is lemmatization more linguistically informed?
15. Why does preprocessing depend on the task?

### Conceptual

16. What is rule-based NLP?
17. Why did rule-based NLP struggle?
18. What is statistical NLP?
19. What is the difference between a rule and a learned pattern?
20. Why do we convert text into numbers?
21. What is Bag-of-Words?
22. What does a Naive Bayes classifier do?

### Thinking questions

23. Why might `"Great!"` and `"Great... another broken update"` have
    different meanings?
24. Why might an English NLP pipeline struggle with Hinglish?
25. If a model performs badly, is preprocessing always the problem?
26. Why can't we simply remove everything that appears frequently?

------------------------------------------------------------------------

# 62. Final Classroom Challenge 🏆

Give students this sentence:

> **"OMG!!! This phone is sick 🔥🔥 but the battery isn't great 😭
> #Review"**

Ask each group to produce:

### Version A -- Search-friendly

Remove what they think is unnecessary.

### Version B -- Sentiment-friendly

Preserve information that carries sentiment.

### Version C -- LLM-friendly

Discuss why an old-fashioned stop-word/stemming pipeline may not be
appropriate.

Then ask:

> **"Which version is the correct one?"**

The answer:

# There isn't one universally correct version.

It depends on the task.

------------------------------------------------------------------------

# 63. Exit Question

Before students leave, ask:

> **If human language is messy, why don't we just write enough rules to
> handle everything?**

A good student answer should be:

> Because human language has too many exceptions, contexts, meanings,
> domains, slang expressions and constantly changing patterns.
> Hand-written rules become difficult to maintain and do not scale.

Then ask:

> **So what did NLP researchers try next?**

Students should answer:

> **Statistical NLP --- learn patterns from data instead of manually
> writing every rule.**

And that gives you the perfect bridge to the next practical/lecture.

------------------------------------------------------------------------

# 64. One-Sentence Takeaway

> **Text preprocessing turns messy language into useful computational
> representations, while statistical NLP shifts us from manually writing
> every language rule toward learning patterns from data.**

------------------------------------------------------------------------

## Instructor Note

Do not rush through the code.

For beginners, use this rhythm:

``` text
EXPLAIN
   ↓
SHOW ONE LINE OF CODE
   ↓
RUN IT
   ↓
SHOW THE OUTPUT
   ↓
ASK "WHAT CHANGED?"
   ↓
EXPLAIN WHY
   ↓
LET STUDENTS MODIFY THE INPUT
```

For example, when teaching tokenization, do not immediately show the
entire pipeline.

Start with:

``` python
text = "I love NLP!"
```

Then:

``` python
tokens = word_tokenize(text)
```

Then:

``` python
print(tokens)
```

Then ask:

> **"What did the computer do to our sentence?"**

Only after students understand that transformation should you move to
the next step.

The objective of this first practical is **not to make students memorize
NLTK functions**.

The objective is to make them understand the journey:

``` text
Human language
      ↓
Messy text
      ↓
Processing decisions
      ↓
Tokens
      ↓
Features / numbers
      ↓
Learning
      ↓
Prediction
```

That mental model will make the later NLP topics---embeddings, RNNs,
attention, Transformers and LLMs---much easier to understand.

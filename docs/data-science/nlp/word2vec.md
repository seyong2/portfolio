---
title: Word Embedding and Word2vec
parent: Natural Language Processing
nav_order: 1
layout: default
---

Converting words to numbers is a crucial step in many natural language processing (NLP) tasks and machine learning models that involve text data. Most machine learning algorithms, especially those involving deep learning, are designed to work with numerical data. They perform mathematical operations that require inputs to be in the form of numbers. By converting words to numbers, we create a numerical representation of text, often in the form of vectors. These vectors capture the semantic meaning of words in a form that machines can process. Techniques like Word2Vec or GloVe represent words as dense vectors in a continuous vector space, where words with similar meanings are located close to each other.

In this post, we'll walk through implementing the Word2Vec algorithm in Python, a widely used technique for generating word embeddings. Word2Vec is a shallow neural network with two layers, designed to reconstruct the linguistic contexts of words. It primarily comes in two flavors: **Continuous Bag of Words (CBOW)** and **Skip-gram**. Our focus will be on the Skip-gram model.

The Skip-gram model operates by taking a target word as input and attempting to predict its surrounding context words. For instance, given the sentence "The cat sat on the mat" and the target word "cat," the model aims to predict the context words "The," "sat," "on," and "mat." We'll explore how the Skip-gram model functions with a straightforward example, demonstrating its practical implementation and effectiveness.

## Load Necessary Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_eager_execution()
```

## Collect Data

We'll use 15 short phrases about cats and dogs.

```python
corpus = [
    "Cats purring by the fireplace",
    "Dogs wagging their happy tails",
    "Playful kittens chasing yarn balls",
    "Loyal dogs guarding the house",
    "Cats curled up on cushions",
    "Dogs running through the park",
    "Cats watching birds intently",
    "Dogs fetching their favorite toys",
    "Mischievous kittens climbing curtains",
    "Faithful dogs following their owners",
    "Cats lounging in sunbeams",
    "Dogs barking at strangers",
    "Cats grooming their silky fur",
    "Dogs splashing in puddles",
    "Cats leaping onto high shelves"
]

# convert phrases into lower case
corpus = [c.lower() for c in corpus]
```

When using Word2Vec, whether or not to remove stop words depends on the specific application and the nature of the data. In this example, we will remove stop words due to the reasons below:

- Reducing noise: Stop words like "the", "is", "and", etc. are very common and often don't carry much semantic meaning. Including them might add noise to the model, as they could dominate the word vectors and lead to less informative embeddings.

- Focusing on meaningful words: By removing stop words, we focus the model on the words that contribute more to the actual meaning of the text. This can lead to better quality word embeddings, especially for downstream tasks like text classification, sentiment analysis, or topic modeling.

- Improving efficiency: Removing stop words reduces the vocabulary size and the number of word pairs the model needs to process, which can lead to faster training and less computational cost.

In summary, while there are compelling reasons to remove stop words before training Word2Vec models, it ultimately depends on the specific use case and the importance of contextual information in your data. Thus, it's often best to experiment with both approches and see which approach works better for your data and objectives.

```python
# define the stop words that we want to remove
stop_words = ['by', 'the', 'in', 'onto', 'at', 'their', 'through', 'up', 'on']

# define a function that removes the stop words
def remove_stop_words(sentences, stop_words):
    stop_words_set = set(stop_words) # Convert list to set for faster lookups
    for i, sentence in enumerate(sentences):
        words = sentence.split() # Split sentence into words
        filtered_words = [word for word in words if word not in stop_words_set] # Filter out stop words
        sentences[i] = ' '.join(filtered_words) # Join the remaining words back into a sentence
    return sentences

corpus = remove_stop_words(corpus, stop_words)
print(corpus)
```

```
    ['cats purring fireplace', 'dogs wagging happy tails', 'playful kittens chasing yarn balls', 'loyal dogs guarding house', 'cats curled cushions', 'dogs running park', 'cats watching birds intently', 'dogs fetching favorite toys', 'mischievous kittens climbing curtains', 'faithful dogs following owners', 'cats lounging sunbeams', 'dogs barking strangers', 'cats grooming silky fur', 'dogs splashing puddles', 'cats leaping high shelves']  
    
```

```python
# extract the unique words that appear on the data
unique_words = []

for c in corpus:
  words = c.split()
  for w in words:
    if w not in unique_words:
      unique_words.append(w)

unique_words, len(unique_words)
```
```
(['cats',
  'purring',
  'fireplace',
  'dogs',
  'wagging',
  'happy',
  'tails',
  'playful',
  'kittens',
  'chasing',
  'yarn',
  'balls',
  'loyal',
  'guarding',
  'house',
  'curled',
  'cushions',
  'running',
  'park',
  'watching',
  'birds',
  'intently',
  'fetching',
  'favorite',
  'toys',
  'mischievous',
  'climbing',
  'curtains',
  'faithful',
  'following',
  'owners',
  'lounging',
  'sunbeams',
  'barking',
  'strangers',
  'grooming',
  'silky',
  'fur',
  'splashing',
  'puddles',
  'leaping',
  'high',
  'shelves'],
 43)

 ```

After removing the stop words, we are left with 43 unique words.

Now we convert the data into numbers before feeding them to the model. The input to the model is a target word, which is typically represented as a one-hot encoded vector. In a one-hot encoded vector, only one element is set to 1 (corresponding to the word's index in the vocabulary), and all other elements are set to 0.

In addition, to construct the input, we also need to determine the context window size, the number of words before and after the target word that are considered context words. If the window size is 2, this means that the model considers the two words immediately before and the two words immediately after the target word as context words. From the example before, "The cat sat on the mat", the context words for "cat" would be "The", "sat", and "on". Thus, the valid training pairs for "cat" with a window size of 2 would be:

(cat, The)
(cat, sat)
(cat, on)
After one-hot encoding, the training pairs would look like:

Input:
[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]

target
[[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]

```python

# define a function that finds the context words for each target word
def data_generation(data, window_size):
    rows = []

    for d in data:
        words = d.split()  # Split sentence into words
        num_words = len(words)

        for i, word in enumerate(words):
            lower = max(i - window_size, 0)
            upper = min(i + window_size + 1, num_words)

            neighbors = words[lower:i] + words[i+1:upper]
            for neighbor in neighbors:
                rows.append((word, neighbor))

    # Convert the rows to a DataFrame
    result_df = pd.DataFrame(rows, columns=['word', 'neighbor'])

    # Drop duplicate pairs if any
    result_df = result_df.drop_duplicates()

    return result_df
  
corpus_df = data_generation(corpus, window_size=2)
corpus_df
```

| word      | neighbor   |
|-----------|------------|
| cats      | purring    |
| cats      | fireplace  |
| purring   | cats       |
| purring   | fireplace  |
| fireplace | cats       |
| ...       | ...        |
| high      | cats       |
| high      | leaping    |
| high      | shelves    |
| shelves   | leaping    |
| shelves   | high       |


```python
# one-hot encode the 'word' and 'neighbor' columns
corpus_encode_df = pd.get_dummies(corpus_df, columns=['word', 'neighbor'], dtype='int')

# split the encoded DataFrame into X and y
word_columns = corpus_encode_df.filter(like='word_').columns
neighbor_columns = corpus_encode_df.filter(like='neighbor_').columns

X = corpus_encode_df[word_columns]
y = corpus_encode_df[neighbor_columns]

# convert X and y into numpy array
X = np.asarray(X)
y = np.asarray(y)

# making placeholders for X_train and y_train
x = tf.compat.v1.placeholder(tf.float32, shape=(None, len(unique_words)))
y_label = tf.compat.v1.placeholder(tf.float32, shape=(None, len(unique_words)))
```

With our input data prepared, the next step is to define the projection layer (recall that skip-gram is a two-layer model). In this layer, the one-hot vector is multiplied by a weight matrix, converting it into a dense vector. This weight matrix is commonly known as the input embedding matrix. The dense vector produced has a lower dimensionality than the original one-hot vector, enabling more efficient encoding of semantic information.

For this example, we want the embedding matrix to be two-dimensional, making it easier to visualize the matrix later.

```python
# define dimension for word embedding
dim_embedding = 2

# hidden layer: which represents word vector eventually
w1 = tf.Variable(tf.random.normal([len(unique_words), dim_embedding]))
b1 = tf.Variable(tf.random.normal([1]))
hidden_layer = tf.add(tf.matmul(x, w1), b1)
```

The dense vector is then multiplied by another weight matrix to produce a score for each word in the vocabulary. These scores are passed through a softmax function, which converts them into probabilities. Each probabiliy represents the likelihood of a word in the vocabulary being a context word for the target word.

```python
# output layer
w2 = tf.Variable(tf.random.normal([dim_embedding, len(unique_words)]))
b2 = tf.Variable(tf.random.normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, w2), b2))
```

We are nearly ready to begin training the model. However, before proceeding, we need to complete the final step of defining the loss function and optimizer. The model adjusts its weights to maximize the probability of correctly predicting context words given a target word, which involves minimizing a loss function. For this task, we will use cross-entropy loss as our loss function and apply stochastic gradient descent (SGD) as the optimizer with a learning rate of 0.05.

```python
# loss function: cross_entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.math.log(prediction), axis=[1]))

# training operation
train_op = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(loss)
```

We now train the model, iterating through the input data 20,000 times to ensure the weight matrix is effectively adjusted.

```python
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

epochs = 20000
for i in range(epochs):
  sess.run(train_op, feed_dict={x: X, y_label: y})
  if i%1000 == 0:    print(f'epoch {str(i)} loss is {sess.run(loss, feed_dict={x: X, y_label: y})}')
```
```
epoch 0 loss is 4.909794330596924
epoch 1000 loss is 3.3665082454681396
epoch 2000 loss is 3.162121534347534
epoch 3000 loss is 2.9629886150360107
epoch 4000 loss is 2.7849252223968506
epoch 5000 loss is 2.6631717681884766
epoch 6000 loss is 2.5729081630706787
epoch 7000 loss is 2.5068137645721436
epoch 8000 loss is 2.455470085144043
epoch 9000 loss is 2.414684295654297
epoch 10000 loss is 2.382246971130371
epoch 11000 loss is 2.3564255237579346
epoch 12000 loss is 2.3353805541992188
epoch 13000 loss is 2.3175764083862305
epoch 14000 loss is 2.301966905593872
epoch 15000 loss is 2.287816047668457
epoch 16000 loss is 2.274552583694458
epoch 17000 loss is 2.2617273330688477
epoch 18000 loss is 2.249037981033325
epoch 19000 loss is 2.2363951206207275
```

We can observe that the loss consistently decreases with each iteration, which aligns with our expectations.

Next, let's examine the embedding matrix generated by the trained model.

```python
# word embedding matrix
vectors = sess.run(w1 + b1)
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = unique_words
w2v_df = w2v_df[['word', 'x1', 'x2']]
w2v_df
```

| word        | x1         | x2         |
|-------------|------------|------------|
| cats        | -3.844231  | -1.199567  |
| purring     | 1.942404   | 1.033176   |
| fireplace   | -2.108984  | 2.973610   |
| dogs        | -0.377736  | -0.048961  |
| wagging     | -3.618866  | -1.449615  |
| happy       | -2.400694  | -2.663845  |
| tails       | -2.094384  | 1.848234   |
| playful     | -2.883839  | -2.953384  |
| kittens     | -2.274646  | 1.039603   |
| chasing     | 0.600096   | 0.297083   |
| yarn        | 0.551523   | 4.058111   |
| balls       | 3.748494   | 0.188655   |
| loyal       | 3.725614   | 0.187645   |
| guarding    | -2.215774  | 1.766397   |
| house       | 0.170705   | 3.163560   |
| curled      | 1.369942   | -3.943550  |
| cushions    | -0.293968  | -0.434090  |
| running     | 1.187384   | 0.929648   |
| park        | 3.477580   | 0.344592   |
| watching    | -0.505951  | -0.353842  |
| birds       | 1.104028   | 2.464244   |
| intently    | -1.365888  | 1.652169   |
| fetching    | -2.129038  | -1.574574  |
| favorite    | -0.506404  | -0.353983  |
| toys        | -2.262563  | 2.267512   |
| mischievous | 1.110344   | 2.417118   |
| climbing    | -2.858588  | -2.927774  |
| curtains    | 0.553090   | 3.827151   |
| faithful    | 1.210875   | 1.583030   |
| following   | -3.964548  | -1.953791  |
| owners      | 2.756517   | 0.832778   |
| lounging    | -2.343008  | 0.866423   |
| sunbeams    | 1.430465   | 1.337379   |
| barking     | 0.229596   | -2.222786  |
| strangers   | -0.294027  | -0.434112  |
| grooming    | 2.053764   | 0.977429   |
| silky       | 1.120451   | 1.715440   |
| fur         | -2.673492  | 0.680454   |
| splashing   | 3.468613   | -0.462991  |
| puddles     | 2.910977   | -2.277403  |
| leaping     | 3.533613   | 0.346079   |
| high        | -2.060866  | 2.757971   |
| shelves     | -4.135650  | -1.584913  |

Finally, we visualize the embedding matrix to examine the spatial arrangement of the words.

For instance, we observe that the words "kittens," "lounging," and "fur" are situated close to each other. This clustering aligns well with our understanding, as kittens are known to lounge and have fur.

```python
fig, ax = plt.subplots(figsize=(10, 10))

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))
    
padding = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - padding
y_axis_min = np.amin(vectors, axis=0)[1] - padding
x_axis_max = np.amax(vectors, axis=0)[0] + padding
y_axis_max = np.amax(vectors, axis=0)[1] + padding
 
plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)

plt.show()
```

![image](https://github.com/user-attachments/assets/6495bd72-56da-4554-a4f8-2bbca0af43b7)

The traditional Skip-gram model faces significant computational challenges due to the need to evaluate probabilities across an entire vocabulary, especially when dealing with very large vocabularies. This process can become prohibitively slow and inefficient, as it involves calculating scores for every word and performing extensive backpropagation.

Negative sampling offers a practical solution by altering the training objective. Instead of predicting probabilities for the entire vocabulary, it narrows the focus to distinguishing between a few positive context words and a limited number of negative samples. This approach dramatically reduces the computational requirements, enabling the model to train effectively even with vast datasets and extensive vocabularies. By streamlining the training process, negative sampling makes it feasible to leverage large-scale data, enhancing both efficiency and scalability in word embedding models.

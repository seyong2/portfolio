---
title: Decoder-Only Transformers
parent: Natural Language Processing
nav_order: 5
layout: default
---

In this post, we will explore the **Decoder-Only Transformer**, the foundation of ChatGPT, through a simple code example. For the code, I referred to Josh Starmer's video, **[Coding a ChatGPT Like Transformer From Scratch in PyTorch](https://www.youtube.com/watch?v=C9QSpl5nmrY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=29)**. I highly recommend watching the video if you're unfamiliar with the concept of Decoder-Only Transformer. At the end of the video, Josh outlines the key differences between a standard Transformer and a Decoder-Only Transformer.

1. A Decoder-Only Transformer has a single unit responsible for both encoding the input and generating the output. In contrast, a standard Transformer uses two units: an Encoder to process the input and a Decoder to generate the output.
2. A standard Transformer uses two types of attention during inference: Self-Attention and Encoder-Decoder Attention. During training, it uses Masked Self-Attention but only on the output. On the other hand, a Decoder-Only Transformer utilizes only one type of attention, Masked Self-Attention.

Now that we understand the differences between the Decoder-Only Transformer and a standard Transformer, let's get started by importing the Python modules that we will use in this post.

## Import Python Modules

```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader
```

## Create Training Data

We want to build a model that can respond correctly to two different prompts.

1. "How is living in Amsterdam?"
2. "Living in Amsterdam is how?"

The answer to both prompts will be "Exciting".

Our vocabulary for this model will consist of 7 words (or tokens): "**how**", "**is**", "**living**", "**in**", "**amsterdam**", "**exciting**", and "**\<EOS>**" (End of Sentence). 

```python
# map the tokens to numbers for word embedding
# nn.Embedding only accepts numbers as input
token_to_id = {'how': 0,
               'is': 1,
               'living': 2,
               'in': 3,
               'amsterdam': 4,
               'exciting': 5,
               '<EOS>': 6}

# from numbers back to the original tokens
id_to_token = dict(map(reversed, token_to_id.items()))

# the tokens for input during training come from the promopts as well as from generating the output
inputs = torch.tensor([[token_to_id['how'],
                        token_to_id['is'],
                        token_to_id['living'],
                        token_to_id['in'],
                        token_to_id['amsterdam'],
                        token_to_id['<EOS>'],
                        token_to_id['exciting']],

                      [token_to_id['living'],
                       token_to_id['in'],
                       token_to_id['amsterdam'],
                       token_to_id['is'],
                       token_to_id['how'],
                       token_to_id['<EOS>'],
                       token_to_id['exciting']]])

labels = torch.tensor([[token_to_id['is'],
                        token_to_id['living'],
                        token_to_id['in'],
                        token_to_id['amsterdam'],
                        token_to_id['<EOS>'],
                        token_to_id['exciting'],
                        token_to_id['<EOS>']],

                       [token_to_id['in'],
                        token_to_id['amsterdam'],
                        token_to_id['is'],
                        token_to_id['how'],
                        token_to_id['<EOS>'],
                        token_to_id['exciting'],
                        token_to_id['<EOS>']]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

# let's look at the first input and label data
next(iter(dataloader))
```

```
[tensor([[0, 1, 2, 3, 4, 6, 5]]), tensor([[1, 2, 3, 4, 6, 5, 6]])]
```

## Positional Encoding

Since `nn.Embedding` will handle creating word embeddings for us, the next step is positional encoding.

Unlike RNNs, where tokens are processed sequentially, in Transformers, we process all tokens simultaneously. Therefore, we need to add a value that provides information about the order of the tokens in the sequence. For this purpose, we use alternating sine and cosine functions.

The formulas for positional encoding are:

$$PE_{pos, 2i}=sin(pos/10000^{2i/d\_model})$$

$$PE_{pos, 2i+1}=cos(pos/10000^{2i/d\_model})$$

Where:
- $$pos$$ is the position of the token in the input.
- $$d\_model$$ is the dimensionality of the word embedding
- $$i$$ indicates the position within the embedding dimension. Both $pos$ and $$i$$ start at 0, and $i$ increments by 1 for each successive sine and cosine pair.

If we assume each token has 4-dimensional word embedding ($$d\_model=4$$), each input token will also have 4 corresponding position encoding values. For example, the positional encoding values for the first token, "how", are

$$PE_{0, 2\times0}=sin(0/10000^{2\times0/4})=sin(0)=0$$

$$PE_{0, 2\times0+1}=cos(0/10000^{2\times0/4})=cos(0)=1$$

$$PE_{0, 2\times1}=sin(0/10000^{2\times1/4})=sin(0)=0$$

$$PE_{0, 2\times1+1}=cos(0/10000^{2\times0/4})=cos(0)=1$$

```python
class PositionEncoding(nn.Module):

  def __init__(self, d_model=4, max_len=20):
    # max_len is the maximum number of tokens our Transformer can process (input + output)
    super().__init__()
    
    # create a zero-filled matrix of position encoding values
    pe = torch.zeros(max_len, d_model)
    
    # column vector that represents the positions
    position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
    
    # row vector that represents 2*i for each word embedding
    embedding_index = torch.arange(start=0, end=d_model, step=2).float()
    
    # pos/10000**(2*i/d_model)
    div_term = 1/torch.tensor(10000)**(embedding_index/d_model)
    
    # we replace the even columns with values from the sine function
    pe[:, 0::2] = torch.sin(position * div_term)

    # we replace the odd columns with values from the cosine function
    pe[:, 1::2] = torch.cos(position * div_term)

    # to ensure the position encoding values get moved to a GPU if we use one
    self.register_buffer('pe', pe)

  def forward(self, word_embeddings):
    # take word embedding values and add the position encoding values element-wisely
    return word_embeddings + self.pe[:word_embeddings.size(0), :]
```

## Masked Self-Attention

As discussed in the previous post, Masked Self-Attention works by comparing each word to itself and all preceding words in the sentence.

For example, in our first prompt, "how is living in Amsterdam," the Masked Self-Attention values for the first token, "how," will only reflect its similarity to itself. In contrast, the Masked Self-Attention values for the third token, "living," will capture its similarity to itself and the preceding tokens, "how" and "is."

The Masked Self-Attention mechanism is mathematically defined as:

$$Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d_k}}+M)V$$

In a standard Transformer, this mechanism is used only in the decoder during training. However, in a Decoder-Only Transformer, masked self-attention is applied continuously, both during training and inference, incorporating both input and output.

```python
class Attention(nn.Module):

  def __init__(self, d_model=4):
    super().__init__()

    self.d_model = d_model

    # the weight matrices to compute Query, Key, and Value have d_model rows and d_model columns
    # we use nn.Linear as we will dot product -> encoded tokens * weights
    self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    # to keep track of which indices are for row and columns
    self.row_dim = 0
    self.col_dim = 1

  def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
    
    # compute Query, Key, and Values
    q = self.W_q(encodings_for_q)
    k = self.W_k(encodings_for_k)
    v = self.W_v(encodings_for_v)

    # matrix multiplication between q and the transpose of k -> similarities
    sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

    # scale the similarities by the square root of the dimension of k
    scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

    # incase we want to add the mask to prevent early tokens from looking at later tokens
    if mask is not None:
      # add the mask to the scaled similarities
      # mask has negative infinity values for the token to be ignored and 0 for the others
      scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

    # softmax function to determine the percentage of influence that each token should have on the others
    attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

    # multiply the attention weights by V
    attention_scores = torch.matmul(attention_percents, v)

    return attention_scores
```

## The Decoder-Only Transformer

We can now create a class for the Decoder-Only Transformer, building on the classes we've already implemented and adding the necessary steps for completing the model.

```python
class DecoderOnlyTransformer(nn.Module):

  def __init__(self, num_tokens=7, d_model=4, max_len=20):
    # num_tokens is the number of tokens inthe vocabulary
    super().__init__()

    # create an embedding object
    self.we = nn.Embedding(num_embeddings=num_tokens,
                           embedding_dim=d_model)

    # position encoding object
    self.pe = PositionEncoding(d_model=d_model,
                               max_len=max_len)

    # attention object
    self.self_attention = Attention(d_model=d_model)

    # create a fully connected layer
    self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

    # self.loss = nn.CrossEntropyLoss()

  def forward(self, token_ids):
    # token_ids: an array of token id numbers for inputs

    # 1. Word embedding
    word_embeddings = self.we(token_ids)

    # 2. Position encoding
    position_encoded = self.pe(word_embeddings)

    # 3. Create the mask
    # torch.tril() leaves the values in the lower triangle as they are (1) and turns everything else into 0s.
    mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
    # Trues for 0s
    mask = mask == 0

    # # 4. Masked self attention
    self_attention_values = self.self_attention(position_encoded,
                                                position_encoded,
                                                position_encoded,
                                                mask=mask)

    # 5. Residual connection
    residual_connection_values = position_encoded + self_attention_values

    # 6. Fully connected layer
    fc_layer_output = self.fc_layer(residual_connection_values)

    return fc_layer_output
```

## Training

Now that we have defined our Transformer model, the next step is to train it by optimizing the model parameters.

```python
# create a model from DecoderOnlyTransformer()
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=4, max_len=20)

# define loss and optimizer
# nn.CrossEntropyLoss() applies softmax function for us
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.05)

# number of epochs
epochs = 100

for epoch in range(epochs):
  # put the model in training mode
  model.train()

  # we have two prompts
  for i, data in enumerate(dataloader):
    inputs, labels = data

    # forward pass
    outputs = model(inputs[0])

    # calculate the loss
    loss = loss_fn(outputs, labels[0])

    # make gradients zero
    optimizer.zero_grad()

    # backpropagation
    loss.backward()
    
    # update the parameters
    optimizer.step()

  if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}")
```

```
Epoch: 0 | Loss: 2.28202
Epoch: 10 | Loss: 0.66447
Epoch: 20 | Loss: 0.06849
Epoch: 30 | Loss: 0.00769
Epoch: 40 | Loss: 0.00344
Epoch: 50 | Loss: 0.00221
Epoch: 60 | Loss: 0.00162
Epoch: 70 | Loss: 0.00126
Epoch: 80 | Loss: 0.00101
Epoch: 90 | Loss: 0.00083
```

## Make Predictions

Finally, we will use the model to predict the response to the prompts we created.

```python
model_input = torch.tensor([token_to_id["how"],
                            token_to_id["is"],
                            token_to_id["living"],
                            token_to_id["in"],
                            token_to_id["amsterdam"],
                            token_to_id["<EOS>"]])

input_length = model_input.size(dim=0)

# the model generates a prediction for each input token
predictions = model(model_input)

# we're only interested in the prediction of <EOS> token (last prediction)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
predicted_ids = predicted_id

# we create a loop to generate output tokens
# until we reach the maximum number of tokens that our model can generate
max_length = 20
for i in range(input_length, max_length):
  # or the model generates the <EOS> token
  if (predicted_id == token_to_id["<EOS>"]): # if the prediction is <EOS>, then we are done
    break

  # each time we generate a new output token, we add it to the input
  model_input = torch.cat((model_input, predicted_id))

  predictions = model(model_input)
  predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
  predicted_ids = torch.cat((predicted_ids, predicted_id))

print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])
```

```
Predicted Tokens:

	 exciting
	 <EOS>
```

```python
model_input = torch.tensor([token_to_id["living"],
                            token_to_id["in"],
                            token_to_id["amsterdam"],
                            token_to_id["is"],
                            token_to_id["how"],
                            token_to_id["<EOS>"]])

input_length = model_input.size(dim=0)

# the model generates a prediction for each input token
predictions = model(model_input)

# we're only interested in the prediction of <EOS> token (last prediction)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
predicted_ids = predicted_id

# we create a loop to generate output tokens
# until we reach the maximum number of tokens that our model can generate
max_length = 20
for i in range(input_length, max_length):
  # or the model generates the <EOS> token
  if (predicted_id == token_to_id["<EOS>"]): # if the prediction is <EOS>, then we are done
    break

  # each time we generate a new output token, we add it to the input
  model_input = torch.cat((model_input, predicted_id))

  predictions = model(model_input)
  predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
  predicted_ids = torch.cat((predicted_ids, predicted_id))

print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])
```

```
Predicted Tokens:

	 exciting
	 <EOS>
```

We observe that the model functions precisely as intended.

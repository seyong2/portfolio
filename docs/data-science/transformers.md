---
title: Understanding Transformers
parent: Natural Language Processing
nav_order: 4
layout: default
---

In this post, we will explore how **Transformer**, the foundation of models like ChatGPT, operates step by step using a simple example. Specifically, we'll focus on how a transformer neural network translates the simple Spanish sentence "*Te quiero*" into English, "*I love you*".

## Encoder

### 1. Word Embedding

To begin, we need to convert words into numbers, as neural networks, including transformers, require numerical input. For this, we'll use word embeddings. The core idea behind word embeddings is to provide a numerical representation of text that captures the semantic meaning of words.

magine we have a vocabulary consisting of five tokens: *<SOS>* (Start of Sentence), *te*, *y*, *quiero*, and *vas*. Suppose we want the network to produce two values for each token, meaning the word embeddings will be in a 2-dimensional space. The word embedding model would then look like this: starting with a one-hot matrix where each row has a value of 1 for its corresponding token and 0 for others. Given that our input tokens are *<SOS>* (Start of Sentence), *te*, and *quiero*, the resulting word embedding vector will be a matrix with a shape of (3 $$\times$$ 2).

<p align="center">
  <img src="https://github.com/user-attachments/assets/76a93ed3-a74b-4770-8d18-dcefd2df45e8" title="word_emb">
</p>

The embedding operation can be represented mathematically as:

$$ IP * W_{emb} = {\left\lbrack \matrix{1 & 0 & 0 & 0 & 0 \cr 0 & 1 & 0 & 0 & 0 \cr 0 & 0 & 0 & 1 & 0} \right\rbrack} * \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{20} & w_{21} \cr w_{30} & w_{31} \cr w_{40} & w_{41} \cr w_{50} & w_{51}} \right\rbrack = \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{40} & w_{41}} \right\rbrack $$

### 2. Position Encoding

Next, we need to encode the order of the words, as transformers don't inherently understand word order (unlike sequential models like RNNs or LSTMs). Positional encoding is added to the embedding vectors to incorporate word order information. These encodings are generated using a sequence of alternating sine and cosine functions.

For each position $$p$$ of a token in the input and each embedding dimension $$2i$$ (even-indexed) and $$2i+1$$ (odd-indexed), the positional encoding is calculated as:

$$PE(p, 2i)=sin(\frac{p}{10000^{\frac{2i}{d}}})$$

$$PE(p, 2i+1)=cos(\frac{p}{10000^{\frac{2i}{d}}})$$

where $$d$$ is the dimension of the word embedding (in this case, 2).

Thus, the positional encoding matrix will have a shape of (3 $$\times$$ 2) and might look like:

$$ PE_{enc} = \left\lbrack \matrix{sin(\frac{0}{1000^{2\times0/2}}) & cos(\frac{0}{1000^{2\times0/2}}) \cr sin(\frac{21}{1000^{2\times0/2}}) & cos(\frac{1}{1000^{2\times0/2}}) \cr sin(\frac{2}{1000^{2\times0/2}}) & cos(\frac{2}{1000^{2\times0/2}})} \right\rbrack = \left\lbrack \matrix{0 & 1 \cr 0.84 & 0.54 \cr 0.91 & -0.42} \right\rbrack $$

We then perform element-wise addition of the word embeddings and the positional encodings to form a new representation.

### 3. Self-Attention

Self-attention is a mechanism that helps the model understand how different tokens in a sentence relate to each other. Each token "attends" to all other tokens in the sentence, allowing the model to gather context and update its understanding of each word.

This is implemented using a query-key-value mechanism:

- **Query**: Represents the token seeking information. We obtain the query by multiplying the encoded values by the transposed query weights matrix ($$W_Q^T$$), producing a matrix $$Q$$ of shape (3 $$\times$$ 2).

- **Key**: Represents the token being "queried." The key is calculated by multiplying the encoded values by the transposed key weights matrix ($$W_K^T$$), producing matrix $$K$$.

- **Value**: Represents the information provided by each token. We calculate the value by multiplying the encoded values by the transposed value weights matrix ($$W_V^T$$), producing matrix $$V$$.

$$ Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d_k}})V $$

The similarity between words is computed using the dot product of $$Q$$ and $$K^T$$. The resulting similarity scores are scaled by dividing by $$\sqrt{d_k}$$ (where $$d_k$$ is the dimension of $$K$$). We then apply a softmax function to these scaled scores to produce attention weights. Finally, the attention output is calculated by multiplying these weights by $$V$$, resulting in a self-attention matrix with a shape of (3 $$\times$$ 2).

In practice, multiple self-attention heads are used (Multi-Head Attention) to capture different types of relationships within the sentence.

### 4. Residual Connection

To ensure that the model doesn't lose information from the word embeddings and positional encodings, we use residual connections. These connections add the original encoded values to the self-attention outputs. Since both matrices have the same shape, the resulting encoded values will also have a shape of (3 $$\times$$ 2).

Now that we've encoded the Spanish input phrase, it's time to decode it into the English phrase, "*I love you*".

## Decoder

### 1. Word Embedding

Similar to the encoder, the decoder starts by embedding the English tokens. Let's assume the vocabulary consists of six tokens: *<SOS>* (to initialize the decoder), *I*, *love*, *you*, *and*, and *<EOS>* (to stop generating output). During training, we use teacher forcing, initializing the decoder with known output values. For instance, we might initialize the first decoder input with *<SOS>*, the second with *i*, the third with *love*, and the final one with *you*. The word embedding vector for the decoder would then be a matrix of shape (4 $$\times$$ 2).

<p align="center">
  <img src="https://github.com/user-attachments/assets/06b565fd-7ecc-4f0b-8243-5711f3b3ee29" title="word_emb_dec">
</p>

$$ OP * W_{emb} = {\left\lbrack \matrix{1 & 0 & 0 & 0 & 0 & 0 \cr 0 & 1 & 0 & 0 & 0 & 0 \cr 0 & 0 & 1 & 0 & 0 & 0 \cr 0 & 0 & 0 & 1 & 0 & 0} \right\rbrack} * \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{20} & w_{21} \cr w_{30} & w_{31} \cr w_{40} & w_{41} \cr w_{50} & w_{51} \cr w_{60} & w_{61}} \right\rbrack = \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{20} & w_{21} \cr w_{30} & w_{31}} \right\rbrack $$

### 2. Position Encoding

As in the encoder, we add positional encodings to the decoder's word embeddings.

$$ PE_{dec} = \left\lbrack \matrix{sin(\frac{0}{1000^{2\times0/2}}) & cos(\frac{0}{1000^{2\times0/2}}) \cr sin(\frac{21}{1000^{2\times0/2}}) & cos(\frac{1}{1000^{2\times0/2}}) \cr sin(\frac{2}{1000^{2\times0/2}}) & cos(\frac{2}{1000^{2\times0/2}}) \cr sin(\frac{3}{1000^{2\times0/2}}) & cos(\frac{3}{1000^{2\times0/2}}} \right\rbrack = \left\lbrack \matrix{0 & 1 \cr 0.84 & 0.54 \cr 0.91 & -0.42 \cr 0.14 & -0.99} \right\rbrack $$

### 3. Masked Self-Attention

The decoder's self-attention is slightly different from that of the encoder. While the encoder can access all tokens at once, the decoder generates one token at a time. To prevent the decoder from "seeing" future tokens (which it shouldn't know yet), we use masked self-attention. This masking ensures that the decoder only attends to previous tokens and not future ones.

We implement this by adding a mask matrix to the scaled similarity scores. The mask assigns 0 to values we want to include in the attention calculation and -infinity to values we need to ignore. When passed through the softmax function, the -infinity values result in zero probability, effectively ignoring those future tokens.

$$ Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d_k}}+M)V $$

where $$M$$ is the mask matrix.

$$M = \left\lbrack \matrix{0 & -inf & -inf & -inf \cr 0 & 0 & -inf & -inf \cr 0 & 0 & 0 & -inf \cr 0 & 0 & 0 & 0} \right\rbrack$$


### 4. Residual Connection

Again, we add the original word embeddings and positional encodings to the self-attention outputs.

### 5. Encoder-Decoder Attention

This step is similar to self-attention but involves both the encoder's output and the decoder's current state. The encoder's output provides the $$V$$ and $$K$$ matrices, while the decoder's output provides the $$Q$$ matrix.

$$ Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d_k}})V $$

where $$Q$$ has a shape of 4 $$\times$$ 2, while both $$K$$ and $$V$$ have shapes of 3 $$\times$$ 2. The resulting encoder-decoder attention score matrix is therefore 4 $$\times$$ 2 in size.

### 6. Residual Connection

The encoder-decoder attention scores are combined with residual connections before being passed to the next layer.

### 7. Fully-Connected Layer

The output from the previous step is passed through a fully connected layer. Given our example, the layer has two input dimensions (from the attention outputs) and six output dimensions (corresponding to the tokens in the output vocabulary). The weights matrix has a shape of (2 $$\times$$ 6), and bias terms are added during this step. As a result, this layer produces a matrix with a shape of 4 $$\times$$ 6.

### 8. SoftMax

Finally, the fully connected layer's output is passed through a softmax function, which generates the probability distribution over the vocabulary for the next word in the sequence. The $$i$$-th row represents the index of the token generated by the decoder at time $i$.

---
#### Resources
 
- [Transformer: Attention is All You Need](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#transformer_intro)
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=20)
- [The matrix math behind transformer neural networks, one step at a time!!!](https://www.youtube.com/watch?v=KphmOJnLAdI&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=25)

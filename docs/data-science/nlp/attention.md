---
title: Attention is All You Need
parent: Natural Language Processing
nav_order: 3
layout: default
---

In the Seq2Seq model that we explored, the encoder processes input data and produces a **context vector**- a single vector that encapsulates the entire input sequence. This vector is then passed to the decoder, which uses it to generate the output sequence. However, it's important to consider the challenge of compressing all the information from the input sequence into a single vector. This compression likely leads to a loss of valuable information, as the context vector may not fully reflect the intricacies of the entire input.

Additionally, during the generation of the output sequence, the decoder might struggle to produce accurate results when relying solely on a single fixed representation of the source. Different parts of the input sequence can be more relevant at different stages of output generation. This limitation—where the context vector becomes a bottleneck—motivates the introduction of the **Attention** mechanism, which addresses the issue of fixed representation by allowing the model to focus on different parts of the input dynamically.

## Attention Mechanism: A Solution to the Fixed Representation Problem

The core idea of attention is to create additional pathways from the encoder to the decoder, enabling the decoder to focus on specific parts of the input sequence at each step of the output generation process. Consider the following example:

<p align="center">
  <img src="https://github.com/user-attachments/assets/99a662df-07f7-46bc-b9db-cec623412a52" title="attention">
</p>

In this scenario, we want to translate the Spanish sentence "te quiero" to the English phrase "I love you". After the encoder processes all tokens in the input sentence, the model is ready to predict the first word in the English sentence. Attention begins by assessing the similarity between each of the encoder's hidden states and the first hidden state in the decoder. This similarity is quantified through attention scores, such as $$score(s_1, h_1)$$, $$score(s_2, h_1)$$, and $$score(s_3, h_1)$$. 

Next, these scores are passed through a softmax function to produce attention weights. The softmax function normalizes the scores, converting them into values between 0 and 1 that sum to 1. These weights indicate the significance of each encoded input word in determining the first translated word. Finally, the attention output is calculated as a weighted sum of the encoder states, using the attention weights.

For a decoder time step $$t$$, given $$m$$ encoder states, the general computation scheme is as follows: 

1. **Attention Scores**:

$$score(h_t, s_k)$, where $k=1,...,m$$

2. **Attention Weights**:

$$a_k^{t}=\frac{e^{score(h_t, s_k)}}{\sum_{i=1}^{m}e^{score(h_t, s_i)}}$$

3. **Attention Output**:

$$c^{t}=a_1^{(t)}s_1+a_2^{(t)}s_2+...+a_m^{(t)}s_m = \sum_{k=1}^{m}a_k^{(t)}s_k$$

### Computing Attention Scores

There are several ways to compute attention scores, but two of the most popular methods are Bahdanau Attention and Luong Attention.

- **Bahdanau Attention** (Additive Attention):

  - The attention score is computed using a feedforward neural network. The decoder's hidden state $$h_{t-1}$$ is combined with each encoder's hidden state $$s_k$$ to compute the attention score. The resulting context vector $$c^{(t)}$$ is then used, along with $$h_{t-1}$$, as input to the decoder at time step $$t$$.
    
$$ score = v_a^T \cdot tanh (W_a \cdot [s_k^T; h_{t-1}] $$
    
- **Luong Attention** (Multiplicative Attention):
  
  - The attention score is calculated as the dot product between the decoder's hidden state and each encoder's hidden state. Luong proposed three variants: dot, general, and concat.
    
    - Dot: The simplest variant, where the attention score is the dot product of the decoder and encoder hidden states. $$score = s_k^T \cdot h_t$$

    - General: Similar to dot, but introduces a learned weight matrix. $$score = s_k^T \cdot W_a \cdot h_t$$

    - Concat: Similar to Bahdanau's method but with a slightly different formulation.

Bahdanau attention is often considered more flexible but computationally more expensive. Luong attention is simpler and faster, particularly in the dot-product form.

---
#### Resources
- [Sequence to Sequence (seq2seq) and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#attention_intro)
- [Attention for Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=PSs6nxngL6k&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=19)

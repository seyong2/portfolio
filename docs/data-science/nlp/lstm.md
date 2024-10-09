---
title: Breaking Down LSTM
parent: Natural Language Processing
nav_order: 2
layout: default
---

## Recurrent Neural Networks

Imagine we want to build a binary classification model to predict whether a phrase contains sarcasm. For instance, consider these two sentences: "You have just broken my favorite dish, great job" and "Thanks for helping out, great job". Although both sentences include the phrase "great job", the first is sarcastic, while the second is sincere. However, traditional neural networks may struggle to distinguish between the two because they don't account for the order of the words. Without considering word sequence, the model might incorrectly classify both sentences as sincere. 

Recurrent Neural Networks (RNNs) can effectively address this challenge. RNNs are a type of artificial neural network specifically designed for processing sequential data, such as time series, text, and audio, where the order of the data points is crucial. RNNs can retain information from previous inputs in a sequence through loops in their structure.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b032ffc2-b193-45e7-a085-729a259ba9eb" title="rnn">
</p>

In the diagram above, a portion of the neural network (represented by the gray box in the middle, labeled as $$h_t$$) receives an input $$x_t$$ and produces an output $$y_t$$ at time step $$t$$. The hidden state $$h_t$$ can be expressed as follows:

$$ h_t = tanh(w_x \times x_t + w_h \times h_{t-1} + b) $$

At first glance, it might not be immediately clear why $$h_t$$ depends on $$h_{t-1}$$. However, by examining the unrolled version of the network, we can better understand the structure of RNNs. A recurrent neural network can be visualized as multiple copies of the same network, each passing information to the next step in the sequence. This design means that we only have three parameters- $$w_x$$, $$w_h$$, and $$b$$- which are shared across all copies of the network.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cb2f3c4f-b476-4a3e-b495-4c5bd01ba8b8" title="rnn-unrolled">
</p>

### Gradient Vanishing and Exploding

While RNNs are well-suited for processing sequential data effectively due to their structure, there are scenarios where they may struggle. Consider a situation where we have a longer text, and we want to predict the final word of the text "I went grocery shopping and bought a steak. Then, I met up with my friend, and we had lunch together... When I came back home, I prepared dinner with what I bought earlier, the *steak*". The recent context suggests that the next word is likely a type of food, but to pinpoint exactly which food, we need to recall the mention of steak from much earlier in the text. As the gap between the relevant information and the point where it's needed increases, RNNs struggle to make the connection. This difficulty in learning long-term dependencies is a significant limitation of RNNs. Let's explore this issue with a simple example.

To illustrate this, let's consider a simpler example. Suppose we want to predict the last word of the sentence: "The mysterious forest echoed with sounds of unknown creatures at dusk." An RNN model tasked with predicting the final word operates as shown in the diagram below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e163de4-b9ef-4abd-971c-69c74da98bb7" title="rnn-problem">
</p>


In this model, the hidden state at each time step, $$h_t$$, has access to the current input, $$x_t$$, and the previous hidden state, $$$$h_{t-1}$$$$. The hidden state $$h_{t-1}$$ implicitly captures information about all the previous words in the sentence, from $$x_0$$ to $$x_{t-1}$$. Therefore, the final hidden state, $$h_9$$, which we want the model to use for predicting the next word, depends on all the previous words. To ensure accurate predictions, the model's weights and bias must be optimized through backpropagation. However, since RNNs deal with sequential data, they utilize a variant of the backpropagation algorithm known as backpropagation through time (BPTT). This process involves computing gradients for each time step, starting from the last one and moving backward to the first.

If we want to compute the gradient of the error, $$E$$ (the difference between the prediction and target), with respect to the weight associated with the hidden states, $$w_h$$, the gradient would be expressed as follows:

$$ \frac{\partial E}{\partial w_h} = \frac{\partial E}{\partial h_9} \cdot \frac{\partial h_9}{\partial w_h} + \frac{\partial E}{\partial h_9} \cdot \frac{\partial h_9}{\partial h_8} \cdot \frac{\partial h_8}{\partial w_h} + ... + \frac{\partial E}{\partial h_9} \cdot \frac{\partial h_9}{\partial h_8} \cdot \frac{\partial h_8}{\partial h_7} \cdot \frac{\partial h_7}{\partial h_6} \cdot \frac{\partial h_6}{\partial h_5} \cdot \frac{\partial h_5}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_h}$$

This equation simplifies to:

$$ \frac{\partial E}{\partial w_h} = \frac{\partial E}{\partial h_9} \cdot \frac{\partial h_9}{\partial w_h} + \frac{\partial E}{\partial h_9} \cdot w_h \cdot \frac{\partial h_8}{\partial w_h} + ... + \frac{\partial E}{\partial h_9} \cdot w_h \cdot w_h \cdot w_h \cdot w_h \cdot w_h \cdot w_h \cdot w_h \cdot w_h \cdot \frac{\partial h_1}{\partial w_h}$$

$$ \frac{\partial E}{\partial w_h} = \frac{\partial E}{\partial h_9} \cdot \frac{\partial h_9}{\partial w_h} + \frac{\partial E}{\partial h_9} \cdot w_h \cdot \frac{\partial h_8}{\partial w_h} + ... + \frac{\partial E}{\partial h_9} \cdot w_h^8 \cdot \frac{\partial h_1}{\partial w_h}$$

Here, it's evident that the gradient contains multiple multiplications of the weight, $$w_h$$. If the value of $$w_h$$ is less than 1, the gradient can become close to zero, leading to what is known as **gradient vanishing**. In this situation, the weight updates during training will be minimal.

$$ w_h = w_h - \alpha \times \frac{\partial E}{\partial w_h} $$

where $$\alpha$$ is the learning rate.

Conversely, if $$w_h$$ is greater than 1, repeated multiplication can cause the gradient to grow exponentially, resulting in **gradient exploding**, which can destabilize the weight updates. Consequently, RNNs often struggle with long sequences due to these gradient issues, making them less effective in handling long-term dependencies.

## Long Short Term Memory Networks

The issue of long-term dependencies, which RNNs struggle with, can be addressed by using a variant called Long Short Term Memomry Networks (LSTMs). While The structure of a standard RNN is relatively simple, with a single tanh activation layer, LSTMs introduce a more complex architecture. Like RNNs, LSTMs have a chain of repeating modules, but they differ in that each module contains more layers, allowing them to better manage the flow of information over time. In this section, we'll explore the function of each layer in an LSTM and how it enables the network to carry forward important past information, unlike RNNs. But first, let's take a look at the basic structure of LSTMs. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/37fc3149-1e56-46b3-ad60-b1d8c8969bf2" title="lstm">
</p>

A central component of the LSTM structure is the cell state, denoted as $$C_t$$. The cell state acts like a conveyor belt, running straight through the entire chain with minimal linear interactions, allowing information to flow largely unchanged. So, how does the cell state differ from the hidden state? 

Imagine an LSTM cell as an office worker tasked with handling information over time. In this analogy, the cell state is like a filing cabinet that stores all the important documents that the worker needs to reference or update over time. It holds both recent and older documents, allowing the worker to access long-term information when needed. On the other hand, the hidden state is like the desk where the worker places the documents they are actively working on. The desk only holds a subset of the documents from the filing cabinet, specifically those that are relevant to the current task or moment.

Let's walk through a single-time step in an LSTM to see how the cell state and hidden state interact.

### Forget Gate

The first step in an LSTM is to decide which information from the cell state should be discarded. This decision is made by the forget gate, a sigmoid layer that outputs values between 0 and 1. Based on the input $$x_t$$ and the previous hidden state $$h_{t-1}$$, the forget gate determines how much of the previous cell state $$C_{t-1}$$ should be retained. For example, if the forget gate outputs 0.7, it means that 70 percent of the information from $$C_{t-1}$$ will be kept, while 30 percent will be forgotten as it's deemed irrelevant.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1724436-5d86-4793-ad70-b3763289265f" title="lstm-forget">
</p>

$$ f_t=\sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

- $$W_f$$ is the weight matrix for the forget gate.
- $$b_f$$ is the bias for the forget gate.

### Input Gate

After discarding unneeded information from $$C_{t-1}$$, the next step is to add new information. The input gate determines how much of the new candidate cell state $$C̃_t$$ should be added to the current cell state C_t. The candidate cell state is the new information that could update the cell state.

<p align="center">
  <img src="https://github.com/user-attachments/assets/26352159-b724-4bc5-bd5c-516f1502f282" title="lstm-input">
</p>

$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$

- $$W_i$$ is the weight matrix for the input gate.
- $$b_i$$ is the bias for the input gate.


$$ C̃_t = tanh(W_c \cdot [h_{t-1}, x_t] + b_C) $$

- $$W_c$$ is the weight matrix for the candidate cell state.
- $$b_c$$ is the bias for the candidate cell state.

Now, we can update the old cell state $$C_{t-1}$$ into the new cell state $$C_{t}$$. We multiply the old state by $$f_t$$ to forget the information we decided to discard. Then, we add $$i_t * C̃_t$$ which is the amount of new information added. The new cell state $$C_t$$ is defined as follows:

$$ C_t = f_t * C_{t-1} + i_t * C̃_t$$

### Output Gate

The final step is to determine how much of the cell state $$C_t$$ should be output as the hidden state $$h_t$$ for the current time step.

<p align="center">
  <img src="https://github.com/user-attachments/assets/643d1eb2-8073-47b7-8ba0-e7293351a1ab" title="lstm-output">
</p>

$$o_t=\sigma(W_o\cdot [h_{t-1}, x_t] + b_o)$$

- $$W_o$$ is the weight matrix for the output gate.
- $$b_o$$ is the bias for the output gate.

Next, we pass the cell state through a tanh function and multiply it by $$o_t$$, so that only the relevant parts are output. The hidden state $$h_t$$ serves as input to the next cell or as the output of the model. 

This gating mechanism enables LSTMs to selectively retain or discard information as they process each input, allowing them to effectively learn and capture long-term dependencies in the data.

---
#### Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=L8HKweZIOmg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=18)
- [딥러닝: LSTM 쉽게 이해하기](https://www.youtube.com/watch?v=bX6GLbpw-A4&t=349s)
- [LSTM Networks: Explained Step by Step!](https://www.youtube.com/watch?v=P_TZN8kRObQ&t=713s)

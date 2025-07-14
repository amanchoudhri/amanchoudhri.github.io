---
title: A Close Reading of Self-Attention
subtitle: Framing self-attention in terms of convex combinations and similarity matrices,
  aiming to precisely ground common intuitive explanations.
date: 2025-07-12
thumbnail: attn-close-reading.png
filetitle: attention-close-reading
---
*This article assumes comfort with linear algebra. It's designed for those who have seen the self-attention formula but haven't yet found the time to really understand it (i.e., me).*


Self-attention is a mechanism in deep learning that allows positions in sequences to aggregate information from other positions of their 'choosing', depending on the specific input. We'll make this intuition precise.


For starters, self-attention is a sequence-to-sequence map. Each element of the output sequence is a *weighted average* of *feature vectors*, where the weights and feature vectors are determined from the input sequence.


Mathematically: Self-attention maps an \(S\)-element sequence of \(d\)-vectors, \(x \in \mathbb{R}^{S \times d}\), to another \(S\)-element sequence of \(d_v\)-vectors, \(\text{Attention}(x) \in \mathbb{R}^{S \times d_v}\). Its elements are given by \[\text{Attention}(x)_{i} = A_{i}V = \sum_{j} A_{ij} V_j,\]for a weight matrix \(A\) whose rows \(A_i\) sum to one: \(\sum_j A_{ij} = 1\). The \(i\)th output vector is the weighted average of feature vectors \(V_{j} \in \mathbb{R}^{d_v}\).


## Feature vectors, \(V\)
The feature vectors are a linear projection of the input. For an input sequence \(x \in \mathbb{R}^{S \times d}\), the feature matrix \(V \in \mathbb{R}^{S \times d_{v}}\) is
\[
V = xW^V,
\]
for some low-dimensional projection matrix \(W^V \in \mathbb{R}^{d \times d_{v}}\). Think of \(V\) as a *batched* projection of \(x\), so each sequence element is passed through the same projection.


![Handdrawn illustration of the equation V = xW^V](/assets/img/attn-close-reading-V.png)


## Attention matrix, \(A\)
The 'weight' matrix \(A \in \mathbb{R}^{S \times S}\) is a *normalized similarity matrix*, where each entry \(A_{ij}\) represents a measure of "alignment" between \(x_{j}\) and \(x_{i}\). Alignment is measured through the kernel matrix,\[
xMx^T,
\]for some learnable matrix \(M \in \mathbb{R}^{d \times d}\). Each element \(ij\) represents the dot product (in the geometry defined by \(M\)) between input vectors \(x_i\) and \(x_j\). The matrix \(M\) may be asymmetric, so\[
(xMx^T)_{ij} \neq (xMx^T)_{ji},
\]in general. 
![Handdrawn image illustrating the matrix xMx^T](/assets/img/attn-close-reading-xMx.png)


The attention matrix \(A\) is constructed by passing this kernel matrix through an element-wise nonnegative function \(\psi\), then normalizing each row to have unit sum. Specifically,
\[
A_{ij} := \frac{\psi(x_{i}Mx_{j}^T)}{\sum_{k} \psi(x_{i}Mx_{k}^T)}
\]
In practice, \(\psi\) is often chosen to be \(\psi = \text{exp}\).[^1] With this choice, \(A\) can be concisely expressed as the result of a softmax operation applied rowwise: \[
A = \text{softmax}_{\text{rowwise}}\left(xMx^T \right).
\]


## Convex combinations, \(AV\)
With \(V\) and \(A\) defined, we can pull together a precise understanding of self-attention.


Self-attention maps one sequence to another of the same length. Each element \(\text{Attention}(x)_i\) in the output sequence is a convex combination of linear features \(V\) of the input sequence \(x\).
\[
\text{Attention}(x)_{i} = A_{i}V = \sum_{j} A_{ij}V_{j}
\]
Each combination weight \(A_{ij}\) is determined based on the kernel similarity between \(x_i\) and \(x_j\), normalized by its similarities to all other sequence elements.[^2]


## Parameterizing \(M\)
In practice, we don't learn the bilinear form matrix \(M\) directly. Instead, we constrain its rank (and hopefully make it easier to learn) by parameterizing as a product of lower-dimensional factors.


We pre-specify a maximum rank \(d_k\), and parameterize \(M\) in terms of learnable \(d_k \times d\) matrices \(W^Q, W^K\), as follows:
\[
M = \frac{1}{\sqrt{d_{k}}} (W^Q)(W^K)^T.
\]
The product is scaled by \(\frac{1}{\sqrt{ d_{k} }}\) to limit the magnitude of the row entries in practice. The goal is to avoid pushing the softmax inputs into regions where the nonlinearity has very small gradients.


## The usual notation
Translating from our expression
\[
\text{Attention}(x) = \text{softmax}_{\text{rowwise}}(xMx^T)V
\]
to the form generally seen in papers,
\[
\text{Attention}(Q, K, V) = \text{softmax}_{\text{rowwise}}\left( \frac{QK^T}{\sqrt{ d_{k} }} \right)V
\]
is quite simple.


To do so, we define the usual \(Q, K\) matrices by "pulling" the factors \(W^Q\) and \(W^K\) onto each \(x\) term in the kernel matrix computation. Specifically, define \(Q, K \in \mathbb{R}^{S \times d_{k}}\) as
\[
Q = xW^Q \quad \text{and} \quad K = xW^K,
\]
Like with the projection matrix \(W^V\) above, this operation is best understood as a "batched projection." The two notations are then trivially equivalent, with\[
xMx^T = x \left(  \frac{1}{\sqrt{ d_{k} }} (W^Q) (W^K)^T \right)x^T = \frac{1}{\sqrt{ d_{k} }} (xW^Q)(xW^K)^T = \frac{1}{\sqrt{ d_{k} }}QK^T.
\]All together:\[
\text{Attention(x)} = AV = \text{softmax}(xMx^T)V = 
\text{softmax}\left( \frac{QK^T}{\sqrt{d_{k} }} \right) V.
\]


## So what?
Abstractly, this article frames self-attention as the composition
\[
\text{Attention}(x)_{i} = \big[ \text{Normalize}(\text{Kernel}(x_{i}, x)) \big] \text{Features}(x).
\]
In words, self-attention computes an output sequence where each element \(i\) is a convex combination of feature vectors \(j\). The contribution of vector \(j\) to the output (the magnitude of its combination weight, \(A_{ij}\)) is determined by a normalized kernel similarity between input elements \(i\) and \(j\).


This framing makes apparent some otherwise hand-wavy observations.


### Observation 1: Self-attention reduces the 'distance' between tokens \(i\) and \(j\)
Self-attention is a powerful mechanism because it calculates each output element \(i\) by *directly* pulling information from other tokens \(j\). Compare this to the dependency between tokens in a recurrent neural network. Define the basic RNN as
\[
  \text{RNN}(x)_{t} := g(h_{t}) \quad \text{and} \quad
 h_{t} := f(x_{t}, h_{t - 1}).
\]
Each output computed, \(\text{RNN}(x)_{i}\), is the result of \(i\) recursive applications of \(g\) and \(f\) along the input sequence. The information about early tokens \(x_1, ..., x_i\) is bottlenecked through the single "hidden state" representation \(h_i\).


In theory, \(h_i\) should be able to represent all the relevant information from the previous sequence tokens. But in practice, RNNs don't learn to compress this information well.[^3]


Self-attention sidesteps this compression problem. It removes the hidden state bottleneck and directly mixes features from other tokens.


In doing so, self-attention also parallelizes the sequence-to-sequence computation.


RNNs are inherently sequential, since their output at time \(t+1\) depends on the previous hidden state \(h_t\). But with self-attention, each output term \(\text{Attention}(x)_{i}\) and \(\text{Attention}(x)_{j}\) can be computed completely in parallel.


This parallelization, however, does come at a cost.


### Observation 2: Self-attention has no "interaction terms"
Because self-attention parallelizes the similarity computation for output \(i\) across all input elements \(j\) at once, it's quite restricted in the way it can combine information from input tokens.


Specifically, self-attention can only 'combine' information from tokens \(j, k\) into \(\text{Attention}(x)_{i}\) through the linear mixing
\[
A_{ij}V_{j} + A_{ik}V_{k}.
\]
Even the linear mixing weights \(A_{ij}\) and \(A_{ik}\) depend largely only on the *individual* similarities \(\kappa(x_i, x_{j})\) and \(\kappa(x_i, x_{k})\) . This turns out to limit the kinds of computations that (a single layer of) self-attention can perform.[^4]


Various generalizations of self-attention aiming to address this limitation by incorporating interaction terms have been proposed.


The basic concept of "higher-order attention" is simple, and fits nicely into the framework we've discussed. For concreteness, let's consider third-order attention.[^5]


There are two key changes. Rather than mixing individual features \(V_j\) associated with individual tokens \(x_j\), we mix *pair* features \(V_{j, k}\) associated with token *pairs* \(x_{j, k}\). Rather than defining the mixing weights for output \(i\) using the similarity between \(x_{i}\) and all other tokens \(x_{j}\), we mix the pair features based on the similarity between \(x_i\) and all *pairs* of tokens \(x_j, x_k\).


The final form of third-order attention is semantically equivalent to standard attention. It composes an output sequence composed by linearly mixing *interaction* features from the input:
\[
\text{Attention}_{3}(x)_{i} = \sum_{j, k}\big[ \text{Normalize}(\text{Kernel}(x, x, x)) \big]_{i, j, k} \text{Features}_{j,k}(x)
\]
Unfortunately, there are no free lunches. Higher-order attention is significantly more expensive. Standard self-attention is \(O(S^2)\), but third-order attention is \(O(S^3)\), since we now need to compute an \(S \times S \times S\) attention *tensor* rather than an \(S \times S\) attention matrix.


## Conclusion
We've now thoroughly examined each component of self-attention. The self-attention mechanism is a sequence-to-sequence transformation that computes each output as a weighted average of features, where the weights are based on normalized similarities between input sequence elements. This framing precisely undergirds the intuition that self-attention allows positions to "look at" and "choose" information from other positions.



----

[^1]: Some interesting papers exploring other nonlinearities: [1](https://arxiv.org/abs/2202.08791), [2](https://arxiv.org/abs/2409.04431), [3](https://arxiv.org/abs/2309.08586). Maybe another blog post to come on this.


[^2]: In practice, we restrict the \(i\)th output to only pull from the first \(i\) tokens. This is called *causal masking*, and is another main reason why transformers are so good at autoregressive generation.


[^3]: There are mathematical arguments for the difficulty of learning these "long-range dependencies," as well. The rough idea is that the contribution of an input token \(i\) to a loss at output token \(j\) decays to zero for \(i \ll j\). See [(Bengio et al, 1994)](https://ieeexplore.ieee.org/author/37323338000) and [(Hochreiter and Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) for more.


[^4]: For example: [(Sanford et al, 2023)](https://arxiv.org/abs/2306.02896) introduce a task, \(\verb|Match3|\), that is very difficult for standard attention but trivially solvable for a "higher-order" attention taking into account pairwise interactions. For a single self-attention layer, solving \(\verb|Match3|\) requires an embedding dimension or number of heads that is *polynomial* in the input sequence length, \(S\). By contrast, a "third-order" self-attention layer can solve \(\verb|Match3|\) for *any* sequence length with *constant* embedding dimension.


[^5]: The exposition here is roughly based on [(Clift et al, 2019)](https://arxiv.org/abs/1909.00668) and its recent computationally-efficient reformulation in [(Roy et al, 2025)](https://arxiv.org/abs/2507.02754). These papers refer to third-order attention as "2-simplicial."


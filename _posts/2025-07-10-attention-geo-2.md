---
title: To Be Determined
subtitle: Not quite sure yet!
date: 2025-07-10
thumbnail: null
filetitle: attention-geo-2
---
## Masking (DRAFT)
In the "causal" variant of self-attention, the output attention sequence is constrained in the feature vectors it can pull from. Specifically, we restrict \(\text{Attention}(x)_{i}\) to only depend on the first \(i\) features, \(V_1, ..., V_i\). The goal is to develop representations that respect the 'forward direction' of language.


Define \(\text{Mask}(B)\) to zero all entries \(B_{ij}\) for \(j > i\), then renormalize the rest. Specifically,\[
\text{Mask}(B)_{ij} = \begin{cases}
B_{ij} / \sum_{k \leq i} B_{ik} & j \leq i \\
0 & j > i,
\end{cases}
\]for any \(B \in \mathbb{R}^{S \times S}\). Then causal attention is simply
\[
\text{Attention}_{\text{causal}}(x) := \text{Mask}(A)V,
\]
for \(A\) the usual attention map and \(V\) the usual feature vectors.


For computational efficiency, masking is usually implemented differently in practice. Rather than zeroing entries and renormalizing after the softmax, standard code implementations add \(-\infty\) to the entries \(j > i\) *within* the softmax, taking the convention that \(\exp(-\infty) = 0\).


Specifically, let \(U\) be an upper-triangular matrix such that \(U_{ij} = \mathbb{1}(i < j)\). The masked attention matrix is usually calculated as
\[
\text{Mask}(A) = \text{softmax}_{\text{rowwise}}(xMx^T + (-\infty)U).
\]
In code,


```python
# DRAFT
# something about (-np.inf) * np.triu(...)
$test$
```
## Multi-Head Attention (DRAFT)
Subspace-level mixing. Linear recombination that ignores the block structure.


Papers that look into doing things better.
http://arxiv.org/abs/2105.14850: probabilistic model. very hard to tell what's going on to be honest.
http://arxiv.org/abs/2410.11842: input-dependent weighting for the recombination


## Positional Encodings (DRAFT)
Unlike convolutions or recurrent networks, there's nothing inherently "order-preserving" or "order-aware" about the self-attention layer. Yes, causal masking acts as a kind of information bottleneck, ensuring that \(\text{Attention}(x)_{i}\) can only depend on \(V_1, ..., V_i\). But other than that, the convex combination weights \(A_{ij}\) have no information about the relative positions of \(i\) and \(j\). The output \(\text{Attention}(x)_i\) is therefore completely invariant to permutations of the previous tokens \(x_1, ..., x_{i - 1}\). See the appendix for a proof of this property.


To solve this, add some position-dependent vector \(p_i\) to each token \(x_i\) before projecting to form \(K, Q, V\). So
\[
K = W^{K}(x + P) = W^Kx + W^KP.
\]
Similarly, \(Q = W^Qx + W^Q P\)  and \(V = W^V x + W^V P\). This affects the attention map computation as
\[
(x + P)M(x + P)^T = xMx^T + xMP^T + PMx^T + PMP^T
\]


In *Attention is All You Need*, Vaswani defined \(p_i\) across dimensions as follows. Partition \(d_\text{model}\) into two-dimensional subspaces. For each subspace \(k = 1, ..., d_\text{model}/2\), define a frequency \(\phi_k\) as
\[
\phi_{k} = 10000^{2k / d_{\text{model}}}
\]
Then defined the vector \(p_i\) as follows:
\[
p_{i, 2k} := \sin(i/\phi_{k}) \quad \text{and} \quad p_{i, 2k+1} := \cos\left( i / \phi_{k} \right)
\]


## Appendix: Permutation Invariance
Earlier, we claimed that, without positional encodings, \(\text{Attention}(x)_i\) is completely invariant to permutations of the previous tokens \(x_1, ..., x_{i - 1}\).


Let's make this precise. Define \(\pi: \left\{ 1, \dots, S \right\} \to \left\{ 1, \dots, S \right\}\) to be any permutation that affects only \(x_1, ..., x_{i - 1}\). So for any \(j \geq i\), the permutation \(\pi\) satisfies \(\pi(j) = j\). Let \(\Pi \in \mathbb{R}^{S \times S}\) be the matrix representation of \(\pi\).[^1] So
\[
\Pi x = \begin{bmatrix}
x_{\pi^{-1}(1)} \\ \vdots \\ x_{\pi^{-1}(S)}
\end{bmatrix}
= \begin{bmatrix}
x_{\pi^{-1}(1)} \\ \vdots \\ x_{\pi^{-1}(i - 1)} \\ x_{i} \\ \vdots \\ x_{S}
\end{bmatrix},
\]based on our definition of \(\pi\).


Our goal is to show that
\[
\text{Attention}(\Pi x)_{i} = \text{Attention}(x)_{i}.
\]
Self-attention on the permuted input is given by
\[
\text{Mask}\bigg(\text{softmax}_{\text{rowwise}}\big((\Pi x)M(\Pi x)^T\big) \bigg) (\Pi V)
\]
We'll start with the similarity kernel. Obviously, we have \[
(\Pi x) M (\Pi x)^T  = \Pi (xMx^T)\Pi^T
\]And since the softmax operation is definitionally applied row-wise, we have
\[
\text{softmax}_{\text{rowwise}}(\Pi (xMx^T)\Pi^T) = \Pi \ \text{softmax}_{\text{rowwise}}((xMx^T)\Pi^T)
\]
Within each row, the softmax respects column permutations. To see this, take any vector \(y \in \mathbb{R}^{S}\), and note that
\[
\text{softmax}(y_{\pi^{-1}(1)}, \dots, y_{\pi^{-1}(S)})_{\pi(i)} = \text{softmax}(y_{1}, \dots, y_{S})_{i}
\]
So \(\text{softmax}(y\Pi^T) = \text{softmax}(y) \ \Pi^T\). In particular, this means
\[
\Pi \ \text{softmax}_{\text{rowwise}}((xMx^T) \Pi^T) = \Pi \ \text{softmax}_{\text{rowwise}}(xMx^T) \ \Pi^T = \Pi A \Pi^T.
\]
Next, we consider the masking operation, \(\text{Mask}(\Pi A \Pi^{T})\). We'll aim to show that
\[
\text{Mask}(\Pi A \Pi^T)_{i} = (\text{Mask}(A)_{i} )\Pi^T.
\]
Right-multiplication of a matrix by \(\Pi^T\) applies the same permutation \(\pi\) but to the columns, meaning\[
(A\Pi^T)_{k, \pi(j)} = A_{kj}.
\]Therefore,
\[
(\Pi A \Pi^T)_{i} = (\Pi A)_{i} \Pi^T = A_{i}\Pi^T = \begin{bmatrix}
A_{i, \pi^{-1}(1)} & \ldots & A_{i, \pi^{-1}(i - 1)} & A_{ii} & A_{i, i+ 1} & \dots & A_{i, S}
\end{bmatrix}
\]
Recall that \(\text{Mask}(B)\) is given by \[
\text{Mask}(B)_{ij} = \begin{cases}
B_{ij} / \sum_{k \leq i} B_{ik} & j \leq i \\
0 & j > i,
\end{cases}
\]for any \(B \in \mathbb{R}^{S \times S}\). So \(\text{Mask}(\Pi A \Pi^T)_{i}\) is given by\[
\text{Mask}(\Pi A \Pi^T)_{i} = \frac{1}{\sum_{k \leq i} A_{i, \pi^{-1}(k)}}\begin{bmatrix}
A_{i, \pi^{-1}(1)} & \dots & A_{i, \pi^{-1}(i - 1)} & A_{ii} & 0 & \dots & 0
\end{bmatrix}.
\]And since \(\pi\) maps the set \(\left\{1, ..., i\right\}\) to itself, the normalizing sum satisfies
\[
\frac{1}{\sum_{k \leq i} A_{i, \pi^{-1}(k)}} = \frac{1}{\sum_{k \leq i}A_{ik}},
\]
so\[\text{Mask}(\Pi A \Pi^T)_{i} = \frac{1}{\sum_{k \leq i} A_{ik}}\begin{bmatrix}
A_{i, \pi^{-1}(1)} & \dots & A_{i, \pi^{-1}(i - 1)} & A_{ii} & 0 & \dots & 0
\end{bmatrix} = \text{Mask}(A)_{i} \Pi^T.\]With this, we can conclude:\[
\text{Mask}(\Pi A \Pi^T)_{i}(\Pi V) = \big((\text{Mask}(A)_{i})\Pi^T \big) \big(\Pi V \big) = \text{Mask}(A)_{i} (\Pi^T \Pi)V = \text{Mask}(A)_{i}V,
\]since \(\Pi^T \Pi = I\). So \(\text{Attention}(\Pi x)_{i} = \text{Attention}(x)_{i}\)



----

[^1]: Recall that a row permutation matrix is itself just a row-permuted identity matrix. For example, consider the 2d 'row-swapping' permutation \[
	\Pi = \begin{bmatrix}
	0 & 1 \\ 1 & 0
	\end{bmatrix}.
	\]Letting \(x\) be the sequence \(x_1 = (1, 2)\) and \(x_2 = (3, 4)\), we have
	\[
	\Pi X = \begin{bmatrix}
	0 & 1 \\ 1 & 0
	\end{bmatrix} \begin{bmatrix}
	1 & 2 \\ 3 & 4
	\end{bmatrix} = \begin{bmatrix}
	3 & 4 \\ 1 & 2
	\end{bmatrix}
	\]


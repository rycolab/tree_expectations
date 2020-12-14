# Tree Expectations
This library contains an implementation for efficient
computation of zeroth-, first-, and second-order expectations
under spanning tree models.
A detailed description of these algorithms including proofs of correctness and rumtime can be found in
["Efficient Computation of Expectations under Spanning Tree Distributions"](https://arxiv.org/abs/2008.12988).


## Citation

This code is for the paper _Efficient Computation of Expectations under Spanning Tree Distributions_. Please cite as:

```bibtex
@inproceedings{zmigrod-etal-2020-efficient,
    title = "Efficient Computation of Expectations under Spanning Tree Distributions",
    author = "Ran Zmigrod and Tim Vieira and Ryan Cotterell",
    journal = "Transactions of the Association for Computational Linguistics",
    year = "2020",
    url = "https://arxiv.org/abs/2008.12988",
}
```

## Requirements and Installation

* Python version >= 3.6
* PyTorch version >= 1.6.0

Installation:
```bash
git clone https://github.com/rycolab/tree_expectations
cd tree_expectations
pip install -e .
```

## Documenation Style
Variable names:

    w: input matrix
    rho: root weight vector
    A: adjacency matrix
    q: multiplicatively decomposable function
    r, s, f: additively decomposable function

We assume that w, r, s are structured as the root weight vector (rho)
along the diagonal and the rest is the adjacency matrix (A).

We give type annotations and use the following dimensions

    N = 'number of nodes'
    E = 'number of egdes (typically N^2)'
    R = 'Dimensionality of r function'
    S = 'Dimensionality of s function'
    F = 'Dimensionality of f function'
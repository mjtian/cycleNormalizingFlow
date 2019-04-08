### Readme for test.py

Read [pytorch tutoial part 2](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) first, everything based on it and some of my habits.

#### 1. Build a pytorch module

**Object:** make `python test.py` runs without error.

This `test.py` implements a simple multi-layer perceptron neural network(MLP) or full-connected network(FC). This kind of net is the basic and simplest net. At each layer, it goes like this:
$$
\textbf{Input: }\mathbf{x}\in[\text{batch},\text{input dimension}]; \textbf{Weight: } \mathbf{w} \in [\text{input dimension},\text{output dimension}] \\
\mathbf{y} = activation(\mathbf{w}\cdot\mathbf{x})
$$
So, the init api of this class should contain the *shape* of each layer, and *activations* of each layer.

#### 2. Build a reversed MLP

**Object:** make `python test2.py` runs without error.

**Don't** reimplement a new class, import code from `test.py`, and implement two dimension transformations MLP of $28*28\rightarrow50\rightarrow1$ and $1\rightarrow 10\rightarrow 28*28$, note that this kind of transformation is what a GAN is doing.
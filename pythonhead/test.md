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



#### 3. Test2 Again

I have moved test.py to `utils/layer`,  so you now have to make `test2.py` run again.

Examples can be found at `utils/layers`

Hint1: Since `test2.py` is not finished at the time I checked out test3 branch, `git cherry-pick` may be useful.

Hint2: This task is about import architecture in python, refer to what I have done at [my utils](https://github.com/li012589/NeuralRG/tree/master/utils). The key is how to write your `__init__.py` .



#### 4. pytest

First install `pytest` using `conda` or `pip`

Then run `pytest` at the folder `pythonhead`, you should see something like this:

```bash
=============================================== test session starts ================================================
platform darwin -- Python 3.6.8, pytest-3.2.5, py-1.4.33, pluggy-0.4.0
rootdir: /Users/lili/Documents/MySpace/cycleNormalizingFlow/pythonhead, inifile:
collected 2 items

test/test_example.py ..

============================================= 2 passed in 0.54 seconds ============================================
```

I have already wroten some simple test at `/test/test_example.py`. You can use this as example to make `test2.py` into a unit test at `/test/test_mlp.py`.



This is an way to use the concept Unit Test in python.
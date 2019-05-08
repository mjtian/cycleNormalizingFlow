#### 1. Build a pytorch module

This `test.py` implements a simple multi-layer perceptron neural network(MLP) or full-connected network(FC). This kind of net is the basic and simplest net. At each layer, it goes like this:
$$
\textbf{Input: }\mathbf{x}\in[\text{batch},\text{input dimension}]; \\ \textbf{Weight: } \mathbf{w} \in [\text{input dimension},\text{output dimension}] \\
\mathbf{y} = activation(\mathbf{w}\cdot\mathbf{x})
$$
So, the init api of this class should contain the *shape* of each layer, and *activations* of each layer.


#### 2. Build a reversed MLP

**Import **code from `test.py`, and implement two dimension transformations MLP of $28*28\rightarrow50\rightarrow1$ and $1\rightarrow 10\rightarrow 28*28$, note that this kind of transformation is what a GAN is doing.


#### 3. Simple bijective(NICE)

Here a **bijective** function is a function that map a set $V \in \mathbb{R}^n$ into another set $U \in \mathbb{R}^n$ and also have the inversed mapping of $U \rightarrow V$. One simple form of this is like :
$$
\begin{split}
&&\mathbf{F} = \mathbf{f}_0 \circ \mathbf{f}_1 \\
&\text{where }
&\mathbf{f}_0(\mathbf{x},\mathbf{y}) = (\mathbf{x}+\mathbf{v}(\mathbf{y}),\mathbf{y}) \\
&&\mathbf{f}_1(\mathbf{x},\mathbf{y}) = (\mathbf{x},\mathbf{y}+\mathbf{u}(\mathbf{x})) \\
&&\mathbf{v}: \mathbb{R}^{\frac{n}{2}} \rightarrow \mathbb{R}^{\frac{n}{2}} ;
\mathbf{u}: \mathbb{R}^{\frac{n}{2}} \rightarrow \mathbb{R}^{\frac{n}{2}} \\
&&\mathbf{x},\mathbf{y}\in \mathbb{R}^{\frac{n}{2}}
\end{split}
$$
So here your task is code a bijecitve network that parameterize this kind of function.

Template is given at `NICE.py`, related tests are given at `test/test_nice.py`.


#### 4. Abstraction of bijective test


#### 5. Abstraction of bijective net


#### 6. RealNVP

Implement this bijective function:
$$
\begin{split}
&&\mathbf{F} = \mathbf{f}_0 \circ \mathbf{f}_1 \\
&\text{where }
&\mathbf{f}_0(\mathbf{x},\mathbf{y}) = (\exp(\mathbf{sv}(\mathbf{y}))*\mathbf{x}+\mathbf{v}(\mathbf{y}),\mathbf{y}) \\
&&\mathbf{f}_1(\mathbf{x},\mathbf{y}) = (\mathbf{x},\exp(\mathbf{su}(\mathbf{x}))*\mathbf{y}+\mathbf{u}(\mathbf{x})) \\
&&\mathbf{v},\mathbf{u},\mathbf{sv},\mathbf{su}: \mathbb{R}^{\frac{n}{2}} \rightarrow \mathbb{R}^{\frac{n}{2}} ; \\
&&\mathbf{x},\mathbf{y}\in \mathbb{R}^{\frac{n}{2}}
\end{split}
$$

Write`realnvp.py` and `test/test_realnvp.py`.


#### 7. Probability Prior(Gaussian)

Write a class that has two method: **sample**; **logProbability**. 

This class take a shape list as input for init method.

Sample take one parameter: batchSize, and return a variables from Gaussian distribution of shape [batchsize, shape].

logProbability take one parameter: a pytorch variable of shape [batchsize, shape] and return the logprobability of each sample, so this is of shape [batchsize]. 


#### 8.Jacobian

Deduce the Jacobian of a NICE transformation here:
$$
jacobian_0=[\begin{smallmatrix}I&v\\0&I\end{smallmatrix}]\\
jacobian_1=[\begin{smallmatrix}I&0\\u&I\end{smallmatrix}]\\
$$
Deduce the Jacobian of a RealNVP transformation here:
$$
jacobian_0=[\begin{smallmatrix}e^{sv(y)}&xe^{sv(y)}sv+v\\0&I\end{smallmatrix}]\\
jacobian_1=[\begin{smallmatrix}I&0\\ye^{su(x)}su+u&e^{su(x)}\end{smallmatrix}]\\
$$
Re-implement the inverse and forward again, this time consider the change of probability and return the change of probability using the newly added `inverse/forwardLogjac` variable.

#### 9 Implement sample and probability

Now, when we init the `NICE` or `RealNVP` , we take another parameter as the prior. So this transforamtion is a transformation of probability distribution, it transforamtion the prior distribution to a distribution we want. So `sample` method will draw samples from the transformed distribution. And `logProbability` method will give the log probabilitys of a batch of  given samples.


#### 10. Download MNIST dataset and import into pytorch

Write a script to download and unzip MNIST data, one example can be seen at <https://github.com/wangleiphy/DL4CSRC/blob/master/1-bp/utils.py#L83>

Also, few lines below, there is a `random_draw` function add all this two function to `utils/MNISTtools.py`


#### 11. Training generative model on MNIST

Add a `main.py` and training realnvp on MNIST model, the process is like this: random draw a batch of MNIST data and let realnvp give the probabilitys of every samples from this batch, and mean this batch of probability and let negative this mean as loss and do gradients descent.



# From Deep Learning MindMap[brief doc]

# Deep Learning Briefing Document

## I. Introduction to Deep Learning Concepts

Deep Learning is a subset of machine learning that employs artificial neural networks with multiple layers to learn representations of data with multiple levels of abstraction. This briefing outlines key concepts, components, and methodologies within deep learning, drawing from the provided "Deep Learning mindmap.pdf" source.

## II. Core Components of a Neural Network

### A. Layers

Neural networks are structured in layers, which are the fundamental building blocks.

- **Input Layer:** "Comprised of multiple Real-Valued inputs. Each input must be linearly independent from each other." This layer receives the initial data.
- **Hidden Layers:** These are "Layers other than the input and output layers." A layer "usually receives weighted input, transforms it with a set of mostly non-linear functions and then passes these values as output to the next layer." The presence of multiple hidden layers is what defines "deep" learning.
- **Output Layer:** This layer produces the final output of the network, based on the transformations performed by the preceding layers.

### B. Weight Initialization

Proper initialization of weights is crucial for effective training.

- **All Zero Initialization:** This is a "mistake" because "if every neuron in the network computes the same output, then they will also all compute the same gradients during back-propagation and undergo the exact same parameter updates." This leads to a lack of asymmetry and prevents neurons from learning distinct features.
- **Initialization with Small Random Numbers:** This is the preferred method, as it ensures "neurons are all random and unique in the beginning, so they will compute distinct updates and integrate themselves as diverse parts of the full network." Values are typically drawn from a normal distribution with zero mean and small standard deviation.
- **Calibrating the Variances:** To ensure consistent output distribution across neurons and improve convergence, "you can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e., its number of inputs)."

### C. Activation Functions

An activation function "Defines the output of that node given an input or set of inputs." They introduce non-linearity into the network, enabling it to learn complex patterns. Common types include:

- ReLU (Rectified Linear Unit)
- Sigmoid / Logistic
- Tanh
- Softmax
- Leaky ReLU, PReLU, RReLU, ELU, SELU, and others.

## III. Training and Optimization

### A. Cost/Loss Functions

Cost or Loss functions quantify the error of a model's predictions compared to the true values. The goal of training is to minimize this function.

- **Maximum Likelihood Estimation (MLE):** Many cost functions, such as Least Squares and Cross-Entropy, are derived from MLE. It selects "the set of values of the model parameters that maximizes the likelihood function," intuitively maximizing "the 'agreement' of the selected model with the observed data." The natural logarithm of the likelihood function (log-likelihood) is often used for convenience due to its monotonic increasing nature.
- **Cross-Entropy:** Used to define the loss function, where "The true probability pi is the true label, and the given distribution qi is the predicted value of the current model." It's particularly common in classification tasks.
- **Quadratic Loss:** "Common, for example when using least squares techniques." It is "more mathematically tractable than other loss functions" and is symmetric, meaning errors of the same magnitude above or below the target result in the same loss.
- **0-1 Loss:** A simple function that assigns 0 if the prediction is correct and 1 if it's incorrect.
- **Hinge Loss:** Used for training classifiers, particularly for Support Vector Machines.
- **Exponential Loss:** Another function used in classification.
- **Hellinger Distance:** Quantifies the similarity between two probability distributions.
- **Kullback-Leibler Divergence:** Measures "how one probability distribution diverges from a second expected probability distribution." It's used for information gain and characterizing entropy.

### B. Optimization Algorithms

Optimization algorithms adjust the model's parameters to minimize the loss function.

- **Gradient Descent:** A "first-order iterative optimization algorithm for finding the minimum of a function." It involves taking "steps proportional to the negative of the gradient."
- **Stochastic Gradient Descent (SGD):** While standard Gradient Descent uses the "total gradient over all examples per update," SGD "updates after only 1 or few examples." "Ordinary gradient descent as a batch method is very slow, should never be used."
- **Mini-batch Stochastic Gradient Descent (SGD):** The "most commonly used now," it updates using a "mini-batch x1...m of size m" of examples, typically between 20 and 1000. This balances the estimation quality of the gradient with computational efficiency due to parallelism.
- **Momentum:** Improves SGD by "Add[ing] a fraction v of previous update to current one." This "will increase the size of the steps taken towards the minimum" when the gradient consistently points in the same direction.
- **Adagrad:** Provides "Adaptive learning rates for each parameter!" This means "learning rate is adapting differently for each parameter and rare parameters get larger updates than frequently occurring parameters."
- **Learning Rate:** A crucial hyperparameter that dictates the step size taken during gradient descent. If too high, weights "will change far too much each iteration, which will make them 'overcorrect' and the loss will actually increase/diverge." Common strategies include keeping it fixed, reducing it by 0.5 when validation error stops improving, or using adaptive methods like AdaGrad.

### C. Backpropagation

"Is a method used in artificial neural networks to calculate the error contribution of each neuron after a batch of data. It calculates the gradient of the loss function." It's essential for gradient-based optimization algorithms. The error is "calculated at the output and distributed back through the network layers," which is why it's also called "backward propagation of errors." This method is efficient because it "reuse[s] partial derivatives computed for higher layers in lower layers."

### D. Regularization

Techniques to prevent overfitting, where the model performs well on training data but poorly on unseen data.

- **L1 Norm (Manhattan Distance):** Also known as Least Absolute Deviations (LAD), it minimizes "the sum of the absolute differences (S) between the target value and the estimated values." It can lead to sparse models.
- **L2 Norm (Euclidean Distance):** Also known as Least Squares, it minimizes "the sum of the square of the differences (S) between the target value and the estimated values."
- **Early Stopping:** Provides "guidance as to how many iterations can be run before the learner begins to over-fit, and stop the algorithm then." Using "parameters that gave best validation error."
- **Dropout:** A "regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data." During training, "randomly set 50% of the inputs to each neuron to 0." At test time, "halve the model weights." This acts as a form of model averaging.
- **Sparse Regularizer on Columns:** Combines L2 norm on each column with an L1 norm over all columns.
- **Nuclear Norm Regularization:** Not detailed, but listed as a type.
- **Mean-Constrained Regularization:** "Constrains the functions learned for each task to be similar to the overall average of the functions across all tasks," useful when tasks are expected to share similarities.

## IV. Neural Network Architectures

### A. Feed-Forward Neural Networks

In these networks, "connections between the units do not form a cycle." Information moves "only in one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes."

- **Single-Layer Perceptron:** Inputs are directly connected to outputs via weights. With a logistic activation function, it becomes a logistic regression model.
- **Multi-Layer Perceptron:** Consists of "multiple layers of computational units, usually interconnected in a feed-forward way." Each neuron in one layer connects to neurons in the subsequent layer, often using a sigmoid activation function.

### B. Recurrent Neural Networks (RNNs)

"A class of artificial neural network where connections between units form a directed cycle." This allows them to "exhibit dynamic temporal behavior" and "use their internal memory to process arbitrary sequences of inputs," making them suitable for tasks like handwriting or speech recognition.

- **Long Short-Term Memory (LSTMs):** A specific type of RNN that allows data to flow both forwards and backwards, making it "well-suited to learn from experience to classify, process and predict time series given time lags of unknown size." LSTMs are relatively insensitive to gap length, offering an advantage over other sequence learning methods.

### C. Recursive Neural Networks (RNNs)

"A kind of deep neural network created by applying the same set of weights recursively over a structure, to produce a structured prediction over variable-size input structures." They have been successful in natural language processing for learning sequence and tree structures, particularly continuous representations of phrases and sentences based on word embeddings.

### D. Convolutional Neural Networks (CNNs)

Primarily used for tasks involving grid-like data, such as images. They have applications in "image and video recognition, recommender systems and natural language processing." Key components include:

- **Convolution:** Applying filters to input data to extract features.
- **Pooling/Subsampling:** Reducing the spatial dimensions of the feature maps, helping to make the representation more robust to small shifts and distortions.

### E. Auto-Encoders

"An artificial neural network used for unsupervised learning of efficient codings." Their aim is "to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction." They are increasingly used for learning generative models.

### F. Generative Adversarial Networks (GANs)

"A class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework." One network (the generator) creates new data, and the other (the discriminator) tries to distinguish between real and generated data.

## V. Practical Deep Learning Strategy & Tools

### A. Gradient Checks

An essential step for debugging neural network implementations.

1. **Implement your gradient.**
2. **Implement a finite difference computation** by adding and subtracting a small epsilon to parameters and estimating derivatives.
3. **Compare the two** to ensure they are "almost the same." If gradient checks fail, simplify the model until the bug is identified.

### B. Model Power and Regularization

- **Check if the model is powerful enough to overfit:** If not, "change model structure or make model 'larger'" by increasing units or layers.
- **If you can overfit:** "Regularize to prevent overfitting." Initial steps include "Reduce model size by lowering number of units and layers" or applying "Standard L1 or L2 regularization on weights."

### C. TensorFlow

TensorFlow is a prominent open-source machine learning framework developed by Google. It provides primitives for defining functions on tensors and automatically computing their derivatives, expressed as a graph.

- **Intuition:** The "Tensorflow Graph is build to contain all placeholders for X and y, all variables for W’s and b’s, all mathematical operations, the cost function, and the optimisation procedure." At runtime, data batches are "fed into that Graph, by placing the data batches in the placeholders and running the Graph." This graph-based approach enables parallelization across networks.
- **Main Components:Variables:** "Stateful nodes that output their current value, their state is retained across multiple executions of the graph." These are typically parameters like Weights (W) and Biases (b).
- **Placeholders:** "Nodes whose value is fed at execution time," typically for inputs (features X and labels y).
- **Mathematical Operations:** Functions like MatMul, Add, ReLU.
- **Graph Nodes:** Operations with inputs and outputs.
- **Edges:** Tensors that flow between nodes.
- **Session:** "A binding to a particular execution context: CPU, GPU." It's used to execute operations in the graph.
1. **Phases:Construction:** Assembling the computational graph (no numerical values yet).
2. **Execution:** Using a Session object to evaluate tensors and run operations.
- **tf.estimator:** TensorFlow's high-level API that simplifies building, training, and evaluating various ML models, including LinearClassifier, LinearRegressor, DNNClassifier, and DNNRegressor.
1. **Main Steps for tf.estimator:Define Feature Columns:** Encoding features for Estimators (e.g., real_valued_column for continuous, sparse_column_with_* for categorical).
2. **Define your Layers, or use a prebuilt model.**
3. **Write the input_fn function:** Holds features and labels.
4. **Train the model** using the fit function.
5. **Predict and Evaluate** using eval_input_fn.
- **TensorBoard:** "TensorFlow has some neat built-in visualization tools (TensorBoard)" for monitoring and visualizing training processes.

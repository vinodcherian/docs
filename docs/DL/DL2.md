# From Deep Learning MindMap

Here is the information from the sources, structured into a hierarchical tree with bullet points as requested:

1. **Concepts**
a. **Input Layer**
i. Comprised of multiple **Real-Valued inputs**.
ii. Each input must be **linearly independent** from each other.
b. **Hidden Layers**
i. Layers other than the input and output layers.
ii. A layer is the **highest-level building block in deep learning**.
iii. A layer usually receives weighted input, transforms it with mostly **non-linear functions**, and passes these values as output to the next layer.
c. **Batch Normalization**
i. Using **mini-batches of examples** is helpful.
ii. The gradient of the loss over a mini-batch is an estimate of the gradient over the training set, and its quality improves as the batch size increases.
iii. Computation over a batch can be much more efficient due to the **parallelism afforded by modern computing platforms**.
2. **Cost/Loss (Min) Objective (Max) Functions**
a. **Maximum Likelihood Estimation (MLE)**
i. Many cost functions are the result of applying Maximum Likelihood, such as the **Least Squares cost function** and **Cross-Entropy**.
ii. The likelihood of a parameter value (or vector of parameter values), θ, given outcomes x, is equal to the probability (density) assumed for those observed outcomes given those parameter values.
iii. The **natural logarithm of the likelihood function**, called the **log-likelihood**, is more convenient to work with.
1. It can be used in place of the likelihood in maximum likelihood estimation because the logarithm is a monotonically increasing function, thus achieving its maximum value at the same points as the function itself.
iv. It selects the set of values of the model parameters that **maximises the likelihood function** for a fixed set of data and underlying statistical model.
v. Intuitively, this maximises the "**agreement**" of the selected model with the observed data.
vi. For discrete random variables, it maximises the probability of the observed data under the resulting distribution.
vii. Maximum-likelihood estimation gives a **unified approach to estimation**, which is well-defined in the case of the normal distribution and many other problems.
b. **Cross-Entropy**
i. Can be used to define the **loss function in machine learning and optimization**.
ii. The true probability *pᵢ* represents the **true label**.
iii. The given distribution *qᵢ* represents the **predicted value** of the current model.
iv. An example is the **Cross-entropy error function used with logistic regression**.
c. **Quadratic**
i. The use of a quadratic loss function is common, for example when using **least squares techniques**.
ii. It is often more **mathematically tractable** than other loss functions due to the properties of variances.
iii. It is **symmetric**: an error above the target causes the same loss as the same magnitude of error below the target.
iv. If the target is *t*, then a quadratic loss function is defined as a specific formula.
d. **0-1 Loss**
i. In statistics and decision theory, this is a **frequently used loss function**.
e. **Hinge Loss**
i. A loss function used for **training classifiers**.
ii. For an intended output *t* = ±1 and a classifier score *y*, the hinge loss of the prediction *y* is defined as a specific formula.
f. **Exponential**
g. **Hellinger Distance**
i. Used to **quantify the similarity between two probability distributions**.
ii. It is a type of **f-divergence**.
iii. The square of the Hellinger distance between two probability measures P and Q (absolutely continuous with respect to a third probability measure λ) is defined as a specific quantity.
h. **Kullback-Leibler Divergence**
i. Is a measure of **how one probability distribution diverges from a second expected probability distribution**.
ii. Applications include:
1. Characterising the **relative (Shannon) entropy** in information systems.
2. Randomness in continuous time-series.
3. Information gain when comparing statistical models of inference.
iii. Can be applied to **Discrete** or **Continuous** distributions.
i. **Itakura–Saito distance**
i. Is a measure of the **difference between an original spectrum *P(ω)* and an approximation *P^(ω)*** of that spectrum.
ii. Although not a perceptual measure, it is intended to reflect **perceptual (dis)similarity**.
3. **Regularization**
a. **L1 norm (Manhattan Distance)**
i. Also known as **least absolute deviations (LAD)** or **least absolute errors (LAE)**.
ii. It is basically minimizing the sum of the absolute differences (S) between the target value and the estimated values.
b. **L2 norm (Euclidean Distance)**
i. Also known as **least squares**.
ii. It is basically minimizing the sum of the square of the differences (S) between the target value and the estimated values.
c. **Early Stopping**
i. Early stopping rules provide guidance as to how many iterations can be run before the learner begins to **over-fit**, and stop the algorithm then.
ii. Use parameters that gave the **best validation error**.
d. **Dropout**
i. A **regularization technique for reducing overfitting in neural networks**.
ii. Prevents complex co-adaptations on training data.
iii. It is a very efficient way of performing model averaging with neural networks.
iv. The term "dropout" refers to **dropping out units** (both hidden and visible) in a neural network.
v. During **training time**: at each instance of evaluation (in online SGD-training), randomly set 50% of the inputs to each neuron to 0.
vi. During **test time**: halve the model weights.
vii. This prevents **feature co-adaptation**, meaning a feature cannot only be useful in the presence of particular other features.
e. **Sparse regularizer on columns**
i. This regularizer defines an **L2 norm on each column** and an **L1 norm over all columns**.
ii. It can be solved by **proximal methods**.
f. **Nuclear norm regularization**
g. **Mean-constrained regularization**
i. This regularizer constrains the functions learned for each task to be **similar to the overall average** of the functions across all tasks.
ii. This is useful for expressing prior information that each task is expected to **share similarities** with each other task.
iii. An example is predicting blood iron levels measured at different times of the day, where each task represents a different person.
iv. This regularizer is similar to the mean-constrained regularizer, but instead enforces a different constraint.
4. **Weight Initialization**
a. **All Zero Initialization**
i. In the ideal situation, with proper data normalization, it is reasonable to assume that approximately half of the weights will be positive and half will be negative.
ii. Setting all initial weights to zero, expecting it to be the "best guess", turns out to be a **mistake**.
iii. If every neuron computes the same output, they will also compute the same gradients during back-propagation and undergo the exact same parameter updates, meaning there is **no source of asymmetry between neurons**.
b. **Initialization with Small Random Numbers**
i. Weights should be **very close to zero, but not identically zero**.
ii. Randomising neurons to small numbers (very close to zero) is treated as **symmetry breaking**.
iii. The idea is that neurons are initially random and unique, so they will compute distinct updates and integrate as diverse parts of the full network.
iv. Implementation for weights might simply involve drawing values from a **normal distribution with zero mean, and unit standard deviation**.
v. Using small numbers drawn from a **uniform distribution** is also possible, but has relatively little impact on final performance in practice.
c. **Calibrating the Variances**
i. A problem with small random number initialization is that the distribution of outputs from a randomly initialized neuron has a variance that **grows with the number of inputs**.
ii. This variance can be normalized to 1 by **scaling each neuron's weight vector by the square root of its fan-in** (its number of inputs).
iii. This ensures that all neurons initially have approximately the same output distribution.
iv. Empirically, this improves the **rate of convergence**.
v. The derivations for this can be found in pages 18 to 23 of the slides (not provided).
vi. Note that the derivations do not consider the influence of ReLU neurons.
5. **Optimization**
a. **Gradient Descent**
i. A **first-order iterative optimization algorithm** for finding the minimum of a function.
ii. To find a local minimum, one takes steps **proportional to the negative of the gradient** (or approximate gradient) of the function at the current point.
iii. If steps are proportional to the positive of the gradient, one approaches a local maximum; this is known as **gradient ascent**.
iv. Uses the **total gradient over all examples per update**.
v. Ordinary gradient descent as a batch method is **very slow** and should never be used.
b. **Stochastic Gradient Descent (SGD)**
i. With SGD, training proceeds in steps, considering a **mini-batch *x1...m* of size *m*** at each step.
ii. The mini-batch is used to approximate the gradient of the loss function with respect to the parameters.
iii. Updates after **only 1 or a few examples**.
iv. On large datasets, **SGD usually wins over all batch methods**.
c. **Mini-batch Stochastic Gradient Descent (SGD)**
i. Most commonly used now.
ii. Size of each mini-batch *B* typically ranges from **20 to 1000**.
iii. Helps **parallelising any model** by computing gradients for multiple elements of the batch in parallel.
d. **Momentum**
i. Idea: **Add a fraction *v* of previous update** to the current one.
ii. When the gradient keeps pointing in the same direction, this will **increase the size of the steps taken towards the minimum**.
iii. Reduce global learning rate when using a lot of momentum.
iv. Update Rule: *v* is initialised at 0.
v. Momentum often increased after some epochs (e.g., from 0.5 to 0.99).
e. **Adagrad**
i. Provides **adaptive learning rates for each parameter**.
ii. The learning rate adapts differently for each parameter.
iii. **Rare parameters get larger updates** than frequently occurring parameters, which is useful for word vectors.
iv. It is a method for **not hand-setting learning rates**.
f. **Learning Rate**
i. Neural networks are often trained by gradient descent on the weights.
ii. At each iteration, backpropagation calculates the derivative of the loss function with respect to each weight, which is then subtracted from that weight.
iii. If the weights change too much each iteration, they will "**overcorrect**" and the loss will increase/diverge.
iv. In practice, people usually **multiply each derivative by a small value called the “learning rate”** before subtracting it from its corresponding weight.
v. **Simplest recipe**: keep it fixed and use the same for all parameters.
vi. **Better results** by allowing learning rates to decrease.
1. Options include: Reducing by 0.5 when validation error stops improving.
2. Reduction by O(1/t) because of theoretical convergence guarantees, with hyper-parameters ε0 and τ and *t* as iteration numbers.
6. **Backpropagation**
a. A method used in artificial neural networks to **calculate the error contribution of each neuron** after a batch of data.
b. It calculates the **gradient of the loss function**.
c. It is commonly used in the **gradient descent optimization algorithm**.
d. Also called **backward propagation of errors**, because the error is calculated at the output and distributed back through the network layers.
e. It reuses partial derivatives computed for higher layers in lower layers, for **efficiency**.
7. **Activation Functions**
a. Defines the **output of that node given an input or set of inputs**.
b. Types include:
i. **ReLU**
ii. **Sigmoid / Logistic**
iii. **Binary**
iv. **Tanh**
v. **Softplus**
vi. **Softmax**
vii. **Maxout**
viii. **Leaky ReLU, PReLU, RReLU, ELU, SELU**, and others.
8. **Architectures Strategy**
a. **Check for implementation bugs with gradient checks**
i. Implement your gradient.
ii. Implement a **finite difference computation** by looping through the network parameters, adding and subtracting a small epsilon (10⁻⁴), and estimating derivatives.
iii. **Compare the two** and make sure they are almost the same.
iv. If your gradient fails and you don’t know why, **simplify your model until you have no bug**.
v. Create a **very tiny synthetic model and dataset**.
vi. Example progression from simplest to more complex model for debugging:
1. Only softmax on fixed input.
2. Backprop into word vectors and softmax.
3. Add single unit single hidden layer.
4. Add multi unit single layer.
5. Add second layer single unit, then multiple units, bias.
6. Add one softmax on top, then two softmax layers.
7. Add bias.
b. **Parameter Initialization**
i. Initialize **hidden layer biases to 0**.
ii. Initialize **output (or reconstruction) biases to optimal value** if weights were 0 (e.g., mean target or inverse sigmoid of mean target).
iii. Initialize **weights Uniform(−r, r)**, where *r* is inversely proportional to fan-in (previous layer size) and fan-out (next layer size).
c. **Optimization**
i. **Gradient Descent** (as described in section 5.a).
ii. **Stochastic Gradient Descent (SGD)** (as described in section 5.b).
1. Ordinary gradient descent as a batch method is very slow and **should never be used**; use 2nd order batch methods such as L-BFGS.
2. On large datasets, **SGD usually wins** over all batch methods.
3. On smaller datasets, L-BFGS or Conjugate Gradients win.
4. Large-batch L-BFGS extends the reach of L-BFGS.
iii. **Mini-batch Stochastic Gradient Descent (SGD)** (as described in section 5.c).
iv. **Momentum** (as described in section 5.d).
v. **Adagrad** (as described in section 5.e).
d. **Check if the model is powerful enough to overfit**
i. If not, **change model structure** or make the model "larger".
ii. If you can overfit, **regularise to prevent overfitting**.
1. **Simple first step**: Reduce model size by lowering the number of units and layers and other parameters.
2. Apply **Standard L1 or L2 regularization** on weights.
3. Use **Early Stopping**: parameters that gave the best validation error.
4. Apply **Sparsity constraints on hidden activations**, e.g., by adding to the cost.
5. Implement **Dropout** (as described in section 3.d).
9. **Neural Network Architectures / Types**
a. **RNNs (Recursive)**
i. A kind of deep neural network created by applying the same set of weights recursively over a structure.
ii. It produces a structured prediction over variable-size input structures, or a scalar prediction, by traversing a given structure in topological order.
iii. RNNs have been successful in learning sequence and tree structures in natural language processing, mainly phrase and sentence continuous representations based on word embedding.
b. **RNNs (Recurrent)**
i. A class of artificial neural network where **connections between units form a directed cycle**.
ii. This allows them to exhibit **dynamic temporal behavior**.
iii. Unlike feedforward neural networks, RNNs can use their **internal memory to process arbitrary sequences of inputs**.
iv. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.
v. **LSTMs** are a type of recurrent RNN.
c. **Convolutional Neural Networks (CNN)**
i. They have applications in **image and video recognition, recommender systems, and natural language processing**.
ii. Key components include:
1. **Pooling**
2. **Convolution**
3. **Subsampling**
d. **Auto-Encoders**
i. An artificial neural network used for **unsupervised learning of efficient codings**.
ii. The aim is to learn a representation (encoding) for a set of data, typically for **dimensionality reduction**.
iii. Recently, the autoencoder concept has become more widely used for **learning generative models of data**.
e. **GANs (Generative Adversarial Networks)**
i. A class of artificial intelligence algorithms used in **unsupervised machine learning**.
ii. Implemented by a system of **two neural networks contesting with each other in a zero-sum game framework**.
f. **LSTMs (Long Short-Term Memory)**
i. It is a type of **recurrent RNN**.
ii. Allows data to flow both **forwards and backwards** within the network.
iii. An LSTM is well-suited to learn from experience to classify, process, and predict time series given time lags of unknown size and bound between important events.
iv. Relative insensitivity to gap length gives an advantage to LSTM over alternative RNNs, hidden Markov models, and other sequence learning methods in numerous applications.
g. **Feed Forward**
i. An artificial neural network wherein **connections between the units do not form a cycle**.
ii. Information moves in only one direction, **forward**, from the input nodes, through the hidden nodes (if any) and to the output nodes.
iii. There are no cycles or loops in the network.
iv. Kinds:
1. **Single-Layer Perceptron**
a. The inputs are fed directly to the outputs via a series of weights.
b. By adding a **Logistic activation function** to the outputs, the model is identical to a classical Logistic Regression model.
2. **Multi-Layer Perceptron**
a. This class of networks consists of multiple layers of computational units, usually interconnected in a **feed-forward way**.
b. Each neuron in one layer has directed connections to the neurons of the subsequent layer.
c. In many applications, the units of these networks apply a **sigmoid function as an activation function**.
10. **TensorFlow**
a. **Packages**
i. **`tf`** (Main).
ii. **`tf.estimator`**: TensorFlow’s high-level machine learning API.
1. Makes it easy to configure, train, and evaluate a variety of machine learning models.
2. Includes pre-canned estimators like:
a. `tf.estimator.LinearClassifier`: Constructs a linear classification model.
b. `tf.estimator.LinearRegressor`: Constructs a linear regression model.
c. `tf.estimator.DNNClassifier`: Constructs a neural network classification model.
d. `tf.estimator.DNNRegressor`: Constructs a neural network regression model.
e. `tf.estimator.DNNLinearCombinedClassifier`: Constructs a neural network and linear combined classification model.
f. `tf.estimator.DNNRegressor`: Constructs a neural network and linear combined regression model.
b. **Main Steps (General)**
i. Create the Model.
ii. Define Target.
iii. Define Loss function and Optimizer.
iv. Define the Session and Initialise Variables.
v. Train the Model.
vi. Test Trained Model.
c. **Main Steps (using `tf.estimator`)**
i. **Define Feature Columns**
1. These are the primary way of encoding features for pre-canned `tf.learn` Estimators.
2. The type of feature column chosen depends on the feature type and the model type.
3. **Continuous Features** can be represented by `real_valued_column`.
4. **Categorical Features** can be represented by any `sparse_column_with_*` column (e.g., `sparse_column_with_keys`, `sparse_column_with_vocabulary_file`, `sparse_column_with_hash_bucket`, `sparse_column_with_integerized_feature`).
ii. **Define your Layers, or use a prebuilt model** (e.g., a pre-built Logistic Regression Classifier).
iii. **Write the `input_fn` function**
1. This function holds the actual data (features and labels).
2. `Features` is a Python dictionary.
iv. **Train the model** using the `fit` function, on the `input_fn`.
1. The feature columns are fed to the model as arguments.
v. **Predict and Evaluate** using the `eval_input_fn` defined previously.
d. **Comparison to Numpy**
i. TensorFlow does **lazy evaluation**.
ii. You need to build the graph, and then run it in a session.
e. **Main Components**
i. **Variables**
1. **Stateful nodes** that output their current value, with their state retained across multiple executions of the graph.
2. Mostly **parameters we’re interested in tuning**, such as Weights (W) and Biases (b).
3. They are in-memory buffers containing tensors.
4. Declared variables must be initialised before they have values.
5. When training a model, variables are used to hold and update parameters.
6. Sharing variables can be done by:
a. Explicitly passing `tf.Variable` objects around.
b. Implicitly wrapping `tf.Variable` objects within `tf.variable_scope` objects.
ii. **Scopes**
1. `tf.variable_scope()`: Provides simple **name spacing** to avoid cases when querying.
2. `tf.get_variable()`: Creates/Accesses variables from a variable scope.
iii. **Placeholders**
1. Nodes whose **value is fed at execution time**.
2. Used for Inputs, Features (X) and Labels (y).
iv. **Mathematical Operations**: Examples include `MatMul`, `Add`, `ReLU`, etc..
v. **Graph Nodes**: They are Operations, containing any number of inputs and outputs.
vi. **Edges**: The tensors that flow between the nodes.
vii. **Session**
1. It is a binding to a **particular execution context**: CPU, GPU.
2. A Session object encapsulates the environment in which Tensor objects are evaluated.
3. Uses a session to execute operations (`ops`) in the graph.
4. Running a Session involves:
a. **Inputs**.
b. **Fetches**: A list of graph nodes, returning the output of these nodes.
c. **Feeds**: A dictionary mapping from graph nodes to concrete values, specifying the value of each graph node given in the dictionary.
f. **Phases**
i. **Construction**: Assembles a **computational graph**.
1. The computation graph has no numerical value until evaluated.
2. All computations add nodes to the global default graph.
ii. **Execution**: A Session object encapsulates the environment in which Tensor objects are evaluated.
1. Uses a session to execute `ops` in the graph.
g. **TensorBoard**: TensorFlow has some neat **built-in visualisation tools**.
h. **Intuition**
i. Google provides primitives for defining functions on tensors and automatically computing their derivatives, **expressed as a graph**.
ii. The TensorFlow Graph is built to contain all placeholders for X and y, all variables for W’s and b’s, all mathematical operations, the cost function, and the optimisation procedure.
iii. At runtime, the values for the data are fed into that Graph, by placing the data batches in the placeholders and running the Graph.
iv. Each node in the Graph can then be connected to each other node over the network, allowing **TensorFlow models to be parallelised**.

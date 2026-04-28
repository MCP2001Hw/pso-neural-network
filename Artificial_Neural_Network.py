import numpy

# Activation function: sigmoid - squashes values to (0, 1)
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

# Activation function: ReLU - returns max(0, x)
def relu(x):
    return numpy.maximum(0, x)

# Activation function: tanh - squashes values to (-1, 1)
def tanh(x):
    return numpy.tanh(x)

# Activation function: linear - no transformation, used for regression output
def linear(x):
    return x

# Dictionary mapping activation names to their functions
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "reLU": relu,
    "linear": linear
}

class ANN:
    def __init__(self, layer_sizes, activations):
        # layer_sizes: list of integers e.g. [n_inputs, n_hidden1, n_hidden2, n_outputs]
        # activations: list of activation function names for each layer except input
        
        # Ensure number of activation functions matches number of layers
        assert len(layer_sizes) - 1 == len(activations), "Activation count must match layer count"
        
        self.layer_sizes = layer_sizes
        self.activations = activations
        
        # Calculate shape and count of weights/biases for each layer
        self.param_sizes = []
        total_params = 0
        
        for i in range(len(layer_sizes) - 1):
            # Weight shape: (output_size, input_size)
            w_shape = (layer_sizes[i + 1], layer_sizes[i])
            # Bias shape: (output_size, 1)
            b_shape = (layer_sizes[i + 1], 1)
            
            # Total parameters for this layer = weights + biases
            size = numpy.prod(w_shape) + numpy.prod(b_shape)
            self.param_sizes.append((w_shape, b_shape))
            total_params += size
        
        # Store total number of parameters (used by PSO for particle dimension)
        self.total_params = total_params

    def get_weights_flat(self, weights, biases):
        # Flatten all weight and bias matrices into a single 1D vector
        flat = numpy.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])
        return flat

    def set_weights_from_vector(self, vector):
        # Convert 1D vector (from PSO particle) to list of weight/bias matrices
        # Used to reshape PSO position vector into proper network structure
        
        weights = []
        biases = []
        idx = 0
        
        # Iterate through each layer and extract its weights and biases
        for w_shape, b_shape in self.param_sizes:
            # Extract weight vector and reshape to proper matrix form
            w_size = numpy.prod(w_shape)
            w = vector[idx:idx + w_size].reshape(w_shape)
            idx += w_size
            
            # Extract bias vector and reshape to proper matrix form
            b_size = numpy.prod(b_shape)
            b = vector[idx:idx + b_size].reshape(b_shape)
            idx += b_size
            
            weights.append(w)
            biases.append(b)
        
        return weights, biases

    def forward(self, X, weights, biases):
        # Forward pass through the network for batch input X
        # Input X shape: (n_features, n_samples)
        # Output shape: (n_output_features, n_samples)
        
        # Start with input data
        a = X
        
        # Pass through each layer
        for i in range(len(weights)):
            # Linear transformation: z = W * a + b
            z = numpy.dot(weights[i], a) + biases[i]
            # Apply activation function
            a = ACTIVATIONS[self.activations[i]](z)
        
        # Return final network output
        return a

    def evaluate_fitness(self, vector, X, y, metric="mae"):
        # Compute fitness (error) for a given parameter vector
        # Lower error = better fitness (PSO minimizes this)
        # 
        # Parameters:
        #   vector: 1D array of network parameters (PSO position)
        #   X: input data
        #   y: target output data
        #   metric: "mae" for Mean Absolute Error or "mse" for Mean Squared Error
        
        # Convert PSO position vector into network weight/bias matrices
        weights, biases = self.set_weights_from_vector(vector)
        
        # Forward pass to get network predictions
        y_pred = self.forward(X, weights, biases)

        # Calculate error based on chosen metric
        if metric == "mse":
            # Mean Squared Error: average of squared differences
            error = numpy.mean((y_pred - y) ** 2)
        else:
            # Mean Absolute Error: average of absolute differences (default)
            error = numpy.mean(numpy.abs(y_pred - y))
        
        return error

if __name__ == "__main__":
    # Test the ANN with a simple regression task
    
    # Create training data: y = sin(x)
    X = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, 50).reshape(1, -1)
    y = numpy.sin(X)

    # Define network architecture:
    # 1 input → 4 hidden units → 1 output
    layer_sizes = [1, 4, 1]
    activations = ["tanh", "linear"]
    ann = ANN(layer_sizes, activations)

    # Create random parameter vector (like a particle position from PSO)
    test_vector = numpy.random.randn(ann.total_params)

    # Evaluate network fitness on test data
    fitness = ann.evaluate_fitness(test_vector, X, y)
    
    # Display results
    print("ANN parameter dimension:", ann.total_params)
    print("Initial random fitness (MAE):", fitness)
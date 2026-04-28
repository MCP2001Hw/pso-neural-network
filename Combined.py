# ---------------------------------------------------------------------
# PSO pseudo-code reference (from coursework)
# ---------------------------------------------------------------------
# Line 1: swarmsize ← desired swarm size
# Line 2: α ← proportion of velocity to be retained
# Line 3: β ← proportion of personal best to be retained
# Line 4: γ ← proportion of the informants' best to be retained
# Line 5: δ ← proportion of global best to be retained
# Line 6: ε ← jump size of a particle
# Line 7: P ← {}
# Line 8: for swarmsize times do
# Line 9:     P ← P ∪ {new random particle x with a random initial velocity v}
# Line 10: Best ← □
# Line 11: repeat
# Line 12:     for each particle p in P do
# Line 13:         AssessFitness(p)
# Line 14:         if Best = □ or Fitness(p) > Fitness(Best) then
# Line 15:         Best ← p
# Line 16:     for each particle p in P do
# Line 17:         x* ← previous fittest location of p
# Line 18:         x+ ← previous fittest location of informants of p
# Line 19:         x! ← previous fittest location of any particle
# Line 20:         for each dimension i do
# Line 21:             b ← random number from 0.0 to β inclusive
# Line 22:             c ← random number from 0.0 to γ inclusive
# Line 23:             d ← random number from 0.0 to δ inclusive
# Line 24:             vᵢ ← α vᵢ + b(xᵢ* − xᵢ) + c(xᵢ+ − xᵢ) + d(xᵢ! − xᵢ)
# Line 25:     for each particle p in P do
# Line 26:         x ← x + ε v
# Line 27: until Best is ideal solution or out of time
# ---------------------------------------------------------------------

import numpy
import pandas


# =====================================================================
# 1. Load dataset
# =====================================================================

data = pandas.read_csv("concrete_data.csv")

target_column = "concrete_compressive_strength"
input_columns = [col for col in data.columns if col != target_column]

# Inputs: transpose to shape (n_features, n_samples)
x1 = data[input_columns].to_numpy().T
# Target: shape (1, n_samples)
y1 = data[target_column].to_numpy().reshape(1, -1)


# =====================================================================
# 2. Activation functions and mapping
# =====================================================================

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
    "linear": linear,
}


# =====================================================================
# 3. ANN class
# =====================================================================

class ANN:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: list of integers e.g. [n_inputs, n_hidden1, n_hidden2, n_outputs]
        activations: list of activation function names for each layer except input
        """
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
        """Flatten all weight and bias matrices into a single 1D vector."""
        flat = numpy.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])
        return flat

    def set_weights_from_vector(self, vector):
        """
        Convert 1D vector (from PSO particle) to list of weight/bias matrices.
        Used to reshape PSO position vector into proper network structure.
        """
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
        """
        Forward pass through the network for batch input X.
        Input X shape: (n_features, n_samples)
        Output shape: (n_output_features, n_samples)
        """
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
        """
        Compute fitness (error) for a given parameter vector.
        Lower error = better fitness (PSO minimizes this).

        Parameters:
          vector: 1D array of network parameters (PSO position)
          X: input data
          y: target output data
          metric: "mae" for Mean Absolute Error or "mse" for Mean Squared Error
        """
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


# ---- helper: forward pass directly from parameter vector (ADDED) ----
def predict_with_params(ann, param_vector, X):
    """Forward pass for given parameter vector and input X."""
    weights, biases = ann.set_weights_from_vector(param_vector)
    return ann.forward(X, weights, biases)


# =====================================================================
# 4. PSO: particle class and helper functions
# =====================================================================

random_number_generator = numpy.random.default_rng(69)


class particle_class:
    def __init__(self, position, velocity):
        # Line 9: Initialize particle with position and velocity
        self.position = position
        self.velocity = velocity
        self.fitness = float("inf")
        self.personal_best = position.copy()
        self.personal_best_fitness = float("inf")
        self.informant_list = []
        self.informant_best_position = None


# Line 13: AssessFitness(p)
def assess_fitness(particle):
    error = ann.evaluate_fitness(particle.position, X, y, metric="mae")
    particle.fitness = error

    if error < particle.personal_best_fitness:
        particle.personal_best_fitness = error
        particle.personal_best = particle.position.copy()


def initialise_particles(swarm_size, dimension, particle_list):
    particle_list.clear()
    # Line 8: for swarmsize times do
    for _ in range(swarm_size):
        # Line 9: P ← P ∪ {new random particle x with a random initial velocity v}
        position = random_number_generator.uniform(low_bound, high_bound, size=dimension)
        velocity = random_number_generator.uniform(-1.0, 1.0, size=dimension)
        particle = particle_class(position=position, velocity=velocity)
        particle_list.append(particle)


def initialise_informants(particle_list):
    # Initialize each particle's informant list (subset of swarm)
    for i, particle in enumerate(particle_list):
        other_indices = [j for j in range(len(particle_list)) if j != i]
        if len(other_indices) > 0 and informant_count > 0:
            particle.informant_list = random_number_generator.choice(
                other_indices,
                size=min(informant_count, len(other_indices)),
                replace=False,
            ).tolist()
        else:
            particle.informant_list = []


def update_fitness_and_global_best(particle_list, current_global_best_position, current_global_best_fitness):
    # Line 12: for each particle p in P do
    for p in particle_list:
        # Line 13: AssessFitness(p)
        assess_fitness(p)

    for p in particle_list:
        # Line 14: if Best = □ or Fitness(p) > Fitness(Best) then
        if current_global_best_position is None or p.fitness < current_global_best_fitness:
            # Line 15: Best ← p
            current_global_best_fitness = p.fitness
            current_global_best_position = p.position.copy()

    return current_global_best_position, current_global_best_fitness


def apply_boundaries(position, velocity, low_bound, high_bound):
    # Handle boundary violations with velocity reversal
    exceeded_lower_bound = position < low_bound
    exceeded_upper_bound = position > high_bound

    position = numpy.clip(position, low_bound, high_bound)
    velocity[exceeded_lower_bound | exceeded_upper_bound] *= -1.0
    return position, velocity


# Line 16-24: Update velocity for each particle based on informant bests
def update_velocity(particle_list, global_best_position):
    # Line 16: for each particle p in P do
    for i, particle in enumerate(particle_list):
        # Line 17: x* ← previous fittest location of p
        x_star = particle.personal_best

        # Line 18: x+ ← previous fittest location of informants of p
        informant_candidates = [particle_list[j] for j in particle.informant_list] + [particle]
        best_informant = min(informant_candidates, key=lambda p: p.personal_best_fitness)
        x_plus = best_informant.personal_best

        # Line 19: x! ← previous fittest location of any particle
        x_exclaim = global_best_position if global_best_position is not None else particle.position

        # Line 21: b ← random number from 0.0 to β inclusive
        b_vector = random_number_generator.uniform(0.0, beta, size=dimension)
        # Line 22: c ← random number from 0.0 to γ inclusive
        c_vector = random_number_generator.uniform(0.0, gamma, size=dimension)
        # Line 23: d ← random number from 0.0 to δ inclusive
        d_vector = random_number_generator.uniform(0.0, delta, size=dimension)

        # Line 24: vᵢ ← α vᵢ + b(xᵢ* − xᵢ) + c(xᵢ+ − xᵢ) + d(xᵢ! − xᵢ)
        inertia_component = alpha * particle.velocity
        cognitive_component = b_vector * (x_star - particle.position)
        social_component = c_vector * (x_plus - particle.position)
        global_component = d_vector * (x_exclaim - particle.position)

        particle.velocity = inertia_component + cognitive_component + social_component + global_component


def update_positions(particle_list, low_bound, high_bound):
    # Line 25: for each particle p in P do
    for p in particle_list:
        # Line 26: x ← x + ε v
        p.position = p.position + epsilon * p.velocity
        p.position, p.velocity = apply_boundaries(p.position, p.velocity, low_bound, high_bound)


# =====================================================================
# 5. Train/test split, normalisation and ANN setup
# =====================================================================

# We already have:
# x1: (n_features, n_samples)
# y1: (1, n_samples)

n_samples = x1.shape[1]

# Use a fixed RNG for reproducible splitting
split_rng = numpy.random.default_rng()
indices = split_rng.permutation(n_samples)

split_index = int(0.7 * n_samples)   # 70% / 30% split

train_idx = indices[:split_index]
test_idx = indices[split_index:]

# Raw train/test splits (still in original units)
X_train_raw = x1[:, train_idx]   # (n_features, n_train)
y_train_raw = y1[:, train_idx]   # (1, n_train)

X_test_raw = x1[:, test_idx]     # (n_features, n_test)
y_test_raw = y1[:, test_idx]     # (1, n_test)

# ---- Normalise based on the training set only ----
X_mean = X_train_raw.mean(axis=1, keepdims=True)
X_std = X_train_raw.std(axis=1, keepdims=True) + 1e-8
X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

y_mean = y_train_raw.mean()
y_std = y_train_raw.std() + 1e-8
y_train = (y_train_raw - y_mean) / y_std
y_test = (y_test_raw - y_mean) / y_std

# PSO will use these globals for training
X = X_train
y = y_train

# Define network architecture: 8 input → 8 hidden → 1 output
n_features = x1.shape[0]             # 8
layer_sizes = [n_features, 8, 1]
activations = ["tanh", "linear"]

ann = ANN(layer_sizes, activations)

# Optional: quick sanity check with a random parameter vector on training set
test_vector = numpy.random.randn(ann.total_params)
fitness = ann.evaluate_fitness(test_vector, X, y)
print("="*40)
print("ANN parameter dimension:", ann.total_params)
print("Initial random training fitness (MAE):", f"{fitness:.5f}")
print("="*40)


# =====================================================================
# 6. PSO hyperparameters and main loop
# =====================================================================

# Line 1: swarmsize ← desired swarm size
swarm_size = 40

# Line 2: α ← proportion of velocity to be retained
alpha = 0.7
# Line 3: β ← proportion of personal best to be retained
beta = 1.4
# Line 4: γ ← proportion of the informants' best to be retained
gamma = 1.4
# Line 5: δ ← proportion of global best to be retained
delta = 0.5
# Line 6: ε ← jump size of a particle
epsilon = 0.5

# Other parameters
informant_count = swarm_size - 1
low_bound = -1.0
high_bound = 1.0

iteration_limit = 100000  # Changeable

# Line 7: P ← {}
particle_list = []

dimension = ann.total_params

# Line 8-9
initialise_particles(swarm_size, dimension, particle_list)
initialise_informants(particle_list)

# Line 10: Best ← □
global_best_position = None
global_best_fitness = float("inf")
last_global_best_fitness = float("inf")
iterations_run = 0

# Line 11: repeat

for iteration in range(iteration_limit):

    display_GBV =repr(global_best_fitness)
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1} / {iteration_limit}, Best fitness: {global_best_fitness}")
    # Line 12-15
    global_best_position, global_best_fitness = update_fitness_and_global_best(
        particle_list,
        global_best_position,
        global_best_fitness,
    )

    # Line 16–24
    update_velocity(particle_list, global_best_position)

    # Line 25–26
    update_positions(particle_list, low_bound, high_bound)

    # Line 27: until termination condition
    if iteration >= iteration_limit - 1:    
        iterations_run = iteration + 1
        break

# Return Best
found_global_best_position = global_best_position
found_global_best_fitness = global_best_fitness

print("Iterations run:", iterations_run)
print("Global best fitness:", f"{found_global_best_fitness:.5f}")
print("="*40)


# =====================================================================
# 7. Evaluate best solution on train and test sets
# =====================================================================

train_mae_norm = ann.evaluate_fitness(found_global_best_position, X_train, y_train, metric="mae")
test_mae_norm = ann.evaluate_fitness(found_global_best_position, X_test, y_test, metric="mae")

def mae_in_original_units(mae_norm, std):
    return mae_norm * std

train_mae_real = mae_in_original_units(train_mae_norm, y_std)
test_mae_real = mae_in_original_units(test_mae_norm, y_std)

print("Final training MAE (normalised):", f"{train_mae_norm:.5f}")
print("Final training MAE:", f"{train_mae_real:.5f}")
print("Final test MAE (normalised):", f"{test_mae_norm:.5f}")
print("Final test MAE:", f"{test_mae_real:.5f}")

# =====================================================================
# 8. Min / max / avg |pred - true| on test datasets 
# =====================================================================

# Predictions in normalised space
y_test_pred_norm = predict_with_params(ann, found_global_best_position, X_test)

# Denormalise predictions back to original units
y_test_pred = y_test_pred_norm * y_std + y_mean

y_test_true = y_test_raw

test_abs_errors = numpy.abs(y_test_pred - y_test_true)

test_min_diff = test_abs_errors.min()
test_max_diff = test_abs_errors.max()
test_mean_diff = test_abs_errors.mean()

print("="*40)
print("Test set error stat")
print("="*40)
print("Min: ", f"{test_min_diff:.5f}")
print("Max: ", f"{test_max_diff:.5f}")
print("Avg: ", f"{test_mean_diff:.5f}")
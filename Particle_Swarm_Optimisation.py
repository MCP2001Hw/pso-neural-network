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
# Line 15:             Best ← p
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

import numpy

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
        
random_number_generator = numpy.random.default_rng(69)

# Line 1: swarmsize ← desired swarm size
swarm_size = 50

# Line 2: α ← proportion of velocity to be retained
alpha = 0.5
# Line 3: β ← proportion of personal best to be retained
beta = 0.5
# Line 4: γ ← proportion of the informants' best to be retained
gamma = 0.5
# Line 5: δ ← proportion of global best to be retained
delta = 0.5
# Line 6: ε ← jump size of a particle
epsilon = 1.0

# Other parameters
informant_count = swarm_size - 1
dimension = 2 # Changeable
low_bound = 0
high_bound = 1.0

iteration_limit = 10000 # Changeable
convergence_threshold = 0 # Changeable

# Line 7: P ← {}
particle_list = []

# Line 13: AssessFitness(p)
def assess_fitness(particle):
    particle.fitness = 0 # ANN 

    if particle.fitness < particle.personal_best_fitness:
        particle.personal_best_fitness = particle.fitness
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
                other_indices, size=min(informant_count, len(other_indices)), replace=False
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

# Line 8: for swarmsize times do
# Line 9:     P ← P ∪ {new random particle x with a random initial velocity v}
initialise_particles(swarm_size, dimension, particle_list)
initialise_informants(particle_list)

# Line 10: Best ← □
global_best_position = None
global_best_fitness = float("inf")

# Line 11: repeat
for iteration in range(iteration_limit):
    # Line 12: for each particle p in P do
    # Line 13:     AssessFitness(p)
    # Line 14:     if Best = □ or Fitness(p) > Fitness(Best) then
    # Line 15:         Best ← p
    global_best_position, global_best_fitness = update_fitness_and_global_best(
        particle_list, global_best_position, global_best_fitness
    )
    
    # Line 16:     for each particle p in P do
    # Line 17:         x* ← previous fittest location of p
    # Line 18:         x+ ← previous fittest location of informants of p
    # Line 19:         x! ← previous fittest location of any particle
    # Line 20:         for each dimension i do
    # Line 21:             b ← random number from 0.0 to β inclusive
    # Line 22:             c ← random number from 0.0 to γ inclusive
    # Line 23:             d ← random number from 0.0 to δ inclusive
    # Line 24:             vᵢ ← α vᵢ + b(xᵢ* − xᵢ) + c(xᵢ+ − xᵢ) + d(xᵢ! − xᵢ)
    update_velocity(particle_list, global_best_position)
    
    # Line 25: for each particle p in P do
    # Line 26:     x ← x + ε v
    update_positions(particle_list, low_bound, high_bound)
    
    # Line 27: until termination condition
    if global_best_fitness <= convergence_threshold:
        iterations_run = iteration + 1
        break
    else:
        iterations_run = iteration_limit

# Return Best
found_global_best_position = global_best_position
found_global_best_fitness = global_best_fitness

print("Iterations run:", iterations_run)
print("Global best fitness:", found_global_best_fitness)
print("Global best position:", found_global_best_position[:5])
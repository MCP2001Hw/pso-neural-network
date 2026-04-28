# PSO-Optimised Neural Network

Group coursework for F20BC Biologically Inspired Computation — Heriot-Watt University.
2-person team. **My contribution: the PSO algorithm (`Particle_Swarm_Optimisation.py`) and integration of both components (`Combined.py`). Partner implemented the ANN (`Artificial_Neural_Network.py`).**

Trains a feedforward neural network using Particle Swarm Optimisation as an alternative to gradient descent, applied to the UCI Concrete Compressive Strength regression dataset.

---

## How It Works

Rather than using backpropagation, the network weights are treated as a search problem. A swarm of particles explores the weight space, each tracking its personal best position and sharing information with neighbours. Over iterations the swarm converges on a low-error solution.

The velocity update combines four components:
- **Inertia** — momentum from the previous velocity
- **Personal best** — pull toward the particle's own best position
- **Informant best** — pull toward the best position among local neighbours
- **Global best** — pull toward the best position found by any particle

---

## Files

| File | Author | Description |
|---|---|---|
| `Particle_Swarm_Optimisation.py` | Me | PSO algorithm — swarm initialisation, velocity update, iteration loop |
| `Artificial_Neural_Network.py` | Partner | Feedforward ANN — forward pass, fitness evaluation (MAE/MSE) |
| `Combined.py` | Me | Integration — loads dataset, normalises inputs, trains ANN via PSO, evaluates on test split |
| `concrete_data.csv` | — | UCI Concrete Compressive Strength dataset |

---

## Running

```bash
pip install numpy pandas
python Combined.py
```

Trains the network and reports MAE on both training and test sets in original units.

---

## Results

PSO converges on a solution competitive with backpropagation on this dataset, at the cost of more iterations. The experiment evaluates convergence speed and final test MAE as the comparison metrics.

---

## Dependencies

- `numpy`
- `pandas`

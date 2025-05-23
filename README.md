# Bipartite Dynamics Simulation

This project explores the time evolution of coupled dynamical systems under three different interaction modes:

- **Quantum–Quantum (QQ)**
- **Classical–Quantum (CQ)**
- **Classical–Classical (CC)**

It supports parametric sweeps over oscillator/projectile masses (`mx`, `my`), simulates their time evolution, and visualizes observables like position, momentum, energy, interaction, and entropy.

---

## 📁 Project Structure

├── RunSimulation_mx_my.py                # Main runner for simulations

├── ClassicalQuantumEhrenfest_mx_my.py    # Coupled CQ dynamics

├── QuantumSimulationModules_mx_my.py     # Quantum dynamics modules

├── ClassicalSimulationModules_mx_my.py   # Classical dynamics modules

├── simulation_analysis_tn4096.py         # Plotting and entropy/energy summary

├── environment.yml                       # Conda environment specification

└── results_tn4096/                       # Output directory for all results
---

## 🔧 Environment Setup

To ensure reproducibility, use the provided `environment.yml` file to set up your environment.

### ▶️ Step 1: Create the Environment

```bash
conda env create -f environment.yml
This will create an environment named py310_env with all necessary dependencies.

### ▶️ Step 2: Activate the Environment
```bash
conda activate py310_env

### ▶️ Optional: Update Environment Later
```bash
conda env update -f environment.yml --prune

🧪 Dependencies

Installed via environment.yml:
	•	python=3.10
	•	numpy, scipy, matplotlib
	•	psutil, memory_profiler

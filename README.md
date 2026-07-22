# Time-Domain Sound-Soft Scattering by Many Small Particles: Asymptotic Modeling Framework

This Python framework provides a numerical realization of asymptotic models for solving **time-domain sound-soft scattering problems** involving multiple small particles of arbitrary shapes. Developed as part of a PhD thesis, the project focuses on the efficiency, accuracy, and robustness of time-domain simulations.

<div align="center">
  <img src="notebooks/pictures/comparison_animation_sgfl_vs_born.gif" width="600" alt="Scattering Animation">
</div>

### Models Included

The framework provides numerical implementations and comparisons of the following models:

*   **Galerkin Foldy-Lax (GFL):** An asymptotic model developed and analyzed during my PhD research.
*   **Simplified GFL:** A streamlined asymptotic model that maintains the accuracy of the GFL model while providing significantly higher numerical robustness.
*   **Born Approximation:** A standard first-order asymptotic model.
*   **BEM Solver:** A Boundary Element Method implementation used to generate a reference solution.

### Key Results

Numerical experiments demonstrate the following absolute error convergence rates as ($\varepsilon \to 0$):

*   **GFL & Simplified GFL:** Both models achieve cubic convergence, $O(\varepsilon^3)$.
*   **Born Approximation:** Achieves quadratic convergence, $O(\varepsilon^2)$.

While the GFL models offer higher-order accuracy, the **Simplified GFL** and **Born approximation** are particularly recommended for their robustness in practical simulations.

### Development Roadmap (In Preparation)

The framework is being extended to include:
*   **High-order asymptotic models** for even greater precision.
*   **Coupling solvers** developed for complex configurations involving one large obstacle and multiple small particles.

## Project Structure
- `src/` — Core mathematical models and geometry utilities.
- `external/bempp/` — Modified source code of the Bempp library (see https://bempp.com/).
- `notebooks/` — Interactive demos and result visualization.
- `environment.yml` — Conda environment specification.

## Installation and Quick Start

### 1. Dependencies
This project requires the Conda package manager. If you do not have it, we recommend installing [Miniconda](https://docs.anaconda.com/miniconda/) for your operating system.

### 2. Setup Environment
Navigate to the project's root directory and create the virtual environment. This will automatically install all required dependencies:

```bash
conda env create -f environment.yml
```

### 3. Usage
Activate the environment:

```bash
conda activate phd_asymptotic
```

To understand the problem context and visualize the numerical results, we recommend starting with the interactive examples in the `notebooks/` directory, specifically: **`demo.ipynb`**.

## References

This framework is the numerical implementation of the models and methods developed in:

*   **Adrian Savchuk**. *[Asymptotic Modeling of Time-Domain Scattering by Small Particles of Arbitrary Shapes](https://hal.science/tel-05639652v1/file/Savchuk_manuscript_v_2.pdf)*. PhD thesis, ENSTA Paris, 2026.

## Authors

**Adrian Savchuk**  
Personal email: adrian.savchuk.v@gmail.com

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
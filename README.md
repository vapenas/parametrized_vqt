# Parametrized Variational Quantum Tomography

Code for [arXiv:2604.27135](https://arxiv.org/abs/2604.27135)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main tomography script:

```bash
python3 main_loop_sic-povm_obs.py
```

## Configuration

Edit `config.ini` to configure the run. Key sections:

- **[user_run]**: Set the output directory (`dir_user_run`), number of qubits (`nqubits`), observable basis (`nombre_base_observables`), and tomography method flags (`vqt`, `vqt_inf`, `vqt_hib`, `maxent`, `maxent_exp`).
- **[parametros_opt]**: Optimization hyperparameters (`alpha`, `beta`, `max_iter_*`).
- **[tomografia_barrido]**: Sweep over number of observables (`min_cant_obs` to `max_cant_obs`).
- **[tomografia_test]**: Fixed observable count test mode.
- **[mediciones]**: Measurement generation. Set `generar = true` to auto-generate, or provide a custom path via `ruta_pickle_lista_mediciones_por_estados_custom`.

## Project Structure

```
├── config.ini                 # Run configuration
├── main_loop_sic-povm_obs.py  # Main entry point
├── state_tomography.py        # Tomography algorithms
├── generate_mediciones.py     # Measurement generation
├── load_config.py             # Configuration loader
├── Estados_random/            # Random state generators
├── runs/                      # Output directory (reconstructed states, results)
├── Mediciones/                # Generated measurements
├── tomografia/                # Tomography utilities
└── utils/                     # Helper functions
```

## Output

Results are saved in `runs/<dir_user_run>/` with nested folders for:
- Estimated density matrices
- Fidelity data
- Plots

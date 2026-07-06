# StringMicrolensing

## Scientific software for cosmic superstring microlensing analysis built around a modular ETL pipeline.

StringMicrolensing contains the software developed during my Ph.D. research. It is a collection of Python tools for 
processing astronomical survey data, generating simulated microlensing events, estimating survey sensitivity, and 
modeling expected event rates. In addition, it contains a modular ETL (Extract, Transform, Load) pipeline used to 
build reproducible workflows and generate intermediate data products. Analysis stages are implemented as modular
components (`ETLTask`s) that can be combined into user-defined workflows. 

## Features

- Modular ETL pipeline
- YAML-specified task configuration and execution
- Dynamic task registration and discovery
- Tools for processing astronomical survey data
- Monte Carlo simulation and survey sensitivity estimation
- Intermediate data products stored as Parquet files
- Unit tests written with `pytest`

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `pipeline/` | Generic ETL infrastructure, task coordination, and command-line interface |
| `tasks/` | Modular ETL tasks implementing the analysis |
| `microlensing/` | Shared helper functions |
| `notebooks/` | Visualization and exploratory analysis |
| `tests/` | Unit tests for the ETL framework and supporting code |

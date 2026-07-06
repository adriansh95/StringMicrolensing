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
- Dynamic task discovery and registration
- Tools for processing astronomical survey data
- Monte Carlo simulation and survey sensitivity estimation
- Intermediate data products stored as Parquet files
- Unit tests written with `pytest`
- Example workflow with a small demonstration dataset (planned)

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `pipeline/` | Generic ETL infrastructure, task coordination, and command-line interface |
| `tasks/` | Modular ETL tasks implementing the analysis |
| `microlensing/` | Shared helper functions |
| `notebooks/` | Visualization and exploratory analysis |
| `tests/` | Unit tests for the ETL framework and supporting code |
| `scripts/` | Utility scripts used during data preparation. |
| `demo/` (planned) | Example dataset and workflow demonstrating the pipeline |

## Quick Start
(planned)

## ETL Pipeline
This repository includes a modular ETL pipeline for scientific data processing. It is built around a base `ETLTask`
class. Subclasses ("tasks") define a transformation from input data products to output data products. An illustrative
workflow diagram is as follows:

```mermaid
graph TD;
    A[YAML workflow config] --> C[run_tasks.py]
    B[Task module] --> C

    C --> D[TaskCoordinator]

    D --> E1[ETLTask A]
    D --> E2[ETLTask B]
    D --> E3[ETLTask C]

    P0["Survey Data (Parquet)"] --> E1[ETLTask A]
    P0 --> E2[ETLTask B]
    P0 --> E3[ETLTask C]

    E1 --> P1[Parquet A]
    E2 --> P2[Parquet B]
    E3 --> P3[Parquet C]

    P1 --> E4[ETLTask D]
    P2 --> E4

    E4 --> P4[Parquet D]

    P4 --> N[Jupyter Notebook]
    P3 --> N

    N --> O[Plots / Figures]

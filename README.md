# Transformer Implementations

Various implementations of transformers etc.

## Prerequisites

- Python3.11

## Installation

Using poetry:

```sh
poetry install
```

Using pip:

```sh
pip install -r requirements.txt
```

Using pip to install development dependencies too:

```sh
pip install -r requirements.dev.txt
```

On Google Colab to avoid conflicts with preinstalled packages:

```sh
pip install -r requirements.colab.txt
```

## Running code

### CLI

An executable is provided in `./bin`. From the root directory run:

```sh
./bin/transformer_implementations
```

This will also pass on additional arguments.

### Programmatic API

Use the exposed `hello` function:

```py
import transformer_implementations

transformer_implementations.hello()
```

### Notebooks

A notebook is provided in `./notebooks` which demonstrates how to use the programmatic API. The notebook provides a link to open in Google Colab. To run locally start a Jupyter notebook server and open the notebook in the browser window which should open automatically:

```sh
jupyter notebook
```

## Development

The following scripts assume the requirements have been installed. If using poetry, they assume `poetry shell` has already been run or else they should be prefixed with `poetry run`.

### Lint

```sh
pylint ./transformer_implementations
```

### Typecheck

```sh
mypy
```

### Format

```sh
black ./transformer_implementations
```

### Generating requirements files

```sh
./scripts/generate_requirements.sh
```

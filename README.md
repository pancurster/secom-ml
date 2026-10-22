# SECOM ML

## Configure and install locally

From the project root, create a virtual environment, activate it in your
current shell, and install the project and its dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the activation command in each new terminal session where you want to use
the environment. The `dev` extra includes Pylint; run it with:

```bash
make lint
```

## Working with a Google Colab

### Install and import

In a Google Colab notebook, install this package from GitHub:

```python
!pip install --no-cache-dir git+https://github.com/pancurster/secom-ml.git
```

For example, import the data loader:

```python
from secomml.data_loader import load_secom
```

`load_secom` accepts a path to the SECOM dataset ZIP file and returns the
combined dataframe, labels dataframe, and sensor-data dataframe:

```python
secom, labels, data = load_secom("/content/secom.zip")

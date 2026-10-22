# SECOM ML

## Working with a Google Colab

### Install and import

In a Google Colab notebook, install this package from GitHub:

```python
!pip install --no-cache-dir git+https://github.com/pancurster/secom-ml.git
```

Import the data loader:

```python
from secomml.data_loader import load_secom
```

`load_secom` accepts a path to the SECOM dataset ZIP file and returns the
combined dataframe, labels dataframe, and sensor-data dataframe:

```python
secom, labels, data = load_secom("/content/secom.zip")

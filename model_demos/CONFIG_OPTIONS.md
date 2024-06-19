# Configuration Options

This document explains the different configuration options available for the Config class.

## Device Options

The following devices can be used:

- `e75`
- `e150`
- `n150`
- `n300`

## Chip Mode Options

The following chip modes must be specified for all devices:

- `single`
- `dual` (Only supported by the `n300` device)

### n300 Specific Options

If the `n300` device is selected, the following additional options must be specified:

#### batch_size

- Must be greater than 1 for the `n300` device.

## Example Usage

```python
from utils.config import Config

# Default configuration
config = Config(device="e75", chip_mode="single")

# n300 single chip with batch size 2
config = Config(device="n300", chip_mode="single", batch_size=2)

# n300 dual chip with batch size 4
config = Config(device="n300", chip_mode="dual", batch_size=4)

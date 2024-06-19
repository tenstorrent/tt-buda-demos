# utils/config.py


class Config:
    """
    Configuration class for setting up the device and other parameters.

    Attributes:
        device (str): The device to be used. Options are 'e75', 'e150', 'n150', 'n300'.
        chip_mode (str): The chip mode to be used. Options are 'single', 'dual'.
        batch_size (int): The batch size for processing. Must be greater than 1 for 'n300'.

    Raises:
        AssertionError: If 'n300' is selected but batch_size <= 1.
        AssertionError: If chip_mode is not 'single' or 'dual'.
        AssertionError: If 'dual' is selected but device is not 'n300'.
    """

    def __init__(self, device, chip_mode="single", batch_size=1):
        self.device = device
        self.chip_mode = chip_mode
        self.batch_size = batch_size

        # Validate chip_mode
        assert self.chip_mode in ["single", "dual"], "chip_mode must be either 'single' or 'dual'."

        # Validate batch_size and chip_mode for n300
        if self.device == "n300":
            assert self.batch_size > 1, "For n300 device, batch_size must be greater than 1."
        else:
            assert self.chip_mode == "single", "Only the n300 device supports 'dual' chip_mode."

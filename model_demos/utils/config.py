# utils/config.py
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

class DeviceType(Enum):
    e150 = "e150"
    n150 = "n150"
    n300 = "n300"

@dataclass
class Config:
    """
    Configuration class for setting up the device and other parameters.

    Attributes:
        device (DeviceType): The device type to run the model on.
        multi_chip (bool): Whether to run the model in dual-chip mode.
        batch_size (int): The batch size for input data.
        multi_card_devices (Optional[List[List[int]]]): The devices to run the model on in multi-card mode.

    Raises:
        AssertionError: If 'n300' is selected but batch_size <= 1.
        AssertionError: If 'multi_chip' or 'multi_card' is selected but device is not 'n300'.
    """
    device: DeviceType = DeviceType.e150
    multi_chip: bool = False
    batch_size: int = 1
    
    # Multi card dp vars
    precompiled_tti_path: Optional[str] = None
    multi_card_devices: Optional[List[List[int]]] = None
    
    def __post_init__(self):
        # Validate batch_size and chip_mode for n300
        if self.device == DeviceType.n300:
            assert self.batch_size > 1 and self.batch_size % 2 == 0, "For n300 device, batch_size must be greater than 1."
        else:
            assert not self.multi_chip and not self.multi_card_devices, "Only the n300 device supports 'dual' chip_mode."
            
        if self.multi_card_devices:
            assert self.batch_size > 1 and self.batch_size % len(self.multi_card_devices) == 0, "For multi-card mode, batch_size must be divisible by the number of cards."
            
    def n300_data_parallel(self) -> bool:
        return self.device == DeviceType.n300 and self.multi_chip
    
    def multi_card_dp(self) -> bool:
        return bool(self.multi_card_devices)
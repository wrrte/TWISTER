"""A logging scalar whose CPU conversion and rounding happen at log time."""

from dataclasses import dataclass


@dataclass
class ScalarInfo:
    tensor: object
    digits: int

    def get_value(self):
        return round(self.tensor.item(), self.digits)

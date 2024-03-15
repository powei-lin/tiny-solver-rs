from dataclasses import dataclass

@dataclass
class Costf:
    dx: float
    dy: float
    dtheta: float
    def __post_init__(self):
        self.name = "se2"
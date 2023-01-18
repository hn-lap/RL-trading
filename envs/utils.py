from enum import Enum


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class Actions(Enum):
    Sell = 0
    Buy = 1

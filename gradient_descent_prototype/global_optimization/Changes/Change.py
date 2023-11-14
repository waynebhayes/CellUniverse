useDistanceObjective = False


class Change:
    @property
    def is_valid(self) -> bool:
        pass

    @property
    def costdiff(self) -> float:
        pass

    def apply(self) -> None:
        pass

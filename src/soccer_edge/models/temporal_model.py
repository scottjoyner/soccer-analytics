from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalModelSpec:
    input_size: int
    output_classes: int = 3
    hidden_size: int = 128
    model_family: str = "recurrent"

    def validate(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.output_classes <= 1:
            raise ValueError("output_classes must be greater than one")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

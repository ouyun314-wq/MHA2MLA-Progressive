from dataclasses import dataclass, field
from typing import List


@dataclass
class ProgressiveCompressionArguments:
    rank_schedule: List[int] = field(
        default_factory=lambda: [16, 12, 8],
        metadata={"help": "Low rank for each progressive stage, e.g. [16, 12, 8]"},
    )
    steps_per_stage: List[int] = field(
        default_factory=lambda: [2000, 2000, 2000],
        metadata={"help": "Training steps for each stage"},
    )
    warmup_steps_per_stage: List[int] = field(
        default_factory=lambda: [200, 100, 100],
        metadata={"help": "Warmup steps for each stage"},
    )
    reset_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether to reset optimizer between stages"},
    )

    def __post_init__(self):
        n = len(self.rank_schedule)
        if len(self.steps_per_stage) != n:
            raise ValueError(
                f"steps_per_stage length ({len(self.steps_per_stage)}) "
                f"must match rank_schedule length ({n})"
            )
        if len(self.warmup_steps_per_stage) != n:
            raise ValueError(
                f"warmup_steps_per_stage length ({len(self.warmup_steps_per_stage)}) "
                f"must match rank_schedule length ({n})"
            )
        for i, (warmup, steps) in enumerate(
            zip(self.warmup_steps_per_stage, self.steps_per_stage)
        ):
            if warmup >= steps:
                raise ValueError(
                    f"Stage {i}: warmup_steps ({warmup}) must be less than steps ({steps})"
                )

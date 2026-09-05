from dataclasses import dataclass


@dataclass(frozen=True)
class DDSConfig:
    image_width: float
    image_height: float
    memory_capacity: int = 100
    read_quality: float = 0.50
    read_threshold: float = 0.35
    memory_temperature: float = 0.10
    memory_decay: float = 0.01
    idle_age: float = 0.10
    duplicate_threshold: float = 0.95
    refresh_delta: float = 0.05
    fall_iou: float = 0.15
    active_peak: float = 0.60
    commit_quality: float = 0.30
    measurement_map: float = 0.30
    measurement_identity: float = 0.50
    active_identity: float = 0.55
    lost_limit: int = 3
    reacquire_map: float = 0.35
    reacquire_identity: float = 0.60
    reacquire_count: int = 2
    association_iou: float = 0.30
    association_embedding: float = 0.60
    beta_max: float = 0.95
    ema: float = 0.95
    r_min: float = 1e-3
    eps: float = 1e-6

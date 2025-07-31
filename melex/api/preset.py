from melex.align.score import XGBoostModel, MelodicsModel, LinearModel, get_linear_model_default_weights
from melex.align.alignment import AlignConfig
from melex.data.io import PathLike

def get_melodics_config(hop_length: int = 1) -> AlignConfig:
    return AlignConfig(
        MelodicsModel({
            "error": 0.7144612760844599,
            "velocity": 0.7600615857281557,
            "miss": 0.9238979987939133,
            "note_mean": 0.4062946651493046,
            "shadow": 0.7234911137929942,
            "between": 0.22881185270861296,
            "above_between": 0.6658396693136427,
        }),
        same_key=False,
        same_speed=True,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=4,
        candidate_min_length=5,
        hop_length=hop_length,
        split_melody=True,
        structural_align=True,
        structural_max_difference=1,
        melody_min_recurrence=0.975,
        duration_tolerance=0.5,
        structural_only=False)

def get_xgboost_config(model_path: PathLike, hop_length: int = 1) -> AlignConfig:
    return AlignConfig(
        XGBoostModel(model_path),
        same_key=False,
        same_speed=True,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=4.9,
        candidate_min_length=5,
        hop_length=hop_length,
        split_melody=True,
        structural_align=True,
        structural_max_difference=1,
        melody_min_recurrence=0.975,
        duration_tolerance=0.5,
        structural_only=False)

def get_linear_config(hop_length: int = 1) -> AlignConfig:
    return AlignConfig(
        LinearModel(get_linear_model_default_weights()),
        same_key=False,
        same_speed=True,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=8.4,
        candidate_min_length=5,
        hop_length=hop_length,
        split_melody=True,
        structural_align=True,
        structural_max_difference=1,
        melody_min_recurrence=0.975,
        duration_tolerance=0.5,
        structural_only=False)

get_default_config = get_melodics_config
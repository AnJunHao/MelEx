from melign.align.score import XGBoostModel, MelodicsModel
from melign.align.alignment import AlignConfig

def get_melodics_config(hop_length: int = 2) -> AlignConfig:
    return AlignConfig(
        MelodicsModel(),
        same_key=False,
        same_speed=True,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=7,
        candidate_min_length=10,
        hop_length=hop_length,
        split_melody=True)

def get_xgboost_config(hop_length: int = 2) -> AlignConfig:
    return AlignConfig(
        XGBoostModel("dev_data/xgb_hop1_miss2_len8_large.json"),
        same_key=False,
        same_speed=True,
        speed_prior=1.0,
        variable_tail=True,
        local_tolerance=0.5,
        miss_tolerance=2,
        candidate_min_score=8,
        candidate_min_length=8,
        hop_length=hop_length,
        split_melody=True)
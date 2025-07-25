from melign.align.alignment import align, scan, AlignConfig, self_eval
from melign.align.eval_and_vis import plot_alignment, evaluate_melody
from melign.align.score import ScoreModel, MelodicsModel, RegressionModel, XGBoostModel
from melign.align.dp import weighted_interval_scheduling, f1_optimal_interval_scheduling
from melign.align import alignment, eval_and_vis, score, dp
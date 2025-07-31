from melex.align.alignment import align, scan, AlignConfig, self_eval
from melex.align.eval_and_vis import plot_alignment, evaluate_melody
from melex.align.score import ScoreModel, MelodicsModel, RegressionModel, XGBoostModel
from melex.align.dp import weighted_interval_scheduling, f1_optimal_interval_scheduling
from melex.align import alignment, eval_and_vis, score, dp
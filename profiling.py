import exmel
import cProfile

def main():
    dataset = exmel.Dataset('dataset')
    song = dataset[0]
    score_model = exmel.XGBoostModel('xgb_hop1_miss1_len10_big.json')
    config = exmel.AlignConfig(score_model=score_model, hop_length=32, candidate_min_score=12)
    exmel.align(song.melody, song.performance, config, verbose=True, defer_score=True)

cProfile.run('main()', sort='cumulative')
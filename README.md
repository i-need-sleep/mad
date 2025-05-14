# MAD
Implementation for MAUVE Audio Divergence (MAD) as described in [Aligning Text-to-Music Evaluation with Human Preferences](https://arxiv.org/abs/2503.16669). Sound examples for the meta-evaluation data can be found on our [demo page](https://mad-metric-83cde1d399d1.herokuapp.com/). The MusicPrefs dataset is available on [Huggingface](https://huggingface.co/datasets/i-need-sleep/musicprefs/tree/main).

## Installation
Install using pip:
```
pip install mad_metric
```
Or, install from source:
```
git clone https://github.com/i-need-sleep/mad.git
cd mad
pip install -e .
```

## Usage
Programmic usage:
```
from mad_metric import compute_mad
score = compute_mad(
    eval_dir='DIR_OF_AUDIO_FILES_TO_BE_EVALUATED',
    ref_dir='DIR_OF_REFERENCE_AUDIO_FILES',
    [optional arguments...]
)
```
Command line usage:
```
mad_metric [-h] [--eval_dir EVAL_DIR] [--ref_dir REF_DIR] [--eval_embs_dir EVAL_EMBS_DIR] [--ref_embs_dir REF_EMBS_DIR] [--log_csv LOG_CSV] [--batch_size BATCH_SIZE] [--model_name MODEL_NAME] [--layer LAYER]
                  [--aggregation AGGREGATION]
```
* `eval_dir`: The directory of audio files to be evaluated.
* `eval_embs_dir`: If both `eval_dir` and `eval_embs_dir` are specified, the computed embeddings of the files will be stored under `eval_embs_dir`. If only `eval_embs_dir` is provided, the MAUVE score will be computed directly using the provided embeddings. At least one of `eval_dir` and `eval_embs_dir` must be provided.
* `ref_dir`: Similar to `eval_dir`. We use [FMA-Pop](https://github.com/microsoft/fadtk/blob/main/datasets/fma_pop_tracks.csv) in our experiments.
* `ref_embs_dir`: Similar to `eval_embs_dir`.
* `log_csv`: The path to the csv file logging the MAD scores.
* `batch_size`: The batch size used when computing embeddings.
* `model_name`, `layer`, `aggregation`: Specifies which embeddings are used to compute the divergence. Defaults to the best setup with MERT according to our synthetic meta-evaluation with FMA-Pop as the reference set.

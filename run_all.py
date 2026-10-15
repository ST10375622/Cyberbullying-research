"""
Runs the full research pipeline, or chosen stages.

  python run_all.py                       # everything (GPU recommended)
  python run_all.py --quick               # 3k-row smoke test (minutes, CPU)
  python run_all.py --stages prepare eda classical bilstm
  python run_all.py --stages transformers --models bert   # e.g. on Colab
  python run_all.py --stages stats analysis report        # after training

Stages (in order):
  prepare       cleaning, deduplication, language ID, tokenising, harmonising, 70/15/15 split
  eda           dataset characterisation tables and figures
  classical     Logistic Regression, SVM, Random Forest (+ imbalance ablation)
  bilstm        Bidirectional LSTM
  transformers  BERT and AfriBERTa
  stats         McNemar / bootstrap / Friedman / Wilcoxon hypothesis tests
  analysis      feature importance, misclassified posts, length/language slices
  report        aggregate tables, comparison figures, deployment choice, findings summary
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

STAGES = ["prepare", "eda", "classical", "bilstm", "transformers", "stats", "analysis", "report"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=None)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Restrict models within the classical/transformers stages "
                         "(e.g. logreg svm, or bert afriberta)")
    ap.add_argument("--quick", action="store_true", help="Small smoke-test run (not research results)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Stratified subsample per dataset (applied equally to every model)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--transformer-epochs", type=int, default=None)
    ap.add_argument("--no-langdetect", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        config.apply_quick_mode()
    if args.max_rows:
        config.MAX_ROWS = args.max_rows
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    if args.transformer_epochs:
        config.TRANSFORMER = {**config.TRANSFORMER, "epochs": args.transformer_epochs}
    if args.no_langdetect:
        config.RUN_LANGDETECT = False
    if args.datasets:
        config.ACTIVE_DATASETS = args.datasets

    from src import utils
    log = utils.get_logger()
    log.info(f"Output directory: {config.OUTPUT_DIR} | data: {config.DATA_DIR} | quick={config.QUICK_MODE}")

    for stage in [s for s in STAGES if s in args.stages]:
        t0 = time.perf_counter()
        log.info(f"========== STAGE: {stage} ==========")
        if stage == "prepare":
            from src import preprocessing
            preprocessing.run(config.ACTIVE_DATASETS)
        elif stage == "eda":
            from src import eda
            eda.run(args.datasets)
        elif stage == "classical":
            from src import classical_models
            models = [m for m in (args.models or config.CLASSICAL_MODELS) if m in config.CLASSICAL_MODELS]
            if models:
                classical_models.run(args.datasets, models)
        elif stage == "bilstm":
            if args.models is None or "bilstm" in args.models:
                from src import bilstm_model
                bilstm_model.run(args.datasets)
        elif stage == "transformers":
            models = [m for m in (args.models or config.TRANSFORMER_MODELS) if m in config.TRANSFORMER_MODELS]
            if models:
                from src import transformer_models
                transformer_models.run(args.datasets, models)
        elif stage == "stats":
            from src import statistical_tests
            statistical_tests.run()
        elif stage == "analysis":
            from src import error_analysis
            error_analysis.run()
        elif stage == "report":
            from src import report
            report.run()
        log.info(f"========== {stage} finished in {time.perf_counter() - t0:,.1f}s ==========")


if __name__ == "__main__":
    main()

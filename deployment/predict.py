"""Command-line scoring of posts with the selected deployment model.

  python deployment/predict.py "you are so stupid nobody likes you" "have a nice day"
  python deployment/predict.py --lightweight "..."     # use the best classical model
"""
import argparse
import json

from predictor import CyberbullyingDetector

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--choice", default=None, help="path to deployment_choice.json")
    a = ap.parse_args()
    det = CyberbullyingDetector(a.choice, "lightweight_alternative" if a.lightweight else "calibrated")
    print(f"Model: {det.model_name} ({det.family}) trained on {det.train_dataset}")
    print(json.dumps(det.predict(a.texts), indent=2, ensure_ascii=False))

"""Regenerate the checked-in card from recorded metrics."""
import argparse
from telemetry_engine.ml.train import write_model_card
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--metrics",default="evals/valve_metrics.json")
    parser.add_argument("--output",default="docs/MODEL_CARD.md")
    args=parser.parse_args()
    write_model_card(args.metrics,args.output)

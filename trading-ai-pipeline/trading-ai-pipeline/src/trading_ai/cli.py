"""Usage:  python -m trading_ai.cli all --synthetic     (offline smoke test)
          python -m trading_ai.cli all                 (real data: needs internet)
"""
import argparse
import logging

from .config import load_config
from .pipeline import STAGES


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trading AI pipeline")
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--synthetic", action="store_true", help="use fake data, written under data/synthetic/")
    ap.add_argument("--config", help="path to an alternative config.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, synthetic=args.synthetic)
    STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()

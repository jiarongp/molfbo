import json
import logging
from pathlib import Path


def save_json(run_results, output_dir, filename="run_results.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    with open(output_file, "w") as outfile:
        json.dump(run_results, outfile)


def load_json(save_dir, filename="run_results.json"):

    with open(Path(save_dir) / filename, "r") as f:
        run_results = json.load(f)

    return run_results


def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = 0
    # prevent adding duplicates to logger
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s | %(message)s')
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

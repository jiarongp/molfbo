import json
import argparse
from pathlib import Path

from utils.helper import save_json, load_json


def aggregate_results(benchmarks, optimizers, root_dir=None):

    root_dir = Path(".") if root_dir is None else root_dir
    for benchmark in benchmarks:
        output_dir = root_dir / benchmark
        run_dirs = sorted(Path(output_dir).glob("runs_*"))

        problem_description = load_json(run_dirs[0], filename="problem_description.json")
        save_json(problem_description, output_dir, filename="problem_description.json")

        for opt in optimizers:
            opt_results = dict()

            for run_dir in run_dirs:
                result_files = sorted(run_dir.glob(f'{opt}_[0-9]*'))
                for result_file in result_files:
                    with open(result_file) as f:
                        result = json.load(f)

                for k, v in result[opt].items():
                    if k in opt_results.keys():
                        opt_results[k].append(v)
                    else:
                        opt_results[k] = [v]

            with open(output_dir / f"{opt}.json", 'w') as f:
                json.dump({f"{opt}": opt_results}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks in parallel.")

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    args = parser.parse_args()

    aggregate_results(args.benchmarks, args.optimizers, root_dir=Path(args.output))

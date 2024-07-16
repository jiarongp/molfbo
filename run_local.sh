#!/bin/bash

BENCHMARK="BraninCurrin"
OPTIMIZERS="Sobol MBORE"
# qParEGO

EVALS=50
STUDIES=20
OUTPUT="results"

for i in $(seq 1 $STUDIES);
do
    python mobo.py \
    --benchmark $BENCHMARK \
    --optimizers $OPTIMIZERS \
    --evaluations $EVALS \
    --studies 1 \
    --output "$OUTPUT/$BENCHMARK/runs_$i" \
    --verbose
done


# still need to run aggregate results
python aggregate_results.py --benchmarks $BENCHMARK --optimizers $OPTIMIZERS --output $OUTPUT
python plot.py --benchmarks $BENCHMARK --root $OUTPUT

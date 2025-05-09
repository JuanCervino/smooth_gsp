#!/bin/bash
# python scripts/test_graph_signals.py
# cd /path/to/your/project  # the parent of smooth_gsp



python scripts/main_train.py --lr 0.1 --epochs 10000 --coefficient_sob 1e-4 --lr 0.1 --method Sobolev
 
# python scripts/main_train.py --lr 0.1 --epochs 10000 --coefficient_sob 1e-4 --lr 0.1 --method nni

#!/bin/bash
export PYTHONPATH="/content/diffusion-med:$PYTHONPATH"
mpiexec -n 2 python scripts/super_res_train.py --config config/config_train.yaml
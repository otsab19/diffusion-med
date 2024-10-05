#!/bin/bash
export PYTHONPATH="/content/diffusion-med:$PYTHONPATH"  # Adjust PYTHONPATH to include the current directory

python scripts/super_res_sample.py --config config/config_test.yaml

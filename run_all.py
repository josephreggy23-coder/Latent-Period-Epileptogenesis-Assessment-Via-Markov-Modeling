"""
run_all.py - run the full pipeline end to end:
    Tier 1 (Gaussian HMM)  ->  Tier 2 (Bayesian HMM)  ->  consolidated summary
Usage:  python run_all.py
"""
import runpy
import sys

for mod in ["tier1_gaussian_hmm", "tier2_bayesian_hmm", "summarize"]:
    print(f"\n{'#'*70}\n# running {mod}\n{'#'*70}")
    sys.argv = [mod + ".py"]
    runpy.run_module(mod, run_name="__main__")

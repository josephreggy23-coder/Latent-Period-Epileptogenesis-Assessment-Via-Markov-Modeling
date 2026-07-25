"""
ECHO V5 - run the entire judge-hardened pipeline end to end:
  pipeline (items 1-5,8) -> survival (6) -> bayes (7 + hierarchical HMM) -> report
Usage:  python run_v5_all.py
"""
import runpy, sys
for mod in ["v5_pipeline", "v5_survival", "v5_bayes", "v5_report"]:
    print(f"\n{'#'*72}\n# {mod}\n{'#'*72}")
    sys.argv = [mod + ".py"]
    runpy.run_module(mod, run_name="__main__")

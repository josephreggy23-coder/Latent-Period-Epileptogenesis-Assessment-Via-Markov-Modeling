## Summary

Describe the scientific or software change and why it is needed.

## Validation

- [ ] `python -m pytest`
- [ ] `python -m compileall -q src scripts`
- [ ] Generated metrics and figures were refreshed if numerical behavior changed
- [ ] Documentation and changelog were updated

## Scientific integrity

- [ ] HMM features exclude the endpoint, dose, group, QC, and behavior columns
- [ ] Fish-level splitting and train-only preprocessing are preserved
- [ ] Unobserved fish keep an `NA` endpoint and are not scored as negatives
- [ ] Claims stay within a retrospective single-cohort analysis

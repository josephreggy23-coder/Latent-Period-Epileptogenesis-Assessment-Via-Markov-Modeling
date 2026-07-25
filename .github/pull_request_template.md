## Summary

Describe the scientific or software change and why it is needed.

## Validation

- [ ] `python -m pytest`
- [ ] `python -m compileall -q src scripts`
- [ ] Generated metrics and figures were refreshed if numerical behavior changed
- [ ] Documentation and changelog were updated

## Scientific integrity

- [ ] Synthetic values remain clearly labeled
- [ ] HMM features exclude truth, dose, group, QC, and behavior columns
- [ ] Fish-level splitting and train-only preprocessing are preserved
- [ ] Claims do not imply experimental efficacy or disease prediction

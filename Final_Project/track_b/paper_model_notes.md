# Paper Model Notes

## PatchTST

Source:
- alphaXiv: https://www.alphaxiv.org/abs/2211.14730
- official repo: https://github.com/yuqinie98/PatchTST

Core idea:
- Treat a time series like a sequence of patch tokens instead of single-step tokens.
- Use `patching` to turn a long sequence into fewer subseries tokens, which improves efficiency and keeps local temporal structure.
- Use `channel-independence`: each variable is modeled as a univariate series, while patch embedding and Transformer weights are shared across variables.
- The original paper targets forecasting and also provides a self-supervised masked pretraining variant.

Adaptation in this project:
- Implemented as `patchtst` in `paper_transformer_variants.py`.
- Uses the current Track B unsupervised pipeline instead of the paper's original forecasting head.
- Input remains the project's sequence window panel.
- Training objective is masked reconstruction over rolling windows.
- Learned window embeddings are clustered and passed into the same cluster-to-action backtest flow as other Track B models.

Approximation note:
- This is a PatchTST-inspired backbone adapted for regime representation learning, not a line-by-line reproduction of the original forecasting setup.

## Pathformer

Source:
- alphaXiv: https://www.alphaxiv.org/abs/2402.05956
- official repo: https://github.com/decisionintelligence/pathformer

Core idea:
- Model time series at multiple temporal scales instead of a single fixed resolution.
- Use multi-scale patch division to create different temporal resolutions.
- Use scale-specific attention blocks to capture both global and local temporal dependencies.
- Use an adaptive router/pathway mechanism so the model can dynamically emphasize different scales depending on the input dynamics.

Adaptation in this project:
- Implemented as `pathformer` in `paper_transformer_variants.py`.
- Uses multiple patch sizes in parallel over the same Track B window input.
- Each scale has a Transformer-style branch plus a local mixing branch.
- A learned router produces sample-wise scale weights and combines the scale reconstructions and embeddings.
- Final embeddings are clustered and evaluated using the same ranking/backtest framework.

Approximation note:
- This is a Pathformer-inspired multi-scale adaptive architecture for regime learning.
- It keeps the paper's high-level ideas of multi-scale division and adaptive pathways, but it is adapted to self-supervised window reconstruction rather than direct long-horizon forecasting.

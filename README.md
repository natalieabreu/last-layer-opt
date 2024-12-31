# last-layer-opt

Last layer optimization:
- Use last_layer_opt.py
- OPT can be set to NONE for standard training, BIAS to train the bias layer separately, or LAST_LAYER to train the last layer separately
- Bias/last layer training uses SGD while outer optimizer is Adam

Gradient norm tracking:
- Use gradient_norms.py
- OPT can be set to STANDARD for standard training or STANDARD+GN for standard training with gradient norm logging before and after last layer optimization
     - (Both do the same training since the last layer params are reset but STANDARD is useful for hyperparam sweeps)
  

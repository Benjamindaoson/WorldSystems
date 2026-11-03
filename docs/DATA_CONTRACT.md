# Data contract

The reference model accepts RGB video `[B,T,3,H,W]` and aligned actions `[B,T,A]` and emits next-frame predictions `[B,T,3,H,W]`. `H=W=image_size`, `A=action_dim`. All tensors must be finite. The synthetic dataset generates `T+1` frames and uses the last `T` as next-frame targets. Split seeds are separated by fixed offsets and training shuffles use epoch-specific seeds, enabling exact epoch-boundary recovery.

External models must declare their own tensor layout, units, coordinate frame, action dimension, version and weight SHA256 in `ModelManifest`. The current loader accepts only the built-in tiny model; unsupported external model families are not silently loaded.

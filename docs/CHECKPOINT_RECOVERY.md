# Checkpoint recovery

Each committed checkpoint directory contains `weights.safetensors`, `training.pt`, and `manifest.json`. Writing occurs in a sibling temporary directory, and the directory is renamed only after both artifacts and their SHA256 hashes are written. Verification checks schema, model family, expected file list and bytes before model loading. The optimizer, scheduler, CPU and CUDA RNG state, epoch, and global step are stored for local resume. Resume rejects a changed training configuration and starts at the next epoch.

The checkpoint format is designed for local artifacts. Do not accept untrusted `training.pt` files from remote users. Mid-epoch resume and distributed sharded checkpointing are not implemented.

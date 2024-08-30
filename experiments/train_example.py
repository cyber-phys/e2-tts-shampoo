import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from torch.optim import Adam
from datasets import load_dataset

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

# duration_predictor = DurationPredictor(
#     transformer = dict(
#         dim = 512,
#         depth = 6,
#     )
# )

epochs = 10
batch_size = 26
grad_accumulation_steps = 4

e2tts = E2TTS(
    cond_drop_prob = 0.0,
    transformer = dict(
        dim = 512,
        depth = 2,
        heads = 6,
        skip_connect_type = 'concat'
    ),
    mel_spec_kwargs = dict(
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24000,
    ),
    frac_lengths_mask = (0.7, 0.9)
)

train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])

optimizer = Adam(e2tts.parameters(), lr=7.5e-5)

name = 'e2tts_adam_aug_30-1'
trainer = E2Trainer(
    e2tts,
    optimizer,
    num_warmup_steps=5000,
    grad_accumulation_steps = grad_accumulation_steps,
    checkpoint_path = f"{name}.pt",
    log_file = f"{name}.txt",
    tensorboard_log_dir = f"runs/{name}",
    use_shampoo = True,
)

trainer.train(train_dataset, epochs, batch_size, save_step=1000)

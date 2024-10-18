import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig
from datasets import load_dataset

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 512,
        depth = 6,
    )
)

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    transformer = dict(
        dim = 512,
        depth = 12,
        skip_connect_type = 'concat'
    ),
)

train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])

optimizer = DistributedShampoo(
    e2tts.parameters(),
    lr=7.5e-5,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=False,
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-08,
    ),
)

trainer = E2Trainer(
    e2tts,
    optimizer,
    num_warmup_steps=20000,
    grad_accumulation_steps = 1,
    checkpoint_path = 'e2tts_shampoo.pt',
    log_file = 'e2tts_shampoo.txt'
)

epochs = 10
batch_size = 32

trainer.train(train_dataset, epochs, batch_size, save_step=1000)

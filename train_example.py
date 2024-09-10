import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from torch.optim import Adam
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

optimizer = Adam(e2tts.parameters(), lr=7.5e-5)

trainer = E2Trainer(
    e2tts,
    optimizer,
    num_warmup_steps=20000,
    grad_accumulation_steps = 1,
    checkpoint_path = 'e2tts.pt',
    log_file = 'e2tts.txt'
)

epochs = 10
batch_size = 32

# trainer.train(train_dataset, epochs, batch_size, save_step=1000)
import torch.multiprocessing as mp
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size):
    setup(rank, world_size)

    e2tts = E2TTS(
        duration_predictor = duration_predictor,
        transformer = dict(
            dim = 512,
            depth = 12,
            skip_connect_type = 'concat'
        ),
    )
    e2tts = e2tts.to(rank)
    e2tts = torch.nn.parallel.DistributedDataParallel(e2tts, device_ids=[rank])

    optimizer = Adam(e2tts.parameters(), lr=7.5e-5)

    trainer = E2Trainer(
        e2tts,
        optimizer,
        num_warmup_steps=20000,
        grad_accumulation_steps = 1,
        checkpoint_path = f'e2tts_{rank}.pt',
        log_file = f'e2tts_{rank}.txt'
    )

    train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])
    trainer.train(train_dataset, epochs, batch_size, save_step=1000)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

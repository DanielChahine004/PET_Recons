import logging
import random
import time
from pathlib import Path

from dotenv import load_dotenv

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from data.load_data import SpatioTemporalDataset
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from utils.get_model import get_model
from utils.collate import fixed_frame_collate_fn
from utils.plotting import plot_image
from utils.io import write_metrics_to_file
from utils.training import empty_torch_cache, get_loss_function
from math import sqrt
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env_path",
    type=str,
    default=".env",
    help="Input server name (default: Local)"
)

args = parser.parse_args()
env_path = args.env_path

load_dotenv(f"{BASE_DIR}/{env_path}", override=True)

NUM_CLASSES = int(os.getenv("NUM_CLASSES", 6))
IMAGE_HEIGHT_INNER = int(os.getenv("IMAGE_HEIGHT_INNER", 207))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", 41))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
BASE_LEARNING_RATE = float(os.getenv("BASE_LEARNING_RATE", 1e-8))
MAX_LEARNING_RATE = float(os.getenv("MAX_LEARNING_RATE", 1e-5))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
LOSS_THRESHOLD = float(os.getenv("LOSS_THRESHOLD", 5.0))
LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "MSE").lower()

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL = os.getenv("MODEL")

RUN_VAL = os.getenv("RUN_VAL", False) == "True"
DEVICE_TYPE = os.getenv("DEVICE_TYPE")

# Get rank and world size from environment
# These are provided by torch run
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
RANK = int(os.getenv("RANK", 0))


def training(device: torch.device, device_count: int, debug: bool = False):
    try:
        # Load model
        model = get_model(device, MODEL)
        # Move the model to the appropriate device
        model.to(device)

        # Set default backend
        backend = 'gloo'

        logging.info("Starting Training...")
        inner_data = os.getenv("INNER_DATA")
        outer_data = os.getenv("OUTER_DATA")
        train_dataset = SpatioTemporalDataset(inner_data, outer_data)
        if RUN_VAL:
            # Split dataset into training and validation sets
            dataset_size = len(train_dataset)
            val_size = int(VALIDATION_SPLIT * dataset_size) if RUN_VAL else 0
            train_size = dataset_size - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # For multiple GPUs across machines, 'nccl' is common, else use 'gloo'
        backend = 'nccl' if device.type == "cuda" else backend
        logging.debug(f"Using: '{backend}' backend")

        # DistributedSampler ensures each process sees a distinct subset of the data.
        if WORLD_SIZE > 1:
            # Initialize the process group before creating a DDP model
            dist.init_process_group(backend=backend, rank=RANK, world_size=WORLD_SIZE)
            # Use DDP for a model to distribute the model across nodes and/or GPUs
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK) if WORLD_SIZE > 1 else model
            # Use the Distributed sampler to distribute data across nodes and/or GPUs
            train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
            if RUN_VAL:
                val_sampler = DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
        else:
            # If running a single process, no sampler needed
            train_sampler = None
            val_sampler = None

        # DataLoaders
        # Note: shuffle only on a single process. If using DistributedSampler,
        # it handles shuffling internally. Avoid passing shuffle=True in that case.
        val_loader = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=device_count,
            shuffle=False,
            pin_memory=True,
            collate_fn=fixed_frame_collate_fn
        )
        if RUN_VAL:
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                sampler=val_sampler,
                shuffle=False,
                num_workers=device_count,
                pin_memory=True,
                collate_fn=fixed_frame_collate_fn
            )
        # Start a training loop
        training_loop(model, device, train_loader, train_sampler, val_loader, debug)
        return True
    except Exception as e:
        logging.exception(e)
        return False


def training_loop(model, device, train_loader, sampler, val_loader=None, debug: bool = False):
    writer = SummaryWriter(log_dir=f"results/{EXPERIMENT_NAME}/runs/{EXPERIMENT_NAME}")
    output_directory = f'results/{EXPERIMENT_NAME}/output/'
    try:
        if WORLD_SIZE == 1:
            if MODEL != "SparsePetnet3D":
                dummy_input = torch.randn(1, 4, 3, IMAGE_HEIGHT_INNER, IMAGE_WIDTH).to(device)
                writer.add_graph(model, [dummy_input])
        criterion = get_loss_function().to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LEARNING_RATE,
            betas=(0.9, 0.99),
            weight_decay=1e-4
        )
        scheduler = OneCycleLR(
            optimiser,
            max_lr=MAX_LEARNING_RATE,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader)
        )
        best_model_loss = float('inf')
        best_training_loss = float('inf')
        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            avg_loss = float('inf')
            train_running_loss = 0.0
            # Set epoch for sampler
            sampler.set_epoch(epoch) if sampler else None

            for batch_idx, burst in enumerate(train_loader):
                # 'frames' is a tensor of shape (B, 2, T, 311, 62)
                frames = burst['frames']
                # 'ground_truth' is a tensor of shape (B, 6)
                ground_truth = burst['ground_truth']

                inputs = frames.to(device).float()

                targets = ground_truth.to(device).float()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                for param in model.parameters():
                    param.grad = None

                loss.backward()
                optimiser.step()

                train_running_loss += loss.item()

                loss_metric = sqrt(loss.item()) if LOSS_FUNCTION == "mse" else loss.item()
                if loss_metric < LOSS_THRESHOLD and loss_metric < best_training_loss:
                    best_training_loss = loss_metric
                    logging.debug(f"A loss of '{loss_metric}' is below the loss threshold of: '{LOSS_THRESHOLD}'. Writing results to file...")
                    if len(outputs) > 50:
                        write_metrics_to_file(output_directory, EXPERIMENT_NAME, outputs, targets, loss_metric)

                if RUN_VAL:
                    # Validation
                    model.eval()
                    val_running_loss = 0.0

                    with torch.no_grad():
                        for val_burst in val_loader:
                            frames = val_burst['frames']
                            ground_truth = val_burst['ground_truth']
                            inputs = frames.to(device, non_blocking=True).float()
                            targets = ground_truth.to(device, non_blocking=True).float()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_running_loss += loss.item()

                    avg_val_loss = sqrt(val_running_loss / len(val_loader)) if LOSS_FUNCTION == "mse" else val_running_loss / len(val_loader)
                # Log training information
                if (batch_idx + 1) % 10 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    avg_train_loss = sqrt(train_running_loss / 10) if LOSS_FUNCTION == "mse" else train_running_loss / 10
                    logging.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx + 1}], Average Training Loss: {avg_train_loss:.4f}mm')
                    writer.add_scalar('Loss/train', avg_train_loss, global_step)
                    for param_group in optimiser.param_groups:
                        current_lr = param_group['lr']
                        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Current Learning Rate: {current_lr}")
                        writer.add_scalar('Learning Rate', current_lr, global_step)
                    if RUN_VAL:
                        writer.add_scalar('Loss/val', avg_val_loss, global_step)
                        logging.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx + 1}], Average Val Loss: {avg_val_loss:.4f}mm')

                    if device.type == "mps":
                        writer.add_scalar("System/GPU Allocated Memory (GB)", torch.mps.current_allocated_memory() * 1e-9, global_step)
                    if device.type == "cuda":
                        writer.add_scalar("System/GPU Allocated Memory (GB)", torch.cuda.memory_allocated() * 1e-9, global_step)

                    # Checkpoint best performing model
                    avg_loss = avg_val_loss if RUN_VAL else avg_train_loss
                    if best_model_loss > avg_loss:
                        logging.info(f"Previous Best Loss: {best_model_loss:.4f}mm")
                        logging.info(f"Avg Loss: {avg_loss:.4f}mm")
                        best_model_loss = avg_loss
                        torch.save(
                            model.state_dict(),
                            f"results/{EXPERIMENT_NAME}/model/{EXPERIMENT_NAME}_best.pth"
                        )
                        logging.info("Best model saved.")
                        writer.add_scalar("Model/Best Loss", best_model_loss, global_step)

                    train_running_loss = 0.0
                    empty_torch_cache()
            scheduler.step()
            if debug:
                #
                if epoch % int(os.getenv("DEBUG_EPOCH_CYCLE")) == 0:
                    random_frame_index = random.randint(0, frames.size(0) - 1)
                    # Plot image for debugging
                    plot_image(
                        writer,
                        frames[random_frame_index].unsqueeze(0),
                        "Inner",
                        f"results/{EXPERIMENT_NAME}/figures",
                        epoch * len(train_loader) + batch_idx,
                        save_image=False
                    )
        torch.save(model.state_dict(), f"results/{EXPERIMENT_NAME}/model/{EXPERIMENT_NAME}_last.pth")
    except Exception as e:
        logging.exception(e)
    finally:
        writer.close()
        torch.save(model.state_dict(), f"results/{EXPERIMENT_NAME}/model/{EXPERIMENT_NAME}_last.pth")


if __name__ == '__main__':
    from dotenv import load_dotenv
    from inference.run_inference import predict
    from utils.io import manage_directory
    import torch
    import argparse
    from utils.get_device import get_device_and_device_count

    logging_level = os.getenv("LOGGING_LEVEL", "DEBUG")
    logging.getLogger().setLevel(logging_level)

    logging.debug(f"Running Experiment: {EXPERIMENT_NAME}")
    logging.debug(f"Number of Epochs: {NUM_EPOCHS}")
    logging.debug(f"Batch Size: {BATCH_SIZE}")
    logging.debug(f"World Size: {WORLD_SIZE}")
    logging.debug(f"Run Val: {RUN_VAL}")
    logging.debug(f"Base learning rate: {BASE_LEARNING_RATE}")
    logging.debug(f"Max learning rate: {MAX_LEARNING_RATE}")

    experiment_path = f"results/{EXPERIMENT_NAME}"

    SEED = os.getenv("SEED")
    DEBUG = os.getenv("DEBUG", "False") == "True"
    PREDICT_ONLY = os.getenv("PREDICT_ONLY", "False") == "True"
    d, dc, _ = get_device_and_device_count()
    start_time = time.time()

    if not PREDICT_ONLY:
        manage_directory(experiment_path)
        assert training(d, dc, DEBUG)

    with open(f"{experiment_path}/metadata.txt", 'w') as file:
        file.write(f"Running training using dataset at path: {os.getenv('INNER_DATA')}\n\n")
        file.write(f"Using image of size: {IMAGE_HEIGHT_INNER} x {IMAGE_WIDTH}\n\n")

    experiment_time = time.time() - start_time

    with open(f"{experiment_path}/metadata.txt", 'a') as file:
        file.write(f"Experiment took: {experiment_time} seconds to complete\n\n")

    predict(
        f'results/{EXPERIMENT_NAME}/model/{EXPERIMENT_NAME}_best.pth',
        f'results/{EXPERIMENT_NAME}/output/',
        EXPERIMENT_NAME,
        d,
        MODEL,
        env_path
    )
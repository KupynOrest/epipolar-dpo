import os
import gc
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from diffusers.optimization import get_scheduler

from reward_lora.dataset import DPOLatentDataset
from reward_lora.loss import create_loss_strategy, LossOutput


@dataclass
class LoraTrainConfig:
    rank: int = 64
    alpha: float = 128.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    )


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    max_epochs: int = 1
    accumulate_grad_batches: int = 4
    precision: str = "bf16"
    strategy: str = "auto"
    beta: float = 500.0
    max_steps: int = 20000
    train_strategy: str = 'dpo'
    static_penalty_lambda: float = 0.0
    inlier_regression_lambda: float = 0.0


@dataclass
class LoggingConfig:
    output_path: str = "./output"
    save_top_k: int = -1
    checkpoint_every_n_steps: Optional[int] = None
    experiment_name: str = "reward_lora"


@dataclass
class ModelConfig:
    dit_path: str = MISSING
    pretrained_lora_path: Optional[str] = None
    inlier_regression_path: Optional[str] = None


@dataclass
class DataConfig:
    metadata_path: str = MISSING
    metric_name: str = "shadow_realism"
    metric_mode: str = "min"
    min_gap: float = 0.0
    metric_threshold: Optional[float] = None
    dataloader_num_workers: int = 1
    batch_size: int = 1


@dataclass
class RewardTrainerConfig:
    training: TrainingConfig = TrainingConfig()
    lora: LoraTrainConfig = LoraTrainConfig()
    logging: LoggingConfig = LoggingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


class FlowDPOTrainer(pl.LightningModule):
    def __init__(self, config: RewardTrainerConfig):
        super().__init__()
        self.config = config
        self.loss_strategy = create_loss_strategy(
            strategy=config.training.train_strategy,
            beta=config.training.beta,
            static_penalty_lambda=config.training.static_penalty_lambda,
            inlier_regression_lambda=config.training.inlier_regression_lambda,
            inlier_model_path=config.model.inlier_regression_path
        )

        # Load model
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(config.model.dit_path):
            model_manager.load_models([config.model.dit_path])
        else:
            dit_path = config.model.dit_path.split(",")
            model_manager.load_models([dit_path])

        # Initialize pipeline
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe.requires_grad_(False)
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
        )
        denoising_model = self.pipe.denoising_model()
        self.peft_model = get_peft_model(denoising_model, peft_config)
        self.pipe.dit = self.peft_model
        self.peft_model.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def print_gpu_memory_usage(self, stage=""):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"[{stage}] GPU Memory: Current: {current_memory:.2f}GB, "
                         f"Max: {max_memory:.2f}GB, Reserved: {reserved_memory:.2f}GB")
            return current_memory, max_memory
        return 0, 0

    def move_to_device(self, data_dict):
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self.move_to_device(value)
            else:
                result[key] = value
        return result

    def prepare_model_inputs(self, batch):
        x_win = batch["x_win"].to(self.device)
        x_lose = batch["x_lose"].to(self.device)
        prompt_emb_win = batch["prompt_emb_win"]
        prompt_emb_lose = batch["prompt_emb_lose"]

        for key in prompt_emb_win:
            if isinstance(prompt_emb_win[key], torch.Tensor):
                prompt_emb_win[key] = prompt_emb_win[key].to(self.device)
                prompt_emb_lose[key] = prompt_emb_lose[key].to(self.device)

        image_emb_win = {}
        if "image_emb_win" in batch:
            image_emb_win = batch["image_emb_win"]
            for key in image_emb_win:
                if isinstance(image_emb_win[key], torch.Tensor):
                    image_emb_win[key] = image_emb_win[key].to(self.device)

        image_emb_lose = {}
        if "image_emb_lose" in batch:
            image_emb_lose = batch["image_emb_lose"]
            for key in image_emb_lose:
                if isinstance(image_emb_lose[key], torch.Tensor):
                    image_emb_lose[key] = image_emb_lose[key].to(self.device)

        return {
            "x_win": x_win,
            "x_lose": x_lose,
            "prompt_emb_win": prompt_emb_win,
            "prompt_emb_lose": prompt_emb_lose,
            "image_emb_win": image_emb_win,
            "image_emb_lose": image_emb_lose,
        }

    def forward_model(self, model, noisy_latent, timestep, prompt_emb, image_emb):
        return model(
            noisy_latent,
            timestep=timestep,
            **prompt_emb,
            **image_emb,
            use_gradient_checkpointing=True,
        )

    def training_step(self, batch, batch_idx):
        if self.pipe.device != self.device:
            self.pipe.to(self.device)

        m_win = batch["m_win"].mean().item()
        m_lose = batch["m_lose"].mean().item()
        inputs = self.prepare_model_inputs(batch)

        noise = torch.randn_like(inputs['x_win'])
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (self.config.data.batch_size,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(device=self.device, dtype=self.pipe.torch_dtype)
        noisy_x_win = self.pipe.scheduler.add_noise(inputs['x_win'], noise, timestep)
        velocity_win = self.pipe.scheduler.training_target(inputs['x_win'], noise, timestep)

        noisy_x_lose = self.pipe.scheduler.add_noise(inputs['x_lose'], noise, timestep)
        velocity_lose = self.pipe.scheduler.training_target(inputs['x_lose'], noise, timestep)

        with torch.set_grad_enabled(True):
            velocity_win_pred = self.forward_model(
                self.peft_model,
                noisy_x_win,
                timestep,
                inputs["prompt_emb_win"],
                inputs["image_emb_win"]
            )

            velocity_lose_pred = self.forward_model(
                self.peft_model,
                noisy_x_lose,
                timestep,
                inputs["prompt_emb_lose"],
                inputs["image_emb_lose"]
            )

        if self.config.training.train_strategy == 'dpo':
            self.peft_model.disable_adapter_layers()
            with torch.no_grad():
                velocity_ref_win_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_win,
                    timestep,
                    inputs["prompt_emb_win"],
                    inputs["image_emb_win"]
                )

                velocity_ref_lose_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_lose,
                    timestep,
                    inputs["prompt_emb_lose"],
                    inputs["image_emb_lose"]
                )
            self.peft_model.enable_adapter_layers()
        else:
            velocity_ref_win_pred = None
            velocity_ref_lose_pred = None

        velocities = {
            "win": velocity_win_pred,
            "lose": velocity_lose_pred,
            "win_ref": velocity_ref_win_pred,
            "lose_ref": velocity_ref_lose_pred,
            "win_target": velocity_win,
            "lose_target": velocity_lose
        }
        inputs = {
            "noisy_x_win": noisy_x_win,
            "noisy_x_lose": noisy_x_lose,
            "timestep": timestep,
            "scheduler": self.pipe.scheduler,
            "prompt_emb_win": inputs["prompt_emb_win"],
            "prompt_emb_lose": inputs["prompt_emb_lose"],
            "image_emb_win": inputs["image_emb_win"],
            "image_emb_lose": inputs["image_emb_lose"],
        }

        loss_output: LossOutput = self.loss_strategy.calculate_loss(velocities, inputs)
        loss = loss_output.loss
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        self.log("win_metric", m_win, prog_bar=False)
        self.log("lose_metric", m_lose, prog_bar=False)
        self.log("metric_gap", m_win - m_lose, prog_bar=False)
        for key, item in loss_output.metrics.items():
            self.log(key, item, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.Adam(
            trainable_modules,
            lr=self.config.training.learning_rate,
        )
        total_steps = self.config.training.max_steps
        warmup_steps = int(total_steps * 0.015)

        lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()

        lora_state_dict = get_peft_model_state_dict(self.peft_model)

        corrected_state_dict = {}
        for key, value in lora_state_dict.items():
            if key.startswith('base_model.model.'):
                new_key = key.replace('base_model.model.', '')
            elif key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
            else:
                new_key = key

            corrected_state_dict[new_key] = value
        checkpoint.update(corrected_state_dict)


def setup_dataset(config: RewardTrainerConfig):
    dataset = DPOLatentDataset(
        metadata_path=config.data.metadata_path,
        metric_name=config.data.metric_name,
        metric_mode=config.data.metric_mode,
        min_gap=config.data.min_gap,
        metric_threshold=config.data.metric_threshold,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.dataloader_num_workers
    )

    return dataloader


def get_experiment_name(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


@hydra.main(config_path="config", config_name="train", version_base=None)
def train(config: RewardTrainerConfig):
    print(f"Training configuration:\n{OmegaConf.to_yaml(config)}")
    dataloader = setup_dataset(config)
    model = FlowDPOTrainer(config)

    current_time = datetime.now()
    date_str = current_time.strftime("%Y_%d_%m-%H_%M_%S")
    output_dir = os.path.join(config.logging.output_path, config.data.metric_name, date_str)
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")

    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="reward_lora"
    )
    experiment_name = get_experiment_name(config.logging.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.logging.output_path}/{experiment_name}",
        save_top_k=config.logging.save_top_k,
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
        filename="{epoch}-step={step}-{train_loss:.4f}",
        auto_insert_metric_name=False,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=config.training.precision,
        strategy=config.training.strategy,
        default_root_dir=output_dir,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, dataloader)
    print(f"Training completed. Model saved to {output_dir}")


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="reward_config", node=RewardTrainerConfig)

if __name__ == "__main__":
    train()

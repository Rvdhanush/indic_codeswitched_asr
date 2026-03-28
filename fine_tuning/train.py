import os
import json
import torch
import yaml
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import evaluate
import wandb

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_KEY = os.getenv("WANDB_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_path: str = "fine_tuning/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model_and_processor(config: dict):
    """Load Whisper base model and processor."""
    model_name = config["model"]["base_model"]
    logger.info(f"Loading base model: {model_name}")

    processor = WhisperProcessor.from_pretrained(
        model_name, token=HF_TOKEN,
        language=config["model"]["language"],
        task=config["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, token=HF_TOKEN
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return processor, model

def apply_lora(model, config: dict):
    """Apply LoRA adapters to Whisper model."""
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def oversample_by_type(samples: list, config: dict) -> list:
    """
    Oversample code_switched and switch_boundary samples.
    Undersample monolingual samples.
    This is the TARGETED part of targeted fine-tuning.
    """
    data_cfg = config["data"]
    oversampled = []

    for sample in samples:
        seg_type = sample.get("segment_type", "unknown")
        switch_count = sample.get("switch_count", 0)

        if seg_type == "code_switched":
            if switch_count > 2:
                # Extra weight for high-switch-count samples
                oversampled.extend(
                    [sample] * data_cfg["oversample_switch_boundary"]
                )
            else:
                # Duplicate code-switched samples 3x
                oversampled.extend(
                    [sample] * data_cfg["oversample_code_switched"]
                )
        elif seg_type in ("monolingual_tamil", "monolingual_english"):
            # Undersample monolingual to 50%
            if np.random.random() < data_cfg["undersample_monolingual"]:
                oversampled.append(sample)
        else:
            oversampled.append(sample)

    logger.info(
        f"Oversampled: {len(samples)} → {len(oversampled)} samples"
    )
    return oversampled

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper fine-tuning.
    Handles audio feature extraction and label padding.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset_for_training(samples: list, processor) -> list:
    """Convert raw samples to model input format."""
    processed = []
    for sample in samples:
        try:
            audio = np.array(sample["audio"], dtype=np.float32)
            inputs = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            labels = processor.tokenizer(
                sample["transcript"],
                return_tensors="pt"
            ).input_ids

            processed.append({
                "input_features": inputs.input_features[0],
                "labels": labels[0]
            })
        except Exception as e:
            logger.warning(f"Skipping sample: {e}")
            continue
    return processed

class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass for Whisper.

    transformers ≥ 4.50 injects `input_ids` into the batch internally during
    compute_loss (for per-sample loss normalisation). WhisperForConditionalGeneration
    uses `input_features` for the encoder — not `input_ids` — so we own the
    full forward pass here to prevent the TypeError.
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss


def compute_metrics_fn(processor):
    """Returns a compute_metrics function for Seq2SeqTrainer."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = wer_metric.compute(
            predictions=pred_str, references=label_str
        )
        return {"wer": round(wer, 4)}

    return compute_metrics

def train(config_path: str = "fine_tuning/config.yaml"):
    """Main training function."""
    config = load_config(config_path)

    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project="indic-codeswitched-asr",
            name="whisper-small-lora-tanglish",
            config=config
        )

    processor, model = load_model_and_processor(config)
    model = apply_lora(model, config)

    logger.info("Loading dataset splits...")
    import sys
    sys.path.insert(0, ".")
    from data.prepare_dataset import (
        authenticate_hf,
        load_indicvoices_tamil,
        build_dataset_splits
    )

    authenticate_hf()
    samples = load_indicvoices_tamil(max_samples=1500)
    splits = build_dataset_splits(samples)

    train_samples = oversample_by_type(splits["train"], config)
    val_samples = splits["validation"]

    logger.info("Preparing dataset for training...")
    train_data = prepare_dataset_for_training(train_samples, processor)
    val_data = prepare_dataset_for_training(val_samples, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    training_cfg = config["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_steps"],
        eval_strategy=training_cfg["evaluation_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        logging_steps=training_cfg["logging_steps"],
        fp16=training_cfg["fp16"] and DEVICE == "cuda",
        dataloader_num_workers=training_cfg["dataloader_num_workers"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],
        optim=training_cfg["optim"],
        predict_with_generate=True,
        generation_max_length=256,
        report_to="wandb" if WANDB_KEY else "none",
        push_to_hub=False,
    )

    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Starting training...")
    trainer.train()

    best_model_path = Path(training_cfg["output_dir"]) / "best_model"
    trainer.save_model(str(best_model_path))
    processor.save_pretrained(str(best_model_path))
    logger.info(f"Best model saved to {best_model_path}")

    if wandb.run:
        wandb.finish()

    return trainer

if __name__ == "__main__":
    train()

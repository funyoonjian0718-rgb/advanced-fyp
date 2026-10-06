import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import settings


class TrainingExample:
    def __init__(self, prompt: str, response: str) -> None:
        self.prompt = prompt
        self.response = response


def _read_examples(path: Path) -> list[TrainingExample]:
    if not path.is_file():
        raise FileNotFoundError(f"Training file was not found: {path}")
    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8-sig") as training_file:
        for line_number, line in enumerate(training_file, start=1):
            if not line.strip():
                continue
            record: Any = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Training row {line_number} must be a JSON object.")
            prompt = record.get("prompt") or record.get("instruction")
            response = record.get("response") or record.get("output")
            context = record.get("input") or record.get("context")
            if context:
                prompt = f"{prompt}\n\nContext:\n{context}"
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Training row {line_number} has no prompt.")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Training row {line_number} has no response.")
            examples.append(TrainingExample(prompt.strip(), response.strip()))
    if len(examples) < 4:
        raise ValueError(
            "Fine-tuning requires at least four curated JSONL examples."
        )
    return examples


def _train(
    examples: list[TrainingExample],
    output_dir: Path,
    epochs: int,
    max_length: int,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.llm_model_id,
        cache_dir=str(settings.model_cache_path),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class InstructionDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            example = examples[index]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Answer dietary questions cautiously. Follow the requested "
                        "output format and never invent evidence."
                    ),
                },
                {"role": "user", "content": example.prompt},
            ]
            prefix = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            complete = tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": example.response}],
                add_generation_prompt=False,
                tokenize=True,
            )
            complete = complete[:max_length]
            prefix_length = min(len(prefix), len(complete))
            labels = [-100] * prefix_length + complete[prefix_length:]
            return {
                "input_ids": complete,
                "attention_mask": [1] * len(complete),
                "labels": labels,
            }

    model_kwargs: dict[str, Any] = {
        "cache_dir": str(settings.model_cache_path),
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=InstructionDataset(),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the local open-weight model with a LoRA adapter."
    )
    parser.add_argument("--data", type=Path, required=True, help="Curated JSONL data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.llm_adapter_path,
        help="Directory where the adapter and tokenizer will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 64:
        parser.error("--epochs must be positive and --max-length must be at least 64.")

    examples = _read_examples(args.data)
    _train(examples, args.output, args.epochs, args.max_length)
    print(f"Saved the fine-tuned adapter to {args.output}.")


if __name__ == "__main__":
    main()

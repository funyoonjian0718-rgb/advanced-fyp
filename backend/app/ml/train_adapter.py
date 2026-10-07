import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import settings


class TrainingExample:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages


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
            if record.get("reviewed") is not True:
                raise ValueError(
                    f"Training row {line_number} must have reviewed=true after "
                    "a human has checked the example."
                )

            raw_messages = record.get("messages")
            if raw_messages is not None:
                if not isinstance(raw_messages, list) or not raw_messages:
                    raise ValueError(
                        f"Training row {line_number} has invalid messages."
                    )
                messages = []
                for message in raw_messages:
                    if (
                        not isinstance(message, dict)
                        or message.get("role")
                        not in {"system", "user", "assistant"}
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip()
                    ):
                        raise ValueError(
                            f"Training row {line_number} messages must contain "
                            "non-empty system, user, or assistant content."
                        )
                    messages.append(
                        {
                            "role": message["role"],
                            "content": message["content"].strip(),
                        }
                    )
                if messages[-1]["role"] != "assistant" or not any(
                    message["role"] == "user" for message in messages[:-1]
                ):
                    raise ValueError(
                        f"Training row {line_number} must end with an assistant "
                        "answer after a user message."
                    )
                examples.append(TrainingExample(messages))
                continue

            prompt = record.get("prompt") or record.get("instruction")
            response = (
                record.get("response")
                or record.get("output")
                or record.get("completion")
                or record.get("answer")
            )
            context = record.get("input") or record.get("context")
            prompt = prompt or record.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Training row {line_number} has no prompt.")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Training row {line_number} has no response.")
            if context:
                prompt = f"{prompt}\n\nContext:\n{context}"
            examples.append(
                TrainingExample(
                    [
                        {"role": "user", "content": prompt.strip()},
                        {"role": "assistant", "content": response.strip()},
                    ]
                )
            )
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
            messages = list(example.messages)
            if messages[0]["role"] != "system":
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": (
                            "Answer dietary questions cautiously. Follow the "
                            "requested output format and never invent evidence."
                        ),
                    },
                )
            prefix = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                tokenize=True,
            )
            complete = tokenizer.apply_chat_template(
                messages,
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
    (output_dir / "curated_training_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "human_reviewed_examples": len(examples),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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

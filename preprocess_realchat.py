from pathlib import Path

from datasets import load_dataset


data_source = "kakaocorp/realchat-v1.2-p1"
dataset = load_dataset(data_source)["train"]

dataset = dataset.map(
    lambda example: {"extra_info": {"chosen": example["chosen"], "rejected": example["reject"]}},
    batched=False,
    num_proc=8,
)
dataset = dataset.remove_columns(["chosen", "reject"])
dataset = dataset.add_column("data_source", [data_source for _ in range(len(dataset))])
dataset = dataset.add_column("ability", ["chat" for _ in range(len(dataset))])
dataset = dataset.add_column("reward_model", [{"ground_truth": None, "style": "model"} for _ in range(len(dataset))])


dataset = dataset.train_test_split(test_size=0.05, shuffle=True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

DIRPATH = Path("/data/lm-old_project_language-model_732/rw/sean/data")
train_dataset.to_parquet(DIRPATH / "realchat-v1.2-prompt" / "train.parquet")
test_dataset.to_parquet(DIRPATH / "realchat-v1.2-prompt" / "test.parquet")
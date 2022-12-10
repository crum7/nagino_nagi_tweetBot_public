from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import DataCollator
from transformers import Trainer
from transformers import TrainingArguments
from transformers import TrainingArguments
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")


dataset = load_dataset(
    path="text",
    data_files={
        "train":"train.txt",
        "test":"train.txt"
    }
)

tokenized_dataset= dataset.map(
    lambda examples: tokenizer(examples["text"]),
)
tokenized_dataset



training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    load_best_model_at_end=True,
    evaluation_strategy="steps"
)




data_collator=DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
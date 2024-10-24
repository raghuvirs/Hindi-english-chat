from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

english_text = "The United Nations leader says there is no military solution in Syria."

inputs = tokenizer(english_text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)

hindi_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"Translated text: {hindi_translation}")


from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

dataset = load_dataset('wmt16', 'en-hi')

def preprocess_function(examples):
    inputs = [ex for ex in examples['translation']['en']]
    targets = [ex for ex in examples['translation']['hi']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()


model.save_pretrained("./english-to-hindi-model")
tokenizer.save_pretrained("./english-to-hindi-model")

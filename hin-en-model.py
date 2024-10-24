from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-hi-en' 
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

hindi_text = "संयुक्त राष्ट्र के नेता कहते हैं कि सीरिया में कोई सैन्य समाधान नहीं है"

inputs = tokenizer(hindi_text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)

english_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"Translated text: {english_translation}")


from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

dataset = load_dataset('wmt16', 'hi-en')

def preprocess_function(examples):
    inputs = [ex for ex in examples['translation']['hi']]
    targets = [ex for ex in examples['translation']['en']]
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


model.save_pretrained("./hindi-to-english-model")
tokenizer.save_pretrained("./hindi-to-english-model")

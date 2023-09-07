import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-1.7b", torch_dtype=torch.float16)
print('model installed')
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-1.7b", use_fast=False)
print('tokenizer installed')
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
set_seed(101)
print('generator installed')

text = generator(
    "おはようございます、今日の天気は",
    max_length=30,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=5,
)

print('generated')

for t in text:
  print(t)

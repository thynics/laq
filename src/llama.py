from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention
from llama_quant import quant_layer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

for layer_index in range(5, 28):
    quant_layer(model.model.layers[layer_index])


model.to("cuda")

from transformers import GenerationConfig
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

# Helper: extract first ground truth answer
def get_ground_truth(example):
    return example["answers"]["text"][0]

squad = load_dataset("squad", split="validation[:20]")
from datasets import load_dataset
from metrics import qa_f1_score
# Evaluate F1 using the LLaMA-KIVI model
scores = []
for example in tqdm(squad):
    question = example["question"]
    context = example["context"]
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            output_scores=False,
        )

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    # Extract model's answer after the prompt
    answer = generated.split("Answer:")[-1].strip()
    ground_truth = get_ground_truth(example)

    score = qa_f1_score(answer, ground_truth)
    scores.append(score)
    break

average_f1 = sum(scores) / len(scores)
print(average_f1)
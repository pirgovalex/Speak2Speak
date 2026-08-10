import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
from hybrid_search import hybrid_search

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# setup bnb config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    model_kwargs={"quantization_config": bnb_config},
    max_new_tokens=60,
)

# Load tokenizer once at module level, not on every call
hf_tokenizer = AutoTokenizer.from_pretrained(model_id)

def clean_output(text):
    items = [m.strip() for m in text.split(',')]
    seen = set()
    unique_items = []
    for m in items:
        if m and m not in seen:
            unique_items.append(m)
            seen.add(m)
    return ', '.join(unique_items)

def llama_interact(q):
# mistral bible
    prime_text = (
    "You are a medical assistant. "
    "RULES (MUST FOLLOW EXACTLY):\n"
    "1. Answer ONLY with a comma-separated list of muscle names.\n"
    "2. NEVER add descriptions, functions, or explanations.\n"
    "3. EXCEPTION: Only give description if user question contains 'DESCRIPTION'.\n"
    "4. If no answer in context → reply: I do not have that information in the documents.\n"
    "5. DO NOT HALLUCINATE.\n"
    "6. Output format: Name1, Name2, Name3\n\n"
    "CONTEXT:\n"
)
    docs = hybrid_search(q)
    context = ("\n".join([doc.page_content for doc in docs]))
    full_prompt = prime_text + context + "\n\nUser Question: " + q
    output = pipe(full_prompt,
              max_new_tokens = 400,
              do_sample= True,
              temperature = 0.3,
              top_p=0.7,
              repetition_penalty = 1.15,
              return_full_text=False,
              eos_token_id=hf_tokenizer.eos_token_id
                 )[0]["generated_text"]
    result = clean_output(output)
    print(result)
    return result
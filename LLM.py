import torch
from transformers  import AutoTokenizer
from transformers import pipeline
from hybrid_search import hybrid_search

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
    model_id = "meta-llama/Llama-3.2-1B"

    pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=60,
    )
    prime_text = (    "You are a concise medical assistant. "
                      "Answer ONLY with a list of names unless specifically asked for details"
                      ". Do NOT include descriptions,"
                      " functions, nerves, fascia, or any extra"
                      " information unless you are specifically asked to do so ."
                    "Answer ONLY using clear, detailed,factual medical statements when asked"
                      " for description."
                      "When asked for anything different than a list"
                      " the keyword DESCRIPTION will we used "
    "Do NOT explain unless asked so via keyword EXPLAIN. Do NOT repeat yourself."
    "Separate items with commas. No explanations."
    ".\n\n"
    "CONTEXT:\n")
    docs = hybrid_search(q)
    context = ("\n".join([doc.page_content for doc in docs]))
    full_prompt = prime_text + context + "\n\nUser Question: " + q
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
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
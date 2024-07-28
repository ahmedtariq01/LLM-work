# Code to inference Hermes with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, MistralForCausalLM
#import bitsandbytes, flash_attn

# Define cache directories (optional)
cache_dir = "./cache"  # You can specify a custom cache directory if needed

# Initialize the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained( 'model/Nous-Hermes-2-Mistral-7B-DPO',
    trust_remote_code=True,
    cache_dir=cache_dir
)

model = MistralForCausalLM.from_pretrained( 'model/Nous-Hermes-2-Mistral-7B-DPO',
    torch_dtype=torch.float16,  # Use float16 for CPU
    device_map={"": "cpu"},  # Load model on CPU
    load_in_8bit=False,
    load_in_4bit=False,  # Loading in lower precision is generally not supported on CPU
    use_flash_attention_2=False,  # FlashAttention is often GPU-specific
    cache_dir=cache_dir
)


# Function to generate a response based on user input
def get_response(prompt):

    # Define maximum length (adjust as needed)
    max_length = 512  # You can change this value based on your needs

    # Tokenize input prompt

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).input_ids # No need for .to("cpu")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=750,
        temperature=0.8,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    return response

# Main loop to interact with the user
def main():
    print("Welcome to the interactive Q&A system. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        prompt = f"""system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.
user
{user_input}"""
        
        response = get_response(prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()

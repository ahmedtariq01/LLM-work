import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, MistralForCausalLM


# Define cache directories (optional)
cache_dir = "./cache"  # You can specify a custom cache directory if needed

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    trust_remote_code=True,
    cache_dir=cache_dir
)

model = MistralForCausalLM.from_pretrained(
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=False,  # Lower precision is generally not supported on CPU
    use_flash_attention_2=False,  # Ensure FlashAttention is disabled
    cache_dir=cache_dir
)


# Resize model embeddings to accommodate any changes if needed (though not adding special tokens now)
model.resize_token_embeddings(len(tokenizer))

# Prune the model
def prune_model(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Apply pruning to the Linear layers
            prune.l1_unstructured(module, name='weight', amount=0.3)  # Example: prune 5% of weights
            # Apply random_unstructured pruning to additional weights
            #prune.random_unstructured(module, name='weight', amount=0.2)  # Additional 20% pruning
            # Remove the pruning reparametrization to save memory
            prune.remove(module, 'weight')

    print("Model pruning complete.")

prune_model(model)

# Set the model to evaluation mode
model.eval()



# Function to generate a response based on user input
cache = {}

# Function to generate a response based on user input
def get_response(prompt):

    if prompt in cache:
        return cache[prompt]

    try:
        # Define maximum length (adjust as needed)
        max_length = 512  # You can change this value based on your needs

        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length, return_attention_mask=True).input_ids.to("cuda")
        
        # Ensure `inputs` is a dictionary
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
        else:
            # Handle case where tokenizer output is not a dictionary
            input_ids = inputs
            attention_mask = None

        # Set pad token id if necessary
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        with torch.no_grad():
        # Generate response from model
          generated_ids = model.generate(
              input_ids,
              attention_mask=attention_mask,
              max_new_tokens=500,
              temperature=0.8,
              repetition_penalty=1.1,
              num_beams=10,  # Use beam search with 5 beams
              length_penalty=2.0,  # Adjust the length penalty as needed
              #do_sample=True,
              eos_token_id=tokenizer.eos_token_id,
              pad_token_id=pad_token_id
          )

        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        response = clean_response(response)  # Apply cleaning logic
        response = format_response(response)  # Apply formatting logic
        cache[prompt] = response
        return response

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, there was an error processing your request. Please try again later."


def clean_response(response):
    # Example: Remove "system" and extra spaces
    response = response.replace("system", "").strip()
    return response

def format_response(response):
    # Example: Add newlines after sentences
    paragraphs = response.split('. ')
    formatted_response = '\n\n'.join(paragraphs)
    return formatted_response

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
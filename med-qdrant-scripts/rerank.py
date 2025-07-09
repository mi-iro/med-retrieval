import torch
from transformers import AutoTokenizer
import os
import sys
from tqdm import tqdm
import time
import math

# --- [New] vLLM Imports ---
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# --- [New] Global Variables and vLLM Model Setup ---

# 1. Model Path
RERANKER_PATH = "/mnt/public/lianghao/wzr/med_reseacher/med-retrieval/model/Qwen3-Reranker-0.6B"

# 2. vLLM and Tokenizer Initialization
print("Initializing vLLM model and tokenizer...")
# Detect number of available GPUs
number_of_gpu = torch.cuda.device_count()
if number_of_gpu == 0:
    print("No GPU detected. vLLM requires a GPU.", file=sys.stderr)
    sys.exit(1)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH)

# Load the model with vLLM
# enable_prefix_caching=True is beneficial for reranking where the system prompt/instruction is shared
model = LLM(
    model=RERANKER_PATH,
    tensor_parallel_size=number_of_gpu,
    max_model_len=1024*2,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.98,
    max_num_seqs=1024,
)
print(f"vLLM model loaded on {number_of_gpu} GPU(s).")

# 3. Define Model-specific Tokens and Parameters
# Get token IDs for "yes" and "no"
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

# Define the suffix that prompts the model to think and respond
# This is applied after the user message via the chat template's generation prompt
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
max_length = 1024*2 - len(suffix_tokens) # Reserve space for the suffix

# Define sampling parameters for vLLM
# We want to get the log probabilities for "yes" and "no"
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1,
    logprobs=2, # Request logprobs for top 2 tokens
    allowed_token_ids=[token_true_id, token_false_id], # Constrain the output
)

# Default instruction task
DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'

# --- [New] vLLM Helper Functions ---

def format_and_tokenize_inputs(pairs, instruction):
    """
    Formats query-document pairs using the Qwen chat template and tokenizes them.
    """
    # Create the structured messages for the chat template
    messages = [
        [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"}
        ]
        for query, doc in pairs
    ]
    
    # Apply the chat template and tokenize
    # We don't add a generation prompt here as we will manually add our specific suffix tokens
    tokenized_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Truncate if necessary and append the required suffix tokens
    tokenized_inputs = [ele[:max_length] + suffix_tokens for ele in tokenized_inputs]
    
    # Wrap in vLLM's TokensPrompt object
    prompts = [TokensPrompt(prompt_token_ids=ele) for ele in tokenized_inputs]
    return prompts


@torch.no_grad()
def compute_scores_vllm(batch_prompts):
    """
    Processes a batch of tokenized prompts with vLLM and computes relevance scores.
    """
    # 1. Run model inference using vLLM's generate method
    outputs = model.generate(batch_prompts, sampling_params, use_tqdm=False)
    
    all_scores = []
    
    # 2. Process outputs to calculate scores
    for output in outputs:
        # The logprobs for the single generated token are in the last step
        final_logprobs = output.outputs[0].logprobs[-1]
        
        # Extract the log probability for "yes" and "no" tokens
        # As requested, access the .logprob attribute from the Logprob object
        true_logit = final_logprobs.get(token_true_id)
        false_logit = final_logprobs.get(token_false_id)

        # Handle cases where a token might not be in the top-k logprobs
        true_logprob = true_logit.logprob if true_logit is not None else -100.0
        false_logprob = false_logit.logprob if false_logit is not None else -100.0

        # 3. Convert log probabilities to probabilities and normalize
        true_score = math.exp(true_logprob)
        false_score = math.exp(false_logprob)
        
        final_score = true_score / (true_score + false_score)
        all_scores.append(final_score)
        
    return all_scores


# --- [Adapted] Main Function ---

def get_reranked_scores(query, articles, batch_size=2048): # vLLM can handle much larger batch sizes
    """
    Computes reranked scores for a query and a list of articles using vLLM.
    
    Args:
        query (str): A single query string.
        articles (list[str]): A list of document strings to be ranked.
        batch_size (int): The batch size for processing.

    Returns:
        tuple[list[float], float]: A tuple containing the list of relevance scores 
                                   and the total time taken.
    """
    start_time = time.time()
    all_scores = []
    
    # Create (query, document) pairs
    pairs = [(query, doc) for doc in articles]

    # Process in batches
    for i in tqdm(range(0, len(pairs), batch_size), desc="Reranking with vLLM"):
        batch_pairs = pairs[i:i + batch_size]
        
        # Format and tokenize the batch
        prompts = format_and_tokenize_inputs(batch_pairs, DEFAULT_INSTRUCTION)
        
        # Compute scores for the batch
        scores = compute_scores_vllm(prompts)
        all_scores.extend(scores)
    
    end_time = time.time()
    return all_scores, end_time - start_time


if __name__ == "__main__":
    
    query = "diabetes treatment"
    
    articles = [
        "Metformin is a first-line medication for the treatment of type 2 diabetes.",
        "Diabetes mellitus and its chronic complications. Diabetes mellitus is a major cause of morbidity and mortality, and it is a major risk factor for early onset of coronary heart disease. Complications of diabetes are retinopathy, nephropathy, and peripheral neuropathy.",
        "Diagnosis and Management of Central Diabetes Insipidus in Adults. Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons...",
        "Adipsic diabetes insipidus. Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance with significant associated morbidity and mortality...",
        "Nephrogenic diabetes insipidus: a comprehensive overview. Nephrogenic diabetes insipidus (NDI) is characterized by the inability to concentrate urine that results in polyuria and polydipsia...",
        "Impact of Salt Intake on the Pathogenesis and Treatment of Hypertension. Excessive dietary salt (sodium chloride) intake is associated with an increased risk for hypertension..."
    ]

    # Get scores using the new vLLM-accelerated function
    rerank_scores, duration = get_reranked_scores(query, articles)
    
    # Combine articles with their scores
    ranked_results = sorted(zip(articles, rerank_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: {query}\n")
    print(f"Reranking with vLLM took {duration:.4f} seconds.")
    print("--- Reranked Results (Score DESC) ---")
    for doc, score in ranked_results:
        print(f"Score: {score:.4f}\nDocument: {doc[:100]}...\n")

    # Clean up vLLM model parallel resources
    print("Destroying model parallel group...")
    destroy_model_parallel()
    print("Done.")
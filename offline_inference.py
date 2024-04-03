from vllm import LLM, SamplingParams
from my_print import print_time
# Sample prompts.
'''
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
'''
batch_size=4
seq_len=128
prompt_token_ids=[[50 for _ in range(seq_len)] for _ in range(batch_size)]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-1.3b", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
#outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
"""warm up prompt"""
llm.my_add_request(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
o,is_prompt=print_time(llm.my_run_prompt)
print(len(o))
print(len(prompt_token_ids))
print(is_prompt)

llm.my_add_request(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
o,is_prompt=print_time(llm.my_run_prompt)
print(len(o))
print(len(prompt_token_ids))
print(is_prompt)
llm.my_add_request(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
o,is_prompt=print_time(llm.my_run_prompt)
print(len(o))
print(len(prompt_token_ids))
print(is_prompt)

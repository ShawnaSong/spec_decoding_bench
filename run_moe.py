from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=2,    
    pipeline_parallel_size=2,  
    dtype="float16",         
    gpu_memory_utilization=0.9, 
)

# 推理
outputs = llm.generate(
    ["Tell me a story about a dragon."],
    sampling_params={"temperature": 0.7, "top_p": 0.9}
)

for output in outputs:
    print(output.outputs[0].text)

from vllm import LLM, SamplingParams

# sample prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# sampling parameters object
sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=1024, stop=["\n"])

# create LLM object
# Note: vllm by default downloads the model from the Hugging Face Hub
llm = LLM(model="facebook/opt-125m")

# Generate texts from the list of prompts
# Output is a list of RequestOutput objects that contains the prompt, generated text and other metadata
outputs = llm.generate(prompts, sampling_params)

# print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 100)



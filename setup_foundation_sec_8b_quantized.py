import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def check_requirements():
    print("Checking requirements...")
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable GPU is required.")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_memory < 7.5:
        raise RuntimeError(f"GPU memory ({gpu_memory:.2f}GB) insufficient. ~7.5GB free required for quantized model.")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Free Memory: {gpu_memory:.2f}GB")
    print("Requirements check passed.")

def load_model():
    print("Loading Foundation-Sec-8B model and tokenizer from local directory...")
    model_path = r"D:\AI\Models\fdtn-ai\Foundation-Sec-8B"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model directory {model_path} not found. Please download the model.")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    print("Model and tokenizer loaded successfully.")
    return tokenizer, model

def test_model(tokenizer, model):
    print("Running test inference...")
    prompt = """CVE-2021-44228 is a remote code execution flaw in Apache Log4j2 via unsafe JNDI lookups (“Log4Shell”). The CWE is CWE-502.
    CVE-2015-10011 is a vulnerability about OpenDNS OpenResolve improper log output neutralization. The CWE is"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=3,
            do_sample=True,
            temperature=0.1,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        print(f"Test output: {response}")
    except Exception as e:
        print(f"Test inference failed: {str(e)}")

def main():
    try:
        check_requirements()
        tokenizer, model = load_model()
        test_model(tokenizer, model)
        print("Setup completed successfully! You can now use quantized Foundation-Sec-8B locally.")
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
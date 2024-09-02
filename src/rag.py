from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from retrieval import get_prompts

model_path = "vinai/PhoGPT-4B-Chat"  

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.to("cuda")
model.eval()  

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) 

def rag(question, topk=3):
    prompt = get_prompts(question, topk=topk)
    input_ids = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(  
    inputs=input_ids["input_ids"].to("cuda"),  
    attention_mask=input_ids["attention_mask"].to("cuda"),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id)  

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Trả lời:")[1]
    print(response)
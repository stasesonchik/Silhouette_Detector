import torch
from transformers import AutoModel, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import time

model = AutoGPTQForCausalLM.from_quantized(
    'openbmb/MiniCPM-o-2_6-int4',
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True
)
tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM-o-2_6-int4',
    trust_remote_code=True
)

model.init_tts()

from PIL import Image
# test.py
image = Image.open('/home/user/Desktop/mcot/1734946487.3900414.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

t1 = time.time()
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(f"{time.time() - t1} sec to chat generate q1")

question = 'Describe the person'
msgs = [{'role': 'user', 'content': [image, question]}]

t1 = time.time()
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(f"{time.time() - t1} sec to chat generate q2")


question = 'Is there a dog in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

t1 = time.time()
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(f"{time.time() - t1} sec to chat generate q3")


t1 = time.time()
vis, pix, sz, tgt = model.only_visual_one(image)
print(f"{time.time() - t1} sec to viz generate")


img_features = {
    'vision_hidden_states': vis,
    'image_sizes': sz,
}

print(sz[0][0].size())
print(vis[0].size())

t1 = time.time()
res = model.chat_one_vision_hidden_state(img_features, "What is in the image?", tokenizer=tokenizer, max_new_tokens=100)
print(f"{time.time() - t1} sec to text generate q1")
print(res)

t1 = time.time()
res = model.chat_one_vision_hidden_state(img_features, "Describe the person", tokenizer=tokenizer, max_new_tokens=100)
print(f"{time.time() - t1} sec to text generate q2")
print(res)

t1 = time.time()
res = model.chat_one_vision_hidden_state(img_features, "Is there a dog in the image?", tokenizer=tokenizer, max_new_tokens=100)
print(f"{time.time() - t1} sec to text generate q3")
print(res)

t1 = time.time()
res = model.chat_one_vision_hidden_state(img_features, "Describe the person", tokenizer=tokenizer, max_new_tokens=100)
print(f"{time.time() - t1} sec to text generate q2")
print(res)

"""
GPT-2 模型下载脚本
运行此脚本可将 GPT-2 预训练模型下载到本地 ./models/gpt2/ 目录。
训练时会从本地路径加载，无需联网。
"""
from transformers import GPT2Model, GPT2Tokenizer

# 下载并保存 GPT-2 模型到本地
model = GPT2Model.from_pretrained("gpt2")
model.save_pretrained("./models/gpt2")  # 保存 config.json + pytorch_model.bin

# 下载并保存 tokenizer 到本地
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained('./models/gpt2')
import os

paths = [
    "./eval/models/checkpoints/llava-1.5-7b-hf",
    "./eval/models/checkpoints/llava-1.5-13b-hf",
    "./eval/models/checkpoints/blip2-flant5xxl",
    "./eval/models/checkpoints/instructblip-vicuna-7b",
    "./eval/models/checkpoints/instructblip-vicuna-13b",
    "./eval/models/checkpoints/instructblip-flan-t5-xl",
    "./eval/models/checkpoints/instructblip-flan-t5-xxl",
    "./eval/models/checkpoints/Qwen-VL-Chat",
    "./eval/models/checkpoints/InternVL-Chat-V1-5-quantable",
    "./eval/models/checkpoints/Phi-3-vision-128k-instruct",
    "./eval/models/checkpoints/llava-v1.6-mistral-7b-hf",
    "./eval/models/checkpoints/llava-v1.6-vicuna-7b-hf",
    "./eval/models/checkpoints/LLaVA-NeXT-Video-7B-hf",
    "./eval/models/checkpoints/LLaVA-NeXT-Video-7B-DPO-hf",
    "./eval/models/checkpoints/llava-interleave-qwen-7b-hf",
    "./eval/models/checkpoints/llava-interleave-qwen-7b-dpo-hf",
    "./eval/models/checkpoints/vila-1.5-3b",
    "./eval/models/checkpoints/vila-1.5-3b-s2",
    "./eval/models/checkpoints/vila-1.5-8b",
    "./eval/models/checkpoints/vila-1.5-13b",
    "./eval/models/checkpoints/cambrian-8b",
    "./eval/models/checkpoints/Mantis-8B-Idefics2",
    "./eval/models/checkpoints/Mantis-llava-7b",
    "./eval/models/checkpoints/Mantis-8B-siglip-llama3",
    "./eval/models/checkpoints/Mantis-8B-clip-llama3",
    "./eval/models/checkpoints/video-llava-7b",
    "./eval/models/checkpoints/chat-univi-7b",
    "./eval/models/checkpoints/chat-univi-13b",
    "./eval/models/checkpoints/pllava-7b",
    "./eval/models/checkpoints/pllava-13b"
]

for path in paths:
    os.makedirs(path, exist_ok=True)


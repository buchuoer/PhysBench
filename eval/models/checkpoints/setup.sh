cd ./eval/models/checkpoints/llava-1.5-7b-hf
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/llava-1.5-13b-hf
huggingface-cli download llava-hf/llava-1.5-13b-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/blip2-flant5xxl
huggingface-cli download Salesforce/blip2-flan-t5-xxl --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/instructblip-vicuna-7b
huggingface-cli download Salesforce/instructblip-vicuna-7b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/instructblip-vicuna-13b
huggingface-cli download Salesforce/instructblip-vicuna-13b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/instructblip-flan-t5-xl
huggingface-cli download Salesforce/instructblip-flan-t5-xl --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/instructblip-flan-t5-xxl
huggingface-cli download Salesforce/instructblip-flan-t5-xxl --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/Qwen-VL-Chat
huggingface-cli download Qwen/Qwen-VL-Chat --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/InternVL-Chat-V1-5-quantable
huggingface-cli download failspy/InternVL-Chat-V1-5-quantable --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/Phi-3-vision-128k-instruct
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/llava-v1.6-mistral-7b-hf
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/llava-v1.6-vicuna-7b-hf
huggingface-cli download llava-hf/llava-v1.6-vicuna-7b-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/LLaVA-NeXT-Video-7B-hf
huggingface-cli download llava-hf/LLaVA-NeXT-Video-7B-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/LLaVA-NeXT-Video-7B-DPO-hf
huggingface-cli download llava-hf/LLaVA-NeXT-Video-7B-DPO-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/llava-interleave-qwen-7b-hf
huggingface-cli download llava-hf/llava-interleave-qwen-7b-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/llava-interleave-qwen-7b-dpo-hf
huggingface-cli download llava-hf/llava-interleave-qwen-7b-dpo-hf --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/vila-1.5-3b
huggingface-cli download Efficient-Large-Model/VILA1.5-3b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/vila-1.5-3b-s2
huggingface-cli download Efficient-Large-Model/VILA1.5-3b-s2 --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/vila-1.5-8b
huggingface-cli download Efficient-Large-Model/Llama-3-VILA1.5-8B --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/vila-1.5-13b
huggingface-cli download Efficient-Large-Model/VILA1.5-13b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/cambrian-8b
huggingface-cli download nyu-visionx/cambrian-8b --local-dir . --local-dir-use-symlinks False
pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
cd ./eval/models/checkpoints/Mantis-8B-Idefics2
huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2 --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/Mantis-llava-7b
huggingface-cli download TIGER-Lab/Mantis-llava-7b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/Mantis-8B-siglip-llama3
huggingface-cli download TIGER-Lab/Mantis-8B-siglip-llama3 --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/Mantis-8B-clip-llama3
huggingface-cli download TIGER-Lab/Mantis-8B-clip-llama3 --local-dir . --local-dir-use-symlinks False
# ---------------------------- video
cd ./eval/models/checkpoints/video-llava-7b
huggingface-cli download LanguageBind/Video-LLaVA-7B --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/chat-univi-7b
huggingface-cli download Chat-UniVi/Chat-UniVi --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/chat-univi-13b
huggingface-cli download Chat-UniVi/Chat-UniVi-13B --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/pllava-7b
huggingface-cli download ermu2001/pllava-7b --local-dir . --local-dir-use-symlinks False
cd ./eval/models/checkpoints/pllava-13b
huggingface-cli download ermu2001/pllava-13b --local-dir . --local-dir-use-symlinks False

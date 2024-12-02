<div align="center">
<h1> <img src="assets/physbench.png" width="40" /> PhysBench </h1>
</div>
<h5 align="center">
    <a href="https://physbench.github.io/">🌐 Homepage</a> | <a href="https://huggingface.co/datasets/USC-GVL/PhysBench">🤗 Dataset</a> | <a href="todo">📑 Paper</a> | <a href="https://github.com/USC-GVL/PhysBench/tree/main/eval">💻 Code</a> | <a href="https://eval.ai/web/challenges/challenge-page/2384/overview">🔺 EvalAI</a>
</h5>


This repo contains evaluation code for the paper "[PhysBench: Benchmarking and Enhancing VLMs for Physical World Understanding](todo)"
If you like our project, please give us a star ⭐ on GitHub for latest update.

![Alt text](assets/tease_scores.png)


## 🔔News

 **🔥[2024-09-23]: Evaluation for test set is now available on [EvalAI](todo). We welcome all submissions and look forward to your participation!**

We will be releasing the training split of the dataset, 3D assets and the remaining code in the near future.

## Introduction
**Vision-Language Models (VLMs)** have emerged as promising tools for building **embodied agents**, whereas their lack of **physical world understanding** hampers their effectiveness in real-world applications. To address this challenge, we present **PhysBench**, a comprehensive benchmark designed to evaluate and enhance VLMs' understanding of the physical world across diverse and complex tasks.

PhysBench comprises **100,000 entries** of interleaved video-image-text data, and the data is categorized into four major classes: **physical object properties**, **physical object relationships**, **physical scene understanding**, and **physics-driven dynamics**, covering **19 subclasses** and **10 distinct capability dimensions**.

Our extensive experiments on 39 representative VLMs reveal significant gaps in physical world understanding, likely due to the absence of physical knowledge in their training data. To improve VLMs' physical understanding, we propose an agent-based method called **PhysAgent**, which leverages prior physical knowledge and expert model assistance to enhance physical world understanding capabilities.

Furthermore, we demonstrate that improving VLMs’ understanding of the physical world can significantly facilitate the deployment of embodied agents in real-world scenarios, moving towards bridging the gap between human and machine intelligence in comprehending the physical world.

![Alt text](assets/data_cases_full.png)
## Dataset Summary

The complete **PhysBench** dataset consists of 100,000 entries, organized into 19 subclasses and 10 distinct capability dimensions. For convenience, we selected a subset of 10,002 entries, which are more challenging and diverse, as the test set, and 200 entries as the validation set for parameter choosing.

- **val**: 200 examples used for model development, validation, or for those with limited computing resources.
- **test**: 10,002 examples for standard evaluation (include val). Notably, the answer labels for test will NOT be publicly released. 
- **train**: The remaining 89,998 examples.

<img src="assets/stat.png" width="900" />

## Evaluation

We have released the **dataset** on Hugging Face [**🤗 Dataset**](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), and you can use the **evaluation tool** in the [eval](eval) folder to reproduce the results from the paper or to explore the performance of your own model !

<img src="assets/cor.png" width="800" />

## 🏆 Leaderboard

This is a subset of the leaderboard for the PhysBench test set. For the complete leaderboard, please refer to the [**🌐 Homepage**](https://physbench.github.io/).

You can submit your model’s predictions for the **test set** on **[EvalAI](https://eval.ai/web/challenges/challenge-page/2287/overview)**.

| **#** | **Model**                | **ALL**   | **object** | **spatial** | **environment** | **phenomena** |
| ----- | ------------------------ | --------- | ---------- | ----------- | --------------- | ------------- |
| -     | **Human Performance**    | **95.87** | 97.10      | 95.67       | 94.91           | 95.68         |
| 1     | **GPT-4o 🥇**             | **49.49** | 56.91      | 64.80       | 30.15           | 46.99         |
| 2     | **Gemini-1.5-pro 🥈**     | **49.11** | 57.26      | 63.61       | 36.52           | 41.56         |
| 3     | **Gemini-1.5-flash 🥉**   | **46.07** | 57.41      | 52.24       | 34.32           | 40.93         |
| 4     | GPT-4o-mini          | **43.15** | 53.54      | 44.24       | 30.59           | 42.90         |
| 5     | GPT-4V               | **41.26** | 49.59      | 45.77       | 26.34           | 42.15         |
| 6     | LLaVA-interleave     | **41.00** | 47.23      | 44.62       | 35.64           | 37.21         |
| 7     | LLaVA-interleave-dpo | **40.83** | 47.97      | 42.67       | 33.73           | 38.78         |
| 8     | Phi-3V               | **38.42** | 43.67      | 37.92       | 34.93           | 36.92         |
| 9     | Mantis-siglip-llama3 | **37.64** | 42.47      | 32.78       | 36.83           | 37.51         |
| 10    | LLaVA-NV-dpo         | **37.43** | 38.83      | 44.31       | 33.86           | 37.21         |
| 11    | Mantis-Idefics2      | **37.39** | 41.97      | 41.44       | 29.53           | 36.56         |
| 12    | VILA-1.5-13B         | **37.15** | 40.53      | 40.15       | 31.96           | 36.07         |
| 13    | Mantis-clip-llama3   | **36.92** | 40.61      | 35.11       | 32.45           | 38.36         |
| 14    | Mantis-LLaVA         | **36.69** | 44.48      | 30.45       | 36.25           | 34.73         |
| 15    | LLaVA-NV             | **35.42** | 38.33      | 30.83       | 34.00           | 37.17         |
| 16    | VILA-1.5-3B          | **34.11** | 32.40      | 33.02       | 34.84           | 35.78         |
| 17    | VILA-1.5-3B-s2       | **33.07** | 33.14      | 30.26       | 35.72           | 33.00         |
| 18    | VILA-1.5-8B          | **32.85** | 33.41      | 29.88       | 30.85           | 35.91         |

## Disclaimers

Some of the data in PhysBench has been annotated based on existing datasets, as noted in the appendix of the paper. For the forensics detection task, we manually collected images that are publicly available through online searches. We have made every effort to comply with applicable copyright laws and ensure proper attribution of the images used in this paper. However, if you are the copyright holder of any image included in our work and believe its use conflicts with your licensing agreements, please [contact](#contact) us directly. We are committed to promptly addressing any legitimate concerns.

## Contact
- Wei Chow: xieqiao@zju.edu.cn
- Jiageng Mao:  jiagengm@usc.edu
- Yue Wang:   yue.w@usc.edu

### Acknowledgements

Our evaluation code implementation was partially inspired by [TaskMeAnything](https://github.com/JieyuZ2/TaskMeAnything).

## Citation

**BibTeX:**
```bibtex
@article{

}
```

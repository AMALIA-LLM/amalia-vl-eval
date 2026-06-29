<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AMALIA-LLM/amalia-llm.github.io/main/source/_static/logo/logo-color-white.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AMALIA-LLM/amalia-llm.github.io/main/source/_static/logo/logo-color-black.png">
  <img src="https://raw.githubusercontent.com/AMALIA-LLM/amalia-llm.github.io/main/source/_static/logo/logo-color-black.png" alt="AMALIA" width="300"/>
</picture>

<br/>

**A Fully Open Large Language Model for European Portuguese**

[![Website](https://img.shields.io/badge/Website-amaliallm.pt-blue)](https://amaliallm.pt/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-amalia--llm-yellow)](https://huggingface.co/amalia-llm)
[![GitHub](https://img.shields.io/badge/GitHub-AMALIA-green)](https://github.com/AMALIA-LLM/AMALIA)
[![AMALIA-VL Paper](https://img.shields.io/badge/Paper-AMALIA--VL-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.19100)
[![PorTEXTO Paper](https://img.shields.io/badge/Paper-PorTEXTO-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.19096)
[![License](https://img.shields.io/badge/License-MIT%20%2B%20Apache%202.0-blue)](LICENSE)

</div>

This repository contains the code for evaluating the AMALIA-VL model, supporting the evaluations reported in:

- **AMALIA-VL: A Native European Portuguese Open-Source Vision and Language Model** — [arXiv:2606.19100](https://arxiv.org/abs/2606.19100)
- **PorTEXTO: A European Portuguese Benchmark for Visual Text Extraction** — [arXiv:2606.19096](https://arxiv.org/abs/2606.19096)

## amalia-vl-eval

This repository includes **amalia-vl-eval**, a benchmark suite for evaluating Large Vision-Language Models with a focus on European Portuguese (PT-PT).

The suite groups **18 leaf tasks** under the tag **`amalia_vl_eval_pt`** and spans diagram and chart understanding, document and infographic VQA, scene-text and OCR, perception and college-level multimodal reasoning, visual mathematics, real-world and embodied spatial reasoning, object-hallucination probing, captioning, and referring-expression grounding. The **PorTEXTO** (visual text extraction) and **CARAVELA** (cultural visual entities) benchmarks are separate native pt-PT groups with their own tags, described below.

The supported tasks are the following:

| Task (`--tasks`) | pt-PT dataset | Original dataset |
|---|---|---|
| `ai2d_pt` | [![AI2D-PT](https://img.shields.io/static/v1?label=&message=AI2D-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/AI2D-PT) | [![lmms-lab/ai2d](https://img.shields.io/static/v1?label=&message=lmms-lab%2Fai2d&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/ai2d) |
| `chartqa_pt` | [![ChartQA-PT](https://img.shields.io/static/v1?label=&message=ChartQA-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/ChartQA-PT) | [![lmms-lab/ChartQA](https://img.shields.io/static/v1?label=&message=lmms-lab%2FChartQA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/ChartQA) |
| `docvqa_pt` | [![DocVQA-PT](https://img.shields.io/static/v1?label=&message=DocVQA-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/DocVQA-PT) | [![lmms-lab/DocVQA](https://img.shields.io/static/v1?label=&message=lmms-lab%2FDocVQA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| `infovqa_pt` | [![InfographicVQA-PT](https://img.shields.io/static/v1?label=&message=InfographicVQA-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/InfographicVQA-PT) | [![lmms-lab/DocVQA](https://img.shields.io/static/v1?label=&message=lmms-lab%2FDocVQA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| `textvqa_pt` | [![TextVQA-PT](https://img.shields.io/static/v1?label=&message=TextVQA-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/TextVQA-PT) | [![lmms-lab/textvqa](https://img.shields.io/static/v1?label=&message=lmms-lab%2Ftextvqa&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/textvqa) |
| `ocrbench_pt` | [![OCRBench-PT](https://img.shields.io/static/v1?label=&message=OCRBench-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/OCRBench-PT) | [![echo840/OCRBench](https://img.shields.io/static/v1?label=&message=echo840%2FOCRBench&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/echo840/OCRBench) |
| `mme_pt` | [![MME-PT](https://img.shields.io/static/v1?label=&message=MME-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/MME-PT) | [![lmms-lab/MME](https://img.shields.io/static/v1?label=&message=lmms-lab%2FMME&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/MME) |
| `mmmu_pt` | [![MMMU-PT](https://img.shields.io/static/v1?label=&message=MMMU-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/MMMU-PT) | [![lmms-lab/MMMU](https://img.shields.io/static/v1?label=&message=lmms-lab%2FMMMU&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/MMMU) |
| `mmmu_pro_pt` | [![MMMU-Pro-PT](https://img.shields.io/static/v1?label=&message=MMMU-Pro-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/MMMU-Pro-PT) | [![MMMU/MMMU_Pro](https://img.shields.io/static/v1?label=&message=MMMU%2FMMMU_Pro&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| `mmstar_pt` | [![MMStar-PT](https://img.shields.io/static/v1?label=&message=MMStar-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/MMStar-PT) | [![Lin-Chen/MMStar](https://img.shields.io/static/v1?label=&message=Lin-Chen%2FMMStar&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| `seedbench_pt` | [![SEED-Bench-PT](https://img.shields.io/static/v1?label=&message=SEED-Bench-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/SEED-Bench-PT) | [![lmms-lab/SEED-Bench](https://img.shields.io/static/v1?label=&message=lmms-lab%2FSEED-Bench&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/SEED-Bench) |
| `mathvision_pt` | [![MATH-Vision-PT](https://img.shields.io/static/v1?label=&message=MATH-Vision-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/MATH-Vision-PT) | [![MathLLMs/MathVision](https://img.shields.io/static/v1?label=&message=MathLLMs%2FMathVision&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/MathLLMs/MathVision) |
| `realworldqa_pt` | [![RealWorldQA-PT](https://img.shields.io/static/v1?label=&message=RealWorldQA-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/RealWorldQA-PT) | [![lmms-lab/RealWorldQA](https://img.shields.io/static/v1?label=&message=lmms-lab%2FRealWorldQA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/RealWorldQA) |
| `embspatial_pt` | [![EmbSpatial-Bench-PT](https://img.shields.io/static/v1?label=&message=EmbSpatial-Bench-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/EmbSpatial-Bench-PT) | [![FlagEval/EmbSpatial-Bench](https://img.shields.io/static/v1?label=&message=FlagEval%2FEmbSpatial-Bench&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) |
| `pope_pt` | [![POPE-PT](https://img.shields.io/static/v1?label=&message=POPE-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/POPE-PT) | [![lmms-lab/POPE](https://img.shields.io/static/v1?label=&message=lmms-lab%2FPOPE&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/POPE) |
| `coco2017_cap_pt` | [![COCO-Caption2017-PT](https://img.shields.io/static/v1?label=&message=COCO-Caption2017-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/COCO-Caption2017-PT) | [![lmms-lab/COCO-Caption2017](https://img.shields.io/static/v1?label=&message=lmms-lab%2FCOCO-Caption2017&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017) |
| `refcoco_bbox_pt` | [![RefCOCO-PT](https://img.shields.io/static/v1?label=&message=RefCOCO-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/RefCOCO-PT) | [![lmms-lab/RefCOCO](https://img.shields.io/static/v1?label=&message=lmms-lab%2FRefCOCO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/RefCOCO) |
| `refcoco_bbox_rec_pt` | [![RefCOCO-PT](https://img.shields.io/static/v1?label=&message=RefCOCO-PT&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/RefCOCO-PT) | [![lmms-lab/RefCOCO](https://img.shields.io/static/v1?label=&message=lmms-lab%2FRefCOCO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/lmms-lab/RefCOCO) |

PorTEXTO is a separate, native pt-PT group (tag **`portexto`**, aggregating group **`portexto_pt`**, run with `--tasks portexto_pt`) and is **not** part of the `amalia_vl_eval_pt` suite:

| Task (`--tasks`) | pt-PT dataset | Original dataset |
|---|---|---|
| `portexto_handwritten_pt` | [![PorTEXTO](https://img.shields.io/static/v1?label=&message=PorTEXTO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/PorTEXTO) (`handwritten`) | — |
| `portexto_handwritten_full_page_pt` | [![PorTEXTO](https://img.shields.io/static/v1?label=&message=PorTEXTO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/PorTEXTO) (`handwritten_full_page`) | — |
| `portexto_in_the_wild_pt` | [![PorTEXTO](https://img.shields.io/static/v1?label=&message=PorTEXTO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/PorTEXTO) (`in_the_wild`) | — |
| `portexto_synthetic_pt` | [![PorTEXTO](https://img.shields.io/static/v1?label=&message=PorTEXTO&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/PorTEXTO) (`synthetic`) | — |

CARAVELA is a separate, native pt-PT group (tag **`caravela`**, run with `--tasks caravela`) and is **not** part of the `amalia_vl_eval_pt` suite:

| Task (`--tasks`) | pt-PT dataset | Original dataset |
|---|---|---|
| `caravela_mcq` | [![CARAVELA](https://img.shields.io/static/v1?label=&message=CARAVELA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/CARAVELA) (`caravela_mcq.parquet`) | — |
| `caravela_vqa` | [![CARAVELA](https://img.shields.io/static/v1?label=&message=CARAVELA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/CARAVELA) (`caravela_vqa.parquet`) | — |
| `caravela_reasoning` | [![CARAVELA](https://img.shields.io/static/v1?label=&message=CARAVELA&color=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/amalia-llm/CARAVELA) (`caravela_reasoning.parquet`) | — |


### PorTEXTO

[PorTEXTO](https://arxiv.org/abs/2606.19096) is the first benchmark for contemporary, culturally relevant **pt-PT visual text extraction**. It is built from frontier-LVLM transcriptions reviewed by native speakers, and spans four subsets — `handwritten`, `handwritten_full_page`, `in_the_wild`, and `synthetic` — so you can see the characteristic drop in quality from synthetic to real-world text.

Scoring uses **ANLS** and **BLEU-1** (unigram, via `sacrebleu`). The four subsets also carry the `portexto` tag, and the aggregating group `portexto_pt` reports the mean of both metrics.

### CARAVELA

CARAVELA (**C**ultural **A**wareness and **R**ecognition **A**ssessment for **V**isual **E**ntity **L**iteracy and **A**nalysis) is a native European Portuguese benchmark probing knowledge of culturally relevant visual entities. It spans three subtasks — `caravela_mcq` (4-option multiple choice), `caravela_vqa` (short open-ended answers), and `caravela_reasoning` (step-by-step reasoning).

Scoring uses **accuracy**, normalised **exact/contains match**, and **token-level F1** respectively. The three subtasks also carry the `caravela` tag, and the aggregating group `caravela` reports the macro-average across them.

## Setup

`lmms-eval` requires Python ≥ 3.10. Using a conda environment (the project default):

```bash
conda create -n amalia-vl-eval python=3.12 -y
conda activate amalia-vl-eval

git clone https://github.com/AMALIA-LLM/amalia-vl-eval.git
cd amalia-vl-eval
pip install -e ".[all]"
```

## Running Evaluations

After activating the environment, evaluate the AMALIA-VL checkpoint on the full pt-PT suite. How you launch depends on how many GPUs you have.

### Single GPU

Use the standard `lmms-eval` entry point and pass `device_map=auto` so the model is placed on the GPU:

```bash
python -m lmms_eval \
  --model llava_hf \
  --model_args pretrained=amalia-llm/AMALIA-VL-SFT,device_map=auto \
  --tasks amalia_vl_eval_pt \
  --batch_size 1 \
  --output_path ./results/ \
  --log_samples
```

You can also pass a single task or a comma-separated list. For example, to run only PorTEXTO on a single GPU:

```bash
python -m lmms_eval \
  --model llava_hf \
  --model_args pretrained=amalia-llm/AMALIA-VL-SFT,device_map=auto \
  --tasks portexto_pt \
  --batch_size 1 \
  --output_path ./results/ \
  --log_samples
```

Or to run the full CARAVELA group (its three leaf tasks):

```bash
python -m lmms_eval \
  --model llava_hf \
  --model_args pretrained=amalia-llm/AMALIA-VL-SFT,device_map=auto \
  --tasks caravela \
  --batch_size 1 \
  --output_path ./results/ \
  --log_samples
```

### Multiple GPUs

Launch with `accelerate` for data parallelism across `N` GPUs (set `--num_processes` to the number of GPUs). Do **not** pass `device_map` here — each process places the model on its own GPU automatically:

```bash
accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
  --model llava_hf \
  --model_args pretrained=amalia-llm/AMALIA-VL-SFT \
  --tasks amalia_vl_eval_pt \
  --batch_size 1 \
  --output_path ./results/ \
  --log_samples
```

> **Note:** `OPENAI_API_KEY` may need to be set to any placeholder value, because the upstream `mmmu`/`mathvision` utils eagerly construct an OpenAI judge client at import time. **The judge is never called** — the pt-PT tasks use rule-based metrics (exact match / accuracy).


## Citation

If you use amalia-vl-eval, please cite the AMALIA-VL and PorTEXTO papers:

```bibtex
@misc{amalia-vl-2026,
    title  = {AMALIA-VL: A Native European Portuguese Open-Source Vision and Language Model},
    author = {Diogo Gl{\'o}ria-Silva and Jo{\~a}o Cardeira and Manuel Letras da Luz and
              Afonso Simpl{\'i}cio and Gon{\c{c}}alo Vinagre and Diogo Tavares and
              Rafael Ferreira and In{\^e}s Calvo and In{\^e}s Vieira and David Semedo and
              Jo{\~a}o Magalh{\~a}es},
    year    = {2026},
    eprint  = {2606.19100},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url     = {https://arxiv.org/abs/2606.19100}
}

@misc{portexto-2026,
    title  = {PorTEXTO: A European Portuguese Benchmark for Visual Text Extraction},
    author = {Jo{\~a}o Cardeira and Diogo Gl{\'o}ria-Silva and Manuel Letras da Luz and
              Rafael Ferreira and Diogo Tavares and David Semedo and Jo{\~a}o Magalh{\~a}es},
    year    = {2026},
    eprint  = {2606.19096},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url     = {https://arxiv.org/abs/2606.19096}
}
```

## Acknowledgements

amalia-vl-eval is a fork of [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) by the EvolvingLMMs-Lab, which is itself a fork of EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). We thank both teams. The pt-PT layer and PorTEXTO benchmark are contributed by the AMALIA project. We recommend you read through the lmms-eval documentation for relevant information on how to add models or tasks.

## License

This project inherits the dual license of upstream [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval):

- The **main pipeline / structure-related code** is under the **MIT License**, consistent with [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
- The **multimodal tasks and models** (code under `lmms_eval/tasks` and `lmms_eval/models`, including the pt-PT layer and PorTEXTO contributed here) are under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) for the full text of both.

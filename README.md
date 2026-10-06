# `MemDec-Universal` <em>(SLM optimized for Local AI)</em> <div style="text-align: right;"><a href="https://github.com/Gii-DE/MemDec-Universal" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Gii--DE-github.com%2FGii--DE%2FMemDec--Universal-darkred?labelColor=black&logo=github&style=plastic&logoColor=white&v=G-7PZXQPZPGX" title="Gii-DE · MemDec-Universal (GitHub)"></a> <a id="contact" href="https://g-dea.com/#contact" target="_blank" rel="noopener noreferrer" style="scroll-margin-top: 80px;"><img src="https://img.shields.io/badge/📧_G--DEA.COM-darkred?style=plastic&v=G-7PZXQPZPGX" alt="g-dea.com" title="Contact Me (Website)"></a></div>

This project implements the [*Memory Decoder (MemDec)*](https://arxiv.org/pdf/2508.09874) approach to extend a base *LLM* (Large Language Model) with domain-specific knowledge through a modular use-case system under [<kbd>MemDec-Universal/`config`</kbd>](config/HowTo.md#config-files), where each domain is defined by its own config files — demonstrated on the German legal domain using the open-source [*Open Legal Data*](https://de.openlegaldata.io/pages/api/) (a non-profit platform providing free API access to German court decisions, laws, and legal texts). The pipeline enables *plug-and-play* domain adaptation across lightweight, instruction-tuned *SLMs* (Small Language Models), including [*Gemma 3-270m*](https://huggingface.co/unsloth/gemma-3-270m-it), [*Qwen 3.5-0.8B*](https://huggingface.co/unsloth/Qwen3.5-0.8B), and [*SmolLM3-3B*](https://huggingface.co/unsloth/SmolLM3-3B), with the active use-case selected via the environment variable `MEMDEC_USECASE` and all pipeline behavior controlled by config files alone. To maximize efficiency, this repository integrates official [*Unsloth models*](https://unsloth.ai/docs/get-started/unsloth-model-catalog). By replacing standard PyTorch layers with custom-written NVIDIA Triton kernels, it achieves 2x to 3x faster training speeds and significantly reduces memory overhead.

## <em>Approach</em>: `MemDec vs. RAG`
MemDec offers a parametric alternative to *Retrieval-Augmented Generation* (RAG) for domain adaptation in LLMs. While RAG retrieves relevant documents from an external vector database at inference time to augment the prompt, it introduces substantial runtime overhead, increased latency, and scaling costs as the datastore grows. In contrast, the MemDec plug-and-play approach eliminates external retrieval by internalizing domain knowledge directly within a lightweight, pretrained transformer decoder. Once trained on a target domain, a single MemDec can be flexibly integrated into any base LLM sharing the same tokenizer with low latency and improved accuracy, as shown in the [paper's results](https://arxiv.org/pdf/2508.09874#page=9) by the reduced perplexity.

## <em>Architecture (Pipeline)</em>
<img src="./assets/project_architecture-pipeline.png?v=G-7PZXQPZPGX" width="50%">

---
<a id="main-components"></a>
## [<em>`Main Components (Features)`</em>](/assets/project_features.md)

> [!CAUTION]
> For readability, the <kbd>***Features Table***</kbd> above is shown in a condensed version. The full <u><kbd>Terminal Commands</kbd></u> for each script can be copied from here:
>
> <details><summary><kbd><strong>utils.py</strong></kbd></summary>
> 
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_utils -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1. Set Default LLM Model <em><small>(default "gemma3")</small></em></u>:<br>
> <code>python -m src.utils &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.utils qwen3.5</code><br>
> ℹ️ <code>python -m src.utils smollm3</code><br><br>
> <u>2. Show Current Default Model</u>:<br>
> ℹ️ <code>python -m src.utils --show</code>
> </details>
>
> <details><summary><kbd><strong>step0_dataset.py</strong></kbd></summary>
> 
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step0_dataset -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>Download &amp; Clean File-Based Dataset</u>:<br>
> <code>python -m src.step0_dataset &lt;dataset_name&gt;</code><br>
> ℹ️ URL: <code>python -m src.step0_dataset bverfg_decisions</code><br>
> ℹ️ Local: <code>python -m src.step0_dataset gerdalir_queries</code>
> </details>
>
> <details><summary><kbd><strong>step0_data_management.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step0_data_management -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>Combine Cleaned Datasets with Corpus Name</u>:<br>
> <code>python -m src.step0_data_management --corpus-name &lt;corpus_name&gt; &lt;dataset1&gt; &lt;dataset2&gt; &lt;datasetN&gt;</code><br>
> ℹ️<code>python -m src.step0_data_management --corpus-name test_corpus bverfg_decisions gerdalir_queries</code>
> </details>
>
> <details><summary><kbd><strong>step1_cleaning.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step1_cleaning -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. Clean HuggingFace Dataset</u>:<br>
> <code>python -m src.step1_cleaning --hf-dataset &lt;dataset_name&gt;</code><br>
> ℹ️ <code>python -m src.step1_cleaning --hf-dataset DomainLLM/german-law-qa</code><br>
> <u>1b. With Specific Config (Subsets)</u>:<br>
> <code>python -m src.step1_cleaning --hf-config &lt;config&gt;</code><br>
> ℹ️ <code>python -m src.step1_cleaning --hf-config dump-20221018-10k</code> or<br>
> <code>… --hf-config cases2022_10k</code><br>
> <u>1c. With Duplicated Config</u>:<br>
> <code>python -m src.step1_cleaning --hf-dataset &lt;dataset_name&gt; --hf-config &lt;config&gt;</code><br>
> ℹ️ <code>python -m src.step1_cleaning --hf-dataset openlegaldata/court-decisions-germany --hf-config dump-20260520-10k</code> or<br>
> <code>… --hf-dataset openlegaldata/court-decisions-germany --hf-config cases2026_10k</code><br>
> ℹ️ <code>python -m src.step1_cleaning --hf-dataset openlegaldata/laws-germany --hf-config dump-20260520-10k</code> or <br><code>… --hf-dataset openlegaldata/laws-germany --hf-config laws_germany_10k</code><br><br>
> 📌Only via <small><kbd>pipeline_config.yaml</kbd></small>:<br>
> ℹ️ <code>min_text_length</code>, <code>min_alpha_ratio</code>, <code>chunk_size</code>, <code>check_duplicates</code>
> </details>
>
> <details><summary><kbd><strong>step2_tokenization.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step2_tokenization -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. Tokenize Cleaned Dataset</u>:<br>
> <code>python -m src.step2_tokenization &lt;dataset_cleaned&gt;</code><br>
> ℹ️ <code>python -m src.step2_tokenization cases2026_10k</code><br>
> <u>1b. Override Base LLM</u>:<br>
> <code>python -m src.step2_tokenization &lt;dataset_cleaned&gt; --model &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.step2_tokenization cases2026_10k --model gemma3-1b</code><br><br>
> <u>2. Custom Params</u>:<br>
> <code>… --train-test-split &lt;ratio&gt; --num-workers &lt;n&gt;</code><br>
> ℹ️ <code>… --train-test-split 0.9 --num-workers 4</code>
> </details>
>
> <details><summary><kbd><strong>step3_pretraining.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step3_pretraining -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. Knowledge Base Creation</u>:<br>
> <code>python -m src.step3_pretraining &lt;dataset_tokenized&gt;</code><br>
> ℹ️ <code>python -m src.step3_pretraining cases2026_10k_tokenized-gemma3</code><br>
> <u>1b. Override Base LLM</u>:<br>
> <code>python -m src.step3_pretraining &lt;dataset_tokenized&gt; --model &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.step3_pretraining cases2026_10k_tokenized-gemma3 --model gemma3-1b</code><br><br>
> <u>2. Custom Params</u>:<br>
> <code>… --batch-size &lt;n&gt; --seed &lt;X&gt; --ncentroids &lt;n&gt; --num-keys-to-add-at-a-time &lt;int&gt;</code><br>
> ℹ️ <code>… --batch-size 32 --seed 777 --ncentroids 4096 --num-keys-to-add-at-a-time 1_000_000</code>
> </details>
>
> <details><summary><kbd><strong>step4_training.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step4_training -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. MemDec Training</u>:<br>
> <code>python -m src.step4_training &lt;dataset_tokenized&gt;</code><br>
> ℹ️ <code>python -m src.step4_training cases2026_10k_tokenized-gemma3</code><br>
> <u>1b. Without Unsloth <small><em>(plain HF/PyTorch)</em></small></u>:<br>
> <code>python -m src.step4_training &lt;dataset_tokenized&gt; --no-unsloth</code><br>
> ℹ️ <code>python -m src.step4_training cases2026_10k_tokenized-gemma3 --no-unsloth</code><br>
> <u>1c. Override Base LLM</u>:<br>
> <code>python -m src.step4_training &lt;dataset_tokenized&gt; --model &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.step4_training cases2026_10k_tokenized-gemma3 --model gemma3-1b</code><br>
> <u>1d. With Specific Checkpoint</u>:<br>
> <code>python -m src.step4_training &lt;dataset_tokenized&gt; --checkpoint &lt;step_X&gt;</code><br>
> ℹ️ <code>python -m src.step4_training cases2026_10k_tokenized-gemma3 --checkpoint step_500</code><br><br>
> <u>2. MemDec Params</u>:<br>
> <code>… --k-neighbors &lt;k&gt; --alpha &lt;float&gt; --lmbda &lt;float&gt;</code><br>
> ℹ️ <code>… --k-neighbors 8 --alpha 0.7 --lmbda 0.5</code><br><br>
> <u>3. Training Params</u>:<br>
> <code>… --max-steps &lt;X&gt; --checkpointing-steps &lt;interval&gt; --num-train-epochs &lt;X&gt; --batch-size &lt;n&gt; --per-device-train-batch-size &lt;n&gt; --gradient-accumulation-steps &lt;n&gt; --learning-rate &lt;rate&gt; --seed &lt;X&gt;</code><br>
> ℹ️ <code>… --max-steps 1500 --checkpointing-steps 100 --num-train-epochs 23 --batch-size 4 --per-device-train-batch-size 4 --gradient-accumulation-steps 4 --learning-rate 5e-5 --seed 777</code>
> </details>
>
> <details><summary><kbd><strong>step5_testing.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step5_testing -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. Test with Checkpoint</u>:<br>
> <code>python -m src.step5_testing --checkpoint &lt;step_X&gt;</code><br>
> ℹ️ <code>python -m src.step5_testing --checkpoint step_1000</code><br>
> <u>1b. Override Base LLM <small><em>(uses latest Checkpoint-Save)</em></small></u>:<br>
> <code>python -m src.step5_testing --model &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.step5_testing --model gemma3-1b</code><br><br>
> <u>2. MemDec Params</u>:<br>
> <code>… --knn-temp &lt;float&gt; --lmbda &lt;float&gt;</code><br>
> ℹ️ <code>… --knn-temp 0.8 --lmbda 0.5</code><br><br>
> <u>3. Generation Params</u>:<br>
> <code>… --max-new-tokens &lt;X&gt; --repetition-penalty &lt;float&gt; --do-sample --temperature &lt;float&gt; --top-p &lt;float&gt; --top-k &lt;n&gt;</code><br>
> ℹ️ <code>… --max-new-tokens 200 --repetition-penalty 1.5</code><br>
> ℹ️ <code>… --do-sample --temperature 0.8 --top-p 0.9 --top-k 40</code> (sampling instead of greedy)<br><br>
> <u>4. Custom Params</u>:<br>
> <code>… --scenarios &lt;scenario&gt; --tasks &lt;task&gt; --save-results --compare-base</code><br>
> ℹ️ <code>… --scenarios withdrawals employment --tasks question_answering summarization</code><br>
> ⚠️ <code>--save-results</code> / <code>--compare-base</code> invert the config default in <small><kbd>pipeline_config.yaml</kbd></small>, leading to <em>disabled</em> result saving & base-model comparison.
> </details>
>
> <details><summary><kbd><strong>step5_evaluation.py</strong></kbd></summary>
>
> <kbd><b>TEST</b></kbd><br>
> <code>python -m unittest test.test_step5_evaluation -v</code><br><br>
> <kbd><b>RUN</b></kbd><br>
> <u>1a. PPL Evaluation</u>:<br>
> <code>python -m src.step5_evaluation &lt;dataset_tokenized&gt; --checkpoint &lt;step_X&gt;</code><br>
> ℹ️ <code>python -m src.step5_evaluation cases2026_10k_tokenized-gemma3 --checkpoint step_1000</code><br>
> <u>1b. Override Base LLM</u>:<br>
> <code>python -m src.step5_evaluation &lt;dataset_tokenized&gt; --model &lt;model_name&gt;</code><br>
> ℹ️ <code>python -m src.step5_evaluation cases2026_10k_tokenized-gemma3 --model gemma3-1b</code><br><br>
> <u>2. MemDec Params</u>:<br>
> <code>… --lmbda &lt;float&gt; --knn-temp &lt;float&gt;</code><br>
> ℹ️ <code>… --lmbda 0.3 --knn-temp 1.0</code><br><br>
> <u>3. Evaluation Params</u>:<br>
> <code>… --split {train|validation|test} --max-examples &lt;X&gt; --batch-size &lt;n&gt; --seed &lt;X&gt;</code><br>
> ℹ️ <code>… --split test --max-examples 100 --batch-size 8 --seed 777</code><br><br>
> <u>4. Custom Params</u>:<br>
> <code>… --save-results --compare-base</code><br>
> ⚠️ <code>--save-results</code> / <code>--compare-base</code> invert the config default in <small><kbd>pipeline_config.yaml</kbd></small>, leading to <em>disabled</em> result saving & base-model comparison.
> </details>

> [!IMPORTANT]
> All CLI flags listed above are optional overrides: arguments not passed on the command line resolve to their configured values in [<kbd>pipeline_config.yaml</kbd>](config/legal_usecase/pipeline_config.yaml) and [<kbd>dataset_config.yaml</kbd>](config/legal_usecase/dataset_config.yaml), so each step can equally be invoked as `python -m src.<step>` without flags. The complete YAML schemas, parameter descriptions and instructions for defining custom domains are documented in [<kbd>config/`HowTo.md`</kbd>](config/HowTo.md).

---
<a id="project-structure"></a>
## <em>Project Structure <small>(sorted)</small></em>
```bash
MemDec-Universal/
├── test/                              # Test Scripts (Unit Tests)
├── src/                               # Source Code
│   ├── utils.py                             # Utility functions
│   ├── step0_dataset.py                     # Data Preparation & Cleaning
│   ├── step0_data_management.py             # Data Management
│   ├── step1_cleaning.py                    # HF-Dataset Preparation & Cleaning
│   ├── step2_tokenization.py                # Data Tokenization
│   ├── step3_pretraining.py                 # MemDec's Knowledge Base Creation
│   ├── step4_training.py                    # MemDec-Model Training
│   ├── step5_testing.py                     # MemDec-Model Testing
│   └── step5_evaluation.py                  # MemDec-Tests Evaluation
├── MemoryDecoder/                     # MemDec's Training Scripts (used & modified)
│   ├── knn_utils/saveEmbedMulti.py          # Step 3
│   ├── utils/cal_loss.py                    # Step 4
│   ├── train_memdec.py                      # Step 4
│   ├── train_base.py                        # Step 4
│   └── demo/memDec.py                       # Step 5 (Testing)      
├── config/                            # Config Files
│   ├── custom_usecase/                      # Custom Usecase (Template)
│   ├── legal_usecase/                       # German Legal Usecase (Project Example)
│   ├── model_config.json                    # LLM Metadata
│   └── HowTo.md                             # Config-Guide
├── dataset/                           # Datasets (cleaned & tokenized)
├── knowledge_base/                    # KNN Datastore & FAISS Index
├── outputs/                           # Output Files
│   ├── logs/                                # Script Logs (while running "stepX".py)
│   └── step_5000/                           # Trained Model Weights
├── assets/                            # README Media
├── .env.example                       # Environment Variables (Template)
├── .gitattributes                     # Git Attributes (LFS for large files)
├── .gitignore                         # Git Ignore (files to exclude from commit)
├── requirements.txt                   # Project Base Dependencies
├── THIRD-PARTY-NOTICES.txt            # 3rd-Party Software & Dataset Notices
├── LICENSE                            # Project License
└── README.md                          # This file
```

## <em>Initial Setup (Instructions)</em>
The pipeline runs on `Python 3.11` with `PyTorch (CUDA 12.8)` and `Unsloth` for efficient model loading. All pinned dependencies are listed in [<kbd>requirements.txt</kbd>](requirements.txt).

0. <kbd>***Set Environment Variables***</kbd> → [`.env.example`](./.env.example)

1. <kbd>***Create & Activate Virtual Environment***</kbd><br>
   a) <u>*Using Python*</u>:
   ```powershell
   # Python v3.11 must be pre-installed: https://www.python.org/downloads/
   py -3.11 -m venv memdec-env
   .\memdec-env\Scripts\Activate.ps1
   ```
   b) <u>*Using Conda*</u>:
   ```powershell
   # Anaconda or Miniconda (lightweight ver.) must be pre-installed: 
   # https://www.anaconda.com/docs/getting-started/installation
   conda create -n memdec-env python=3.11
   conda activate memdec-env
   ```

2. <kbd>***Install Project Dependencies***</kbd>
   ```powershell
   # 1) Project base packages
   pip install -r requirements.txt --no-cache-dir
   # 2) CUDA 12.8 toolkit libraries required by PyTorch
   pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 --no-cache-dir
   # For Conda: conda install nvidia::cuda-nvcc nvidia::cuda-runtime -c nvidia
   # 3) PyTorch with CUDA 12.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
   # 4) Unsloth last, so it patches the final PyTorch build
   pip install unsloth
   ```

> [!CAUTION]
> On Windows, the install order is critical: **pip** re-resolves shared dependencies, so installing *PyTorch* or *Unsloth* first can silently upgrade/downgrade the pinned packages from [<kbd><u>requirements.txt</u></kbd>](requirements.txt), resulting in **CUDA DLL** conflicts and memory-isolation crashes (**Hypervisor Errors**).

3. <kbd>***Authenticate with Hugging Face***</kbd><br>
   <u>*Setup Steps*</u>:
   1. Login to [*Hugging Face*](https://huggingface.co/) to **request access** on the *Open Legal Data* datasets which are used in this project: 
   [<kbd>openlegaldata/court-decisions-germany</kbd>](https://huggingface.co/datasets/openlegaldata/court-decisions-germany) and
   [<kbd>openlegaldata/laws-germany</kbd>](https://huggingface.co/datasets/openlegaldata/laws-germany).
   2. Create an **User Access Token** at [<kbd>huggingface.co/settings/`tokens`</kbd>](https://huggingface.co/settings/tokens) to get your `HF_TOKEN`.
   3. Run the following command in the terminal to authenticate:
   
   ```bash
   huggingface-cli login   # Paste your Access Token when prompted
   ```

> [!IMPORTANT]
> Authentication via `huggingface-cli login` is required to access the gated **Open Legal Data** dataset. However, as this pipeline uses **Unsloth**-optimized model weights (including Gemma 3), there is no need to accept separate model license agreements on HuggingFace.

## <em>Testing</em>
To run **all** test scripts in the [<kbd>MemDec-Universal/`test`</kbd>](./test) directory:

```bash
python -m unittest discover -s test -v
```

> [!IMPORTANT]
> To run **individual** test scripts, refer to the **Terminal Command** column in the [<kbd><u>Main Components</u></kbd>](#main-components) table above.

---
## <em>Contributing & Contact</em>
This project is a config-driven prototype for domain adaptation. While ongoing maintenance is limited, contributions via Pull Requests are highly appreciated! 🤝

Key areas for contribution include:
- <kbd>***New Use-Cases***</kbd>: Add ready-made `<your_domain>_usecase` folders with config files, data sources, and test scenarios for new domains or languages.
- <kbd>***Config Enhancements***</kbd>: Extend `dataset_config.yaml` field mappings or `pipeline_config.yaml` options to cover more source types and pipeline behaviors.
- <kbd>***Framework Robustness***</kbd>: Fix bugs, refine pipeline steps, and expand test coverage or evaluation benchmarks.

For inquiries, technical feedback, or potential collaborations, feel free to reach out via my website's [<kbd><u>contact form</u></kbd>](#contact).

## <em>Credits / Acknowledgements</em>
This is an open-source reimplementation of the research paper 
> **Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models by Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, and Zhouhan Lin (2025)** [<img src="/assets/icon-arxiv.png?v=G-7PZXQPZPGX" alt="MemoryDecoder arxiv Research Paper" width="20" style="margin-left: 4px;">](https://arxiv.org/pdf/2508.09874)[<img src="/assets/icon-pdf.png?v=G-7PZXQPZPGX" alt="MemoryDecoder Slides" width="28">](https://neurips.cc/media/neurips-2025/Slides/119458.pdf) [<img src="/assets/icon-github.png?v=G-7PZXQPZPGX" alt="MemoryDecoder GitHub Repo" width="20">](https://github.com/LUMIA-Group/MemoryDecoder)

trained on the [*Open Legal Data*](https://openlegaldata.io/) [<img src="/assets/icon-arxiv.png?v=G-7PZXQPZPGX" alt="Open Legal Data arxiv Research Paper" width="20" style="margin-left: 4px;">](https://arxiv.org/pdf/2005.13342) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Open Legal Data Hugging Face Datasets" width="20" style="margin-right: 4px;">](https://huggingface.co/openlegaldata/datasets) datasets for research, testing, and educational purposes. It is built on the open-weights foundation models [*Gemma 3*](https://deepmind.google/models/gemma/gemma-3/) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Gemma 3 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/google/gemma-3-270m-it), [*Qwen 3.5*](https://qwen.ai/blog?id=qwen3.5) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Qwen 3.5 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/Qwen/Qwen3.5-0.8B), and [*SmolLM3*](https://smollm3.org/) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="SmolLM3 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). This repository does not host or redistribute model weights or datasets. The base models provided via [*Unsloth*](https://huggingface.co/unsloth/models) [<img src="/assets/icon-github.png?v=G-7PZXQPZPGX" alt="Unsloth GitHub Repo" width="20">](https://github.com/unslothai/unsloth) and the gated Open Legal Data datasets must be accessed directly from Hugging Face, subject to their respective licenses and terms of use.

## <em>License</em>
This project is licensed under the <a href="LICENSE"><img src="https://img.shields.io/badge/Apache_2.0-darkred?style=plastic&v=G-7PZXQPZPGX" alt="Apache 2.0 (License)" title="Apache 2.0 (License)" style="position: relative; top: 2px;"></a>.

### <em>[`Third-Party Code`](THIRD-PARTY-NOTICES.txt)</em>
This repository includes and modifies code from the official Memory Decoder (MemDec) repository, vendored under [<kbd>MemDec-Universal/`MemoryDecoder`</kbd>](/MemoryDecoder/) and licensed under the <a href="MemoryDecoder/LICENSE"><img src="https://img.shields.io/badge/Apache_2.0-black?style=plastic&v=G-7PZXQPZPGX" alt="Apache 2.0 (License)" title="Apache 2.0 (License)" style="position: relative; top: 2px;"></a>. All modified files are listed in the [<kbd><u>Project Structure</u></kbd>](#project-structure) tree above, and each one carries a header notice documenting the changes made.

<footer style="margin-top: 40px;">
  <small>
    © 2026
    <a href="https://g-dea.com" target="_blank" rel="noopener noreferrer" title="Gii-DE · Personal Website">
      <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6664f3de18f429c92609f06b/CLe4qN0b0GqJdbG1Ic38a.jpeg?v=G-7PZXQPZPGX"
           alt="Own Logo" width="30" height="30" align="absmiddle"
           style="border-radius:50%; object-fit:cover; margin-left:5px;">
    </a>
  </small>
</footer>

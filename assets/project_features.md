# `MemDec-Universal` <em>Features-Table</em> <div style="text-align: right;"><a href="https://github.com/Gii-DE/MemDec-Universal" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Gii--DE-github.com%2FGii--DE%2FMemDec--Universal-darkred?labelColor=black&logo=github&style=plastic&logoColor=white&v=G-7PZXQPZPGX" title="Gii-DE · MemDec-Universal (GitHub)"></a> <a id="contact" href="https://g-dea.com/#contact" target="_blank" rel="noopener noreferrer" style="scroll-margin-top: 80px;"><img src="https://img.shields.io/badge/📧_G--DEA.COM-darkred?style=plastic&v=G-7PZXQPZPGX" alt="g-dea.com" title="Contact Me (Website)"></a></div>

<table style="width: 100%; table-layout: fixed; overflow-wrap: break-word; scrollbar-width: none; -ms-overflow-style: none;">
  <thead>
    <tr style="background-color: #dfdfdf; color: #000;">
      <th width="25%" align="left">Script</th>
      <th width="75%" align="left">Functions &amp; Purpose</th>
    </tr>
  </thead>
  <tbody style="background-color: #f6f8fa; color: #000;">
    <tr>
      <td><a href="../src/utils.py"><kbd>utils.py</kbd></a></td>
      <td>
        Shared helper module used by every pipeline step. Its most important job is picking the correct base LLM per step through a fixed priority chain: <code>CLI flag &gt; step config &gt; tokenized-dataset info &gt; checkpoint &gt; pipeline/env default</code>.<br><br>
        • Logging to <code>outputs/logs/</code> and the console<br>
        • Load the config files (<code>pipeline_config.yaml</code>, <code>dataset_config.yaml</code>) and <code>.env</code> settings<br>
        • Resolve model references (including HuggingFace model names) via <code>model_config.json</code><br>
        • Resolve checkpoints by name, path or <code>latest</code>; validate model type<br>
        • Enforce a resolvable checkpoint for <em>step5</em> runs (auto-picks newest <code>step_*</code>, aborts otherwise)<br>
        • Manage CPU/GPU device and temporary-file cleanup<br>
        • Set/show the global default model (writes <code>model_config.json</code> and <code>.env</code>)
      </td>
    </tr>
    <tr>
      <td><a href="../src/step0_dataset.py"><kbd>step0_dataset.py</kbd></a><br><small style="color:#666;"><i>(Optional)</i></small></td>
      <td>
        Downloads a file-based dataset (remote URL or local file) and cleans it for training. The dataset sources are defined in <code>dataset_config.yaml</code>. Optional: only needed for file-based corpora; Hugging Face datasets start at <em>step1</em>.<br><br>
        • Resolve the source by config key, direct URL or local path<br>
        • Download with progress bar and resume support<br>
        • Unpack <code>.gz</code> / <code>.tar.gz</code> / <code>.tgz</code> archives<br>
        • Parse <code>.json</code> / <code>.jsonl</code> / <code>.csv</code> / <code>.tsv</code> files (incl. header and split handling)<br>
        • Clean the texts with the <em>step1</em> pipeline and save to <code>dataset/&lt;name&gt;</code>
      </td>
    </tr>
    <tr>
      <td><a href="../src/step0_data_management.py"><kbd>step0_data_management.py</kbd></a><br><small style="color:#666;"><i>(Optional)</i></small></td>
      <td>
        Merges cleaned datasets into a single training corpus with index-based tracking to avoid duplicates. The merge list defaults to <code>steps.step0_data_management.datasets</code> in <code>pipeline_config.yaml</code>; the corpus name (<code>pipeline.corpus_name</code> or <code>--corpus-name</code>, required) is written to <code>dataset/&lt;corpus_name&gt;</code> next to its <code>&lt;corpus_name&gt;_index.json</code>.<br><br>
        • Combine multiple cleaned datasets and append new ones to an existing corpus<br>
        • Track merged datasets via index file (skips already-merged sets)<br>
        • Atomic corpus updates via temporary directories to prevent corrupted states<br>
        • Abort if the corpus is unreadable but the index lists merged datasets (prevents silent data loss)
      </td>
    </tr>
    <tr>
      <td><a href="../src/step1_cleaning.py"><kbd>step1_cleaning.py</kbd></a></td>
      <td>
        Downloads a dataset from Hugging Face and cleans it ready for tokenization, with the dataset and its columns defined in <code>dataset_config.yaml</code> and the quality thresholds in <code>pipeline_config.yaml</code>; the same pipeline is also integrated into <em>step0_dataset</em> to clean other types of datasets.<br><br>
        • Load any Hugging Face dataset by name and config; <code>--hf-config</code> resolves a YAML entry via its <code>config</code> or <code>output_name</code> (aborts if multiple matches are found, so add <code>--hf-dataset</code> or pass a unique <code>output_name</code>)<br>
        • Map source columns to the standard MemDec fields<br>
        • Remove HTML, links, URLs, e-mails and broken characters<br>
        • Filter out too-short, low-quality or repeated texts<br>
        • Skip duplicate texts via hashing<br>
        • Save the cleaned dataset to <code>dataset/&lt;output_name&gt;</code>
      </td>
    </tr>
    <tr>
      <td><a href="../src/step2_tokenization.py"><kbd>step2_tokenization.py</kbd></a></td>
      <td>
        Turns the cleaned texts into token sequences the chosen base LLM can process. Produces everything later steps need: an Arrow dataset for <em>step3</em> and train/test JSON files for <em>step4</em> and <em>step5_evaluation</em>, saved under <code>dataset/&lt;dataset&gt;_tokenized-&lt;model&gt;</code>.<br><br>
        • Load the cleaned dataset and the tokenizer of the chosen model<br>
        • Tokenize all texts (max 512 tokens, with attention masks and labels)<br>
        • Create a reproducible train/test split (default 80/20)<br>
        • Save Arrow and JSON formats plus tokenizer metadata for model consistency checks<br>
        • Reuse an existing tokenized dataset instead of re-tokenizing<br>
        • Parallelize over multiple workers
      </td>
    </tr>
    <tr>
      <td><a href="../src/step3_pretraining.py"><kbd>step3_pretraining.py</kbd></a></td>
      <td>
        Builds the MemDec knowledge base: the tokenized dataset is passed through the base LLM and the hidden state of every token is stored as a searchable key/value pair (KNN datastore + FAISS index in <code>knowledge_base/</code>). This is what <em>step4</em> learns to imitate, so no retrieval is needed at inference.<br><br>
        • Pass the tokenized dataset through the chosen base LLM<br>
        • Capture hidden states of the last FFN layer for every token<br>
        • Save key/value pairs as an Arrow datastore<br>
        • Build the FAISS index for fast similarity lookup<br>
        • Reuse a valid datastore/index or rebuild it if missing or broken<br>
        • Check that the model matches the dataset's tokenizer<br><br>
        <b>Modifications to <code>MemoryDecoder/knn_utils/saveEmbedMulti.py</code>:</b><br>
        • Support for more models (gemma3, qwen3.5, smollm3) with automatic dimension projection for cross-model compatibility; falls back to the text stack for multimodal checkpoints<br>
        • Only real (non-padding) tokens become datastore entries<br>
        • Lower memory use: skips unused loss computation and single-process overhead<br>
        • Faster, dimension-aware datastore writing
      </td>
    </tr>
    <tr>
      <td><a href="../src/step4_training.py"><kbd>step4_training.py</kbd></a></td>
      <td>
        Runs the actual MemDec training where the model learns to imitate the retrieval-informed behavior of the knowledge base from <em>step3</em>. All hyperparameters come from the config or CLI; checkpoints are saved to <code>outputs/step_*</code>. Based on a heavily adapted version of the paper's <code>train_memdec.py</code> that also runs on CPU and low-memory GPUs.<br><br>
        • Assemble all training arguments from config and CLI<br>
        • Validate dataset, tokenizer and model compatibility<br>
        • Resume automatically from the latest or a given checkpoint<br>
        • Configurable MemDec parameters (λ, k_neighbors, alpha, seed)<br>
        • Auto-raises the epoch count to reach <code>--max-steps</code>; <code>num_train_epochs</code> in <code>pipeline_config.yaml</code> sets a manual minimum<br>
        • Support for multiple base LLMs (gemma3, qwen3.5, smollm3)<br><br>
        <b>Modifications to <code>MemoryDecoder/train_memdec.py</code>:</b><br>
        • Fast Unsloth fine-tuning in bf16 (plain HF via <code>--no-unsloth</code>)<br>
        • 8-bit AdamW optimizer to reduce memory use<br>
        • GPU/CPU-aware batching, mixed precision and gradient checkpointing<br>
        • Memory-mapped datastore loading and efficient batching<br>
        • RAM monitoring, OOM handling and checkpoint cleanup<br>
        • Auto-derived epoch count so training always reaches <code>--max-steps</code><br><br>
        <b>Modifications to <code>MemoryDecoder/utils/cal_loss.py</code>:</b><br>
        • Chunked KL loss computation to prevent out-of-memory crashes<br>
        • Numerically stable handling of zero probabilities and shape mismatches
      </td>
    </tr>
    <tr>
      <td><a href="../src/step5_testing.py"><kbd>step5_testing.py</kbd></a></td>
      <td>
        Qualitative check of the trained model: generates answers to scenario prompts from <code>test_cases.yaml</code> with the trained MemDec model and the base model side by side, so they can be compared directly. Scenarios cover question answering, completion, summarization and content creation. Results are saved to <code>outputs/test_results/</code>.<br><br>
        • Filter and run scenario prompts by scenario and task type<br>
        • Generate answers with MemDec and base model in parallel<br>
        • Clean up repeated or broken generation output<br>
        • Score answer fidelity against the source texts (<code>sources.json</code>)<br>
        • Save timestamped JSON results with the comparison winner<br>
        • Load the newest <code>step_*</code> checkpoint from <code>outputs/</code> when none is given<br><br>
        <b>Modifications to <code>MemoryDecoder/demo/memDec.py</code>:</b><br>
        • Generation via <code>GenerationConfig</code>: greedy or multinomial sampling, repetition penalty, EOS stop<br>
        • Compatible with modern transformers versions (eager-attention fix)<br>
        • Batched left-padded prompts with per-row EOS trimming
      </td>
    </tr>
    <tr>
      <td><a href="../src/step5_evaluation.py"><kbd>step5_evaluation.py</kbd></a></td>
      <td>
        Quantitative check of the trained model: measures perplexity (PPL) score of MemDec versus the base model on the tokenized dataset and reports the improvement (ΔPPL). The joint scoring is adapted from the paper's <code>evaluate_joint.py</code>. Use the same λ as in <em>step4</em>. Results are saved to <code>outputs/test_results/</code>.<br><br>
        • Evaluate on the train or test split (<code>--split</code>)<br>
        • Score only the answer tokens (prompt and padding are masked)<br>
        • Combine base and trained-model scores via λ interpolation<br>
        • Boundary λ scores a single model (λ=0 → base only, λ=1 → trained checkpoint only)<br>
        • Load the trained checkpoint (newest <code>step_*</code> when none is given)<br>
        • Memory-friendly batched inference with <code>--max-examples</code> cap<br>
        • Report base PPL, joint PPL and ΔPPL
      </td>
    </tr>
  </tbody>
</table>

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
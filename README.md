# FGAI Assignment 3 — English-to-Spanish Neural Machine Translation

**Student:** Vidya R Nayak  
**Registration:** 230911018  
**Course:** Foundations of Generative AI (ICT4403)

## Objective
Compare four translation systems on English-to-Spanish parallel text: a basic encoder-decoder Transformer trained from scratch, fine-tuned mT5-small, and prompted decoder-only LLaMA 3.2 1B Instruct and Gemma 3 1B IT.

## Dataset
[OPUS Books, English–Spanish (Helsinki-NLP)](https://huggingface.co/datasets/Helsinki-NLP/opus_books). The notebook filters and shuffles the dataset, then selects 6,000 pairs (4,800 train, 600 validation, 600 test). All models use the same 100 held-out test examples for evaluation.

## Execution
Open `Vidya_FGAI_Assignment3.ipynb` in Google Colab, select a T4 GPU if available, and execute the cells in order. The notebook installs its dependencies (including `torch`, `transformers`, `datasets`, `sentencepiece`, `sacrebleu`, `accelerate`, `bitsandbytes`) and reads a Hugging Face token from Colab Secrets named `fgai`. Access to the gated LLaMA and Gemma repositories is required. **Never commit the token.** Model weights are not included in this repository.

## Results

| Model | BLEU | chrF | Training (s) | Inference (s) | Parameters |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.0534 | 3.0843 | 25.39 | 13.56 | 3,743,040 |
| mT5-small | 0.0785 | 3.5691 | 63.23 | 90.51 | 300,176,768 |
| LLaMA 3.2 1B Instruct | 9.1940 | 30.5297 | N/A | 70.17 | 749,275,136 |
| Gemma 3 1B IT | **13.2806** | **36.8286** | N/A | 142.09 | 651,005,056 |

Inference times are for 100 sentences and include tokenization and generation. LLaMA and Gemma were prompted rather than fine-tuned, so training time is N/A. All models used deterministic decoding; sampling-temperature analysis was not performed because it is not part of the main machine-translation evaluation question.

## Qualitative and ethical analysis
The Basic Transformer produced `<unk>` tokens; mT5 often generated `<extra_id_0>` and repetition. In one test example, LLaMA changed an expression of uncertainty into "No te preocupes", illustrating semantic drift. Evaluate bias, hallucinations, cultural references and sensitive content using bilingual review; BLEU/chrF alone do not guarantee safe translations. See the notebook's ethical analysis section.

## Repository files
- `Vidya_FGAI_Assignment3.ipynb`: executed code, evaluation and analysis
- `model_comparison.csv`, `model_comparison.png`: quantitative results and visualization
- `all_model_translations.csv`: aligned test outputs from all four models
- Individual model `*_metrics.csv` and `*_translations.csv` files: supporting results

## Conclusion
Gemma performed best by BLEU and chrF on this test subset, though inference was slowest and output quality was imperfect. Differences in model size, pretraining and adaptation limit architecture-only conclusions.

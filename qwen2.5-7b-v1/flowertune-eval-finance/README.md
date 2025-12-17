# Finance challenge Evaluation Guide (Qwen/Qwen2.5-7B)

We evaluate on Acc ([FPB](https://huggingface.co/datasets/takala/financial_phrasebank), [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)) following the Flower leaderboard rules.

## Environment setup
- Install deps: `pip install -r requirements.txt`
- Hugging Face auth: `huggingface-cli login`

## Run example
- Default: 4bit quantization is required for leaderboard (`--quantization=4`), batch size 16.
- Base model: `Qwen/Qwen2.5-7B`

```bash
python eval.py \
--base-model-name-path=Qwen/Qwen2.5-7B \ 
--peft-path=./workspace/results/<timestamp>/peft_10 \ 
--run-name=eval_qwen-2.5  \ 
--batch-size=16 \
--quantization=4 \
--datasets=fpb,fiqa,tfns
```

## Outputs
- Generations/accuracy: `benchmarks/generation_{dataset}_{category}_{run_name}.jsonl`, `benchmarks/acc_{dataset}_{category}_{run_name}.txt`
- No extra public benchmarks beyond `qwen2.5-7b-v1/flowertune-eval-finance/benchmarks`

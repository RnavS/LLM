$ErrorActionPreference = "Stop"

$python = ".\.venv312\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3.12 -m venv .venv312
}

& $python -m pip install -r requirements.txt
& $python retrieval.py build --knowledge-dir data/knowledge --output data/index/knowledge_index.pkl --chunk-words 350 --overlap-words 50
& $python train.py `
  --input cleaned_data.txt `
  --fiction-input cleaned_data.txt `
  --general-knowledge-dir data/knowledge/general `
  --medical-knowledge-dir data/knowledge/medical `
  --seed-chat-dir data/chat_seed `
  --general-weight 7 `
  --medical-weight 2 `
  --fiction-weight 1 `
  --system-preset factual-medical-lite `
  --model-preset small `
  --tokenizer-prefix data/tokenizer/smoke_local `
  --output-dir checkpoints/smoke_local `
  --knowledge-index-path data/index/knowledge_index.pkl `
  --retrieval-top-k 4 `
  --vocab-size 512 `
  --batch-size 2 `
  --max-steps 12 `
  --log-interval 1 `
  --eval-interval 6 `
  --sample-interval 6 `
  --save-interval 12 `
  --sample-max-new-tokens 48
& $python chat.py --checkpoint checkpoints/smoke_local --knowledge-index data/index/knowledge_index.pkl

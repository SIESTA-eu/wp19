#!/bin/bash
# Python venv with the project machine's versions (transformers 4.53.2), then
# pre-downloads the three models. Needs Python 3.10+ and an NVIDIA GPU.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch "transformers==4.53.2" "tokenizers==0.21.4" numpy
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('torch', torch.__version__, 'GPUs', torch.cuda.device_count())"
for m in dslim/bert-base-NER dbmdz/bert-large-cased-finetuned-conll03-english Jean-Baptiste/roberta-large-ner-english; do
  .venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification as M; AutoTokenizer.from_pretrained('$m'); M.from_pretrained('$m'); print('ok $m')"
done
echo "Setup done."

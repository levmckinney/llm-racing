# Note .env should include your API key
source .env
export CUDA_VISIBLE_DEVICES=0

python -m llm_racing time_trial --model EleutherAI/pythia-70m-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-160m-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-410m-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-1.4b-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-2.8b-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-6.9b-deduped
python -m llm_racing time_trial --model EleutherAI/pythia-12b-deduped
OPENAI_API_KEY=$OPENAI_API_KEY python -m llm_racing time_trial --model gpt-3.5-turbo
OPENAI_API_KEY=$OPENAI_API_KEY python -m llm_racing time_trial --model gpt-4

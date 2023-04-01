from tqdm import tqdm
from llm_racing.openai import OpenAIChatRacer
from llm_racing.local_deepspeed import DeepSpeedRacer
from argparse import ArgumentParser, Namespace


def get_available_models():
    """Get available models from the models directory."""
    return {
        'gpt-3.5-turbo': lambda: OpenAIChatRacer('gpt-3.5-turbo'),
        'gpt-4': lambda: OpenAIChatRacer('gpt-4'),
        'EleutherAI/pythia-70m-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-70m-deduped'),
        'EleutherAI/pythia-160m-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-160m-deduped'),
        'EleutherAI/pythia-410m-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-410m-deduped'),
        'EleutherAI/pythia-1.4b-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-1.4b-deduped'),
        'EleutherAI/pythia-2.8b-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-2.8b-deduped'),
        'EleutherAI/pythia-6.9b-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-6.9b-deduped'),
        'EleutherAI/pythia-12b-deduped': lambda: DeepSpeedRacer('EleutherAI/pythia-12b-deduped'),
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--models', nargs='*', type=str, default=['EleutherAI/pythia-70m-deduped'], help='Subset of {get_available_models().keys()} to race.'
    )
    parser.add_argument(
        '--prompts', nargs='*', type=str, default=[f'Count from 1 to {i}. \n 1, 2' for i in range(50, 550, 50)], help='A set of prompts to run against each model.'
    )
    parser.add_argument(
        '--output', type=str, default='results.csv', help='Where to save the results as a CSV with columns: tokens,time,model_name'
    )
    parser.add_argument(
        '--target_tokens', nargs='*', type=int, default=[i + 20 for i in range(50, 550, 50)], help='The target number of tokens for each prompt to elicit.'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=1024, help='The maximum number of tokens to generate for each prompt.'
    )

    racers = get_available_models()
    
    # print(args.models)
    
    args = parser.parse_args()
    df = None
    for model in tqdm(args.models):
        racer = racers[model]()
        results = racer.time_trial(args.prompts, max_tokens=args.max_tokens, target_tokens=args.target_tokens)
        df = results if df is None else df.append(results)
    df.to_csv(args.output, index=True)
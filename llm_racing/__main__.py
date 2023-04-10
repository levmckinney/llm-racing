import os
import json
import datetime
import subprocess
import pandas as pd
from llm_racing.plotting import plot_results, scatter_plots, run_regression

from llm_racing.openai import OpenAIChatRacer
from llm_racing.local_deepspeed import DeepSpeedRacer
from argparse import ArgumentParser


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


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        commit_hash = 'unknown'
    return commit_hash


def get_git_diff():
    try:
        diff = subprocess.check_output(['git', 'diff']).decode('utf-8')
    except subprocess.CalledProcessError:
        diff = 'unknown'
    return diff


def get_gpu_info():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total,memory.used,memory.free', '--format=csv']).decode('utf-8')
    except subprocess.CalledProcessError:
        gpu_info = 'unknown'
    return gpu_info


def get_cpu_info():
    try:
        cpu_info = subprocess.check_output(['lscpu']).decode('utf-8')
    except subprocess.CalledProcessError:
        cpu_info = 'unknown'
    return cpu_info


def run_time_trial(args):
    # Run a time trial for a given model.
    racers = get_available_models()
    racer = racers[args.model]()
    if args.output is None:
        output = os.path.join('results', args.model + datetime.datetime.now().isoformat())
    results = racer.time_trial(args.prompts, max_tokens=args.max_tokens, target_tokens=args.target_tokens)
    os.makedirs(output, exist_ok=True)
    results.to_csv(os.path.join(output, 'results.csv'), index=True)

    replication = {
        'args': args.__dict__,
        'commit_hash': get_git_commit_hash(),
        'diff': get_git_diff(),
        'timestamp': datetime.datetime.now().isoformat(),
        'gpu_info': get_gpu_info(),
        'cpu_info': get_cpu_info(),
    }

    with open(os.path.join(output, 'replication.json'), 'w') as f:
        json.dump(replication, f)


def run_plot(args):
    frames = []
    for results_path in args.csvs:
        frames.append(pd.read_csv(results_path))
    df = pd.concat(frames)

    regression_results = run_regression(df, args.ci)
    fig = plot_results(regression_results, args.bar_plot_legend, not args.no_bar_plot_x_ticks)
    os.makedirs(args.output, exist_ok=True)
    regression_results.to_csv(os.path.join(args.output, 'regression_results.csv'), index=True)
    fig.savefig(
        os.path.join(args.output, 'results.pdf'), 
        format="pdf", 
        bbox_inches="tight",
    )
    df = pd.concat(frames)
    fig = scatter_plots(df)
    fig.savefig(
        os.path.join(args.output, 'scatter.pdf'),
        format="pdf",
        bbox_inches="tight",
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    time_trial = subparsers.add_parser('time_trial')

    time_trial.add_argument(
        '--model', type=str, default='EleutherAI/pythia-70m-deduped', help=f'Model to evaluate must be one of {get_available_models().keys()} to race.'
    )
    time_trial.add_argument(
        '--prompts', nargs='*', type=str, default=[f'Count from 1 to {i}. \n 1, 2' for i in range(50, 550, 50)], help='A set of prompts to run against each model.'
    )
    # Default to a timestamped file in the results directory.
    time_trial.add_argument(
        '--output', type=str, default=None, help='Where to save the results as a CSV with columns: tokens,time,model_name. And replication json.'
    )
    time_trial.add_argument(
        '--target_tokens', nargs='*', type=int, default=[i + 20 for i in range(50, 550, 50)], help='The target number of tokens for each prompt to elicit.'
    )
    time_trial.add_argument(
        '--max_tokens', type=int, default=1024, help='The maximum number of tokens to generate for each prompt.'
    )
    time_trial.set_defaults(func=run_time_trial)

    plot = subparsers.add_parser('plot')
    plot.add_argument(
        '--csvs', nargs='*', type=str, default=None, help='A set of CSVs to plot.'
    )
    plot.add_argument(
        '--output', type=str, default='figures', help='Where to save the results as a PDF.'
    )
    plot.add_argument(
        '--ci', type=float, default=0.95, help='The confidence interval to use for the error bars.'
    )
    plot.add_argument(
        '--bar_plot_legend', action='store_true', default=False, help='Whether to include a legend.'
    )
    plot.add_argument(
        '--no_bar_plot_x_ticks', action='store_true', default=False, help='Whether to include x ticks.'
    )
    plot.set_defaults(func=run_plot)

    args = parser.parse_args()

    if 'func' not in args:
        parser.print_help()
        exit(1)

    args.func(args)

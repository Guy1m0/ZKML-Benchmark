import numpy as np
import re, os, argparse, sys
import concurrent.futures, subprocess, threading, psutil, time

def multi_tasks(models_list, size, start):
    for model in models_list:
        command = ["python", "benchmark.py", "--model", model, "--size", str(size), "--agg", str(start), "--save"]
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark result for a given model and testsize.",
        epilog="Example usage: python benchmark.py --size 100 --model model_name"
    )    

    parser.add_argument('--size', type=int, help='Test Size')
    parser.add_argument('--agg', type=int, help='Set the start for aggregating benchmark results')

    args = parser.parse_args()
    models_list = ['28_6_16_10_5', '14_5_11_80_10_3', '196_25_10','784_56_10']

    multi_tasks(models_list, args.size, args.agg)
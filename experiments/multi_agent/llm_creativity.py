import argparse
import sys
import os
from pathlib import Path
from compose import LLM_Review_SciFi
from types import SimpleNamespace


# This file runs the LLM Review Framework

def main():
    parser = argparse.ArgumentParser(description="Orchestrate a peer review framework with multiple AI agents.")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file for agents.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the dataset file.")
    parser.add_argument("-r", "--rounds", type=int, default=5, help="Number of rounds in the peer review.")
    parser.add_argument("-t", "--type", choices=["SciFi-Review"], help="Type of task to run.", default="SciFi-Review")
    # parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="Run in evaluation mode.")
    parser.add_argument("-p", "--prompt", type=int, default=1, help="Prompt Test")
    args = parser.parse_args()
    
    if args.type == "SciFi-Review":
        agents_config = LLM_Review_SciFi.load_config(args.config)
        discussion_runner = LLM_Review_SciFi(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    else:
        print("Invalid task type specified.")
        sys.exit(1)
        
    print('start session')
    discussion_output = discussion_runner.run()
    print('end session')

if __name__ == "__main__":
    main()

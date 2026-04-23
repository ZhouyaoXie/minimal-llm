import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Minimal-LLM Train',
                    description='Launches a training run using the input config file')
    parser.add_argument('-s', '--seed', default = 42, help = "random seed for reproducibility")
    parser.add_argument('-c', '--config', required = True, help="path to training config")

    args = parser.parse_args()
    print(f"seed: {args.seed}")
    print(f"config: {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
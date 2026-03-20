import argparse

from interactor.interactor import CowrieInteractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, required=False, default=10, help='Number of SSH sessions to run')
    parser.add_argument('--max', type=int, required=False, default=10, help='Max number of inputs per session')
    parser.add_argument('--continue_file', type=str, required=False, default='interactor/cmd_continue.json', help='State-action preference file')
    parser.add_argument('--sample_file', type=str, required=False, default='interactor/cmd_samples.json', help='Sample input file')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    interactor = CowrieInteractor()
    interactor.init_cmd_dicts(cmd_file=args.sample_file, cmd_continue_file=args.continue_file)
    interactor.run(args.num, args.max)

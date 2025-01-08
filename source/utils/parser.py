import argparse


def parse_options():
    parser = argparse.ArgumentParser(
        description='Camera pipeline optimization')

    parser.add_argument("--exp_id", type=str, default='exp_00',
                        help="experiment id")

    parser.add_argument("--trials", type=int, default=100,
                        help="number of optimization trials")

    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed")

    parser.add_argument("--data_dir", type=str, default='../data/test/',
                        help="dataset directory")

    parser.add_argument("--save_dir", type=str, default='./experiments/',
                        help="directory where to save output data and checkpoints")
    parser.add_argument('-p', '--plot',
                        action='store_true')

    return parser.parse_args()

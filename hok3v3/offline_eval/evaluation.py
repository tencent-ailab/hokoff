import sys

sys.path.append('.')
import argparse
from offline_eval.single_evaluation import evaluate as single_evaluate

parser = argparse.ArgumentParser(description='Offline 3v3 Train')

parser.add_argument("--root_path", type=str, default="/code/offline_logs", help="The root path of the offline models.")
parser.add_argument("--run_prefix", type=str, default="run_10086", help="The run prefix of the offline exp.")
parser.add_argument("--levels", type=str, default="0", help="The levels of the agents.")
# parser.add_argument("--train_step",type=int,default=0,help='trainning step')
parser.add_argument("--cpu_num", type=int, default=1, help="cpu_num")
parser.add_argument("--eval_num", type=int, default=1, help="eval_num")
parser.add_argument("--final_test", type=bool, default=0, help="eval_num")
parser.add_argument("--tensorflow_oppo", type=bool, default=1, help="use tensorflow eval")
parser.add_argument("--max_steps", type=int, default=500000, help="eval_num")
parser.add_argument("--dataset_name", type=str, default='level-0-0', help="eval_num")
args = parser.parse_args()

if __name__ == "__main__":

    root_path = args.root_path
    run_prefix = args.run_prefix
    levels = args.levels
    cpu_num = args.cpu_num
    eval_num = args.eval_num
    max_steps = args.max_steps if 'gain_gold' not in args.dataset_name else 100000

    single_evaluate(root_path, run_prefix, levels, eval_num, cpu_num, args.final_test, args.tensorflow_oppo, max_steps, args.dataset_name)

'''
Example Usage:
python main.py --expert_policy_file experts/Hopper-v1.pkl --envname Hopper-v1 --train --test --epochs 20 --num-rollouts 20 --out data.pkl

Testing with rendering:
python main.py --expert_policy_file experts/Hopper-v1.pkl --envname Hopper-v1 --test --render --model models/model.pth

Workflow:

* DONE: get env options by input
* DONE: data collection by running expert
* DONE: Process data, transform
* DONE: Process data, split
* DONE: import model
* DONE: train model
* DONE: test model

'''
from __future__ import print_function


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument('--input', type=str, help='expert data file to use')
    parser.add_argument('--out', type=str, help='save expert data to file')    
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, help='File to load model from')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    if args.input is None and args.model is None:
        from policy.run import run_expert
        run_expert(args) # saves data as str(args.out)
    else:
        args.out = args.input

    from policy.train import train_policy
    if args.train and args.out:
        train_policy(args)

    if args.test:
        # if args.model is None:
        import data_util
        args.model = data_util.get_latest('models/*')

        from policy.evaluate import run_policy
        run_policy(args)

if __name__ == '__main__':
    main()
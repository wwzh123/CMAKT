import argparse
import json
import time

import torch
from numpy.distutils.fcompiler import str2bool

from data_process import *
from train import train


def parse_args():
    arg_parser = argparse.ArgumentParser(description="train dkt model")

    # model
    arg_parser.add_argument('--model', type=str, default='dkt')
    arg_parser.add_argument('--model_name', type=str, default='GIKT')

    # path
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument('--cache_dir', type=str, default='cache')
    arg_parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    arg_parser.add_argument("--log_dir", type=str, default='logs')

    # Dataset
    arg_parser.add_argument('--dataset', type=str, default='assist09_3')
    arg_parser.add_argument("--field_size", type=int, default=3)
    arg_parser.add_argument("--select_index", type=int, default=[0, 1, 2])

    # hyper-parameter
    arg_parser.add_argument("--batch_size", type=int, default=2048)
    arg_parser.add_argument('--num_epochs', type=int, default=10)
    arg_parser.add_argument('--train', type=str2bool, default='true')
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--lr_decay", type=float, default=0.92)
    arg_parser.add_argument('--l2_weight', type=float, default=1e-8)
    arg_parser.add_argument('--dropout_keep_probs', nargs='?', default='[0.6, 0.8, 1]', type=str)

    # LSTM
    arg_parser.add_argument("--max_step", type=int, default=10)
    arg_parser.add_argument("--embedding_size", type=int, default=100)
    arg_parser.add_argument('--hidden_neurons', type=int, default=[200, 100])
    arg_parser.add_argument('--limit_max_len', type=int, default=200)
    arg_parser.add_argument('--limit_min_len', type=int, default=3)


    arg_parser.add_argument('--n_hop', type=int, default=0)
    arg_parser.add_argument('--aggregator', type=str, default='sum')
    arg_parser.add_argument('--skill_neighbor_num', type=int, default=10)
    arg_parser.add_argument('--question_neighbor_num', type=int, default=4)
    arg_parser.add_argument('--hist_neighbor_num', type=int, default=3, help='ei num in information interaction')
    arg_parser.add_argument('--next_neighbor_num', type=int, default=0, help='si num in information interaction')

    # misc
    arg_parser.add_argument('--sim_emb', type=str, default='skill_emb')
    arg_parser.add_argument('--att_bound', type=float, default=0.5)


    arg_parser.add_argument('--MAX_SEQ', type=int, default=9)
    arg_parser.add_argument('--EMBED_DIMS', type=int, default=100)
    arg_parser.add_argument('--ENC_HEADS', type=int, default=4)
    arg_parser.add_argument('--DEC_HEADS', type=int, default=4)
    arg_parser.add_argument('--NUM_ENCODER', type=int, default=4)
    arg_parser.add_argument('--NUM_DECODER', type=int, default=4)

    return arg_parser.parse_args()


def save_model_dir(args):
    return '{}_{}_{}lr_{}hop_{}sn_{}qn_{}hn_{}nn_{}_{}bound_{}keep_{}'.format(args.dataset,
                                                                              args.model,
                                                                              args.lr,
                                                                              args.n_hop,
                                                                              args.skill_neighbor_num,
                                                                              args.question_neighbor_num,
                                                                              args.hist_neighbor_num,
                                                                              args.next_neighbor_num,
                                                                              args.sim_emb,
                                                                              args.att_bound,
                                                                              args.dropout_keep_probs,
                                                                              args.tag)


def save_config(args):
    config = {}
    for k, v in vars(args).items():
        config[k] = vars(args)[k]
    jsObj = json.dumps(config, indent=4, sort_keys=True)
    config_name = 'logs/%f_config.json' % args.tag
    with open(config_name, 'w') as f:
        f.write(jsObj)


def save_tag(args):
    tag_path = os.path.join("%s/%s_tag.txt" % (args.data_dir, args.dataset))
    with open(tag_path, 'w') as f:
        f.write(str(args.tag))


def main():
    args = parse_args()
    args.tag = time.time()
    args.cache_dir = os.path.join(args.cache_dir, args.dataset)
    args.model_dir = save_model_dir(args)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_dir)
    args.checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name + '_checkpoint.pt')
    print('args: %s' % args)
    print('algorithm: %s' % args.model)

    save_config(args)

    args = data_process(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: %s' % args.device)

    train(args, True)

    save_tag(args)

    num_students = 1
    num_questions = args.max_step
    feature_answer_index = np.random.randint(0, args.feature_answer_size, (num_students, num_questions + 1, 3))
    hist_neighbor_index = np.random.randint(0, args.feature_answer_size, (num_students, num_questions, args.max_step))


if __name__ == "__main__":
    main()

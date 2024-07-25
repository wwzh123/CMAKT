import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from data_process import DataGenerator
from model import CMAKT


def train(args, isTrain):
    model = CMAKT(args).to(device=args.device)
    print("model: %s" % model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_valid_auc = 0
    for epoch in tqdm(range(args.num_epochs)):
        print(f"===epoch:%03d===" % epoch)

        # train
        train_impl(args, epoch, model, optimizer, criterion)

        # valid
        auc_value = valid_impl(args, epoch, model, optimizer)
        if auc_value > best_valid_auc:
            print('%3.4f to %3.4f' % (best_valid_auc, auc_value))
            best_valid_auc = auc_value
            best_epoch = epoch

            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)

            torch.save({
                'global_step': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.checkpoint_path)
            print('Save checkpoint at %d' % best_epoch)
        print(args.model_dir + "\t" + str(best_valid_auc))

    try:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('CKPT loaded')
    except Exception as e:
        print(f'load checkpoint failed, {e}')
        return

    test_impl(args, model)


def train_impl(args, epoch, model, optimizer, criterion):
    model.train()

    train_generator = DataGenerator(args.train_seqs, args.max_step, batch_size=args.batch_size,
                                    feature_size=args.feature_answer_size - 2,
                                    hist_num=args.hist_neighbor_num)
    overall_loss = 0
    train_generator.shuffle()
    preds, binary_preds, targets = list(), list(), list()
    train_step = 0
    while not train_generator.end:
        train_step += 1

        [skill_quest_answer, target_answers, seq_lens, hist_neighbor_index] = train_generator.next_batch()
        binary_pred, pred = model(skill_quest_answer, hist_neighbor_index)

        target_answers = torch.from_numpy(target_answers).float().to(args.device)

        loss = criterion(pred, target_answers)
        overall_loss += loss
        for seq_idx, seq_len in enumerate(seq_lens):
            preds.append(pred[seq_idx, 0:seq_len])
            binary_preds.append(binary_pred[seq_idx, 0:seq_len])
            targets.append(target_answers[seq_idx, 0:seq_len])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # compute metrics
    train_loss = overall_loss / train_step

    preds = torch.cat(preds).detach().cpu().numpy()
    binary_preds = torch.cat(binary_preds).detach().cpu().numpy()
    targets = torch.cat(targets).detach().cpu().numpy()

    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)

    print("train loss = {0}, auc={1}, accuracy={2}".format(train_loss, auc_value, accuracy))
    write_log(args, args.model_dir, auc_value, accuracy, epoch, name='train_')


@torch.no_grad()
def valid_impl(args, epoch, model, optimizer):
    model.eval()

    valid_generator = DataGenerator(args.valid_seqs, args.max_step, batch_size=args.batch_size,
                                    feature_size=args.feature_answer_size - 2,
                                    hist_num=args.hist_neighbor_num)
    # valid
    valid_generator.reset()
    preds, binary_preds, targets = list(), list(), list()
    valid_step = 0
    # overall_loss = 0
    while not valid_generator.end:
        valid_step += 1
        [skill_quest_answer, target_answers, seq_lens, hist_neighbor_index] = valid_generator.next_batch()
        binary_pred, pred = model(skill_quest_answer, hist_neighbor_index)

        # overall_loss += loss
        for seq_idx, seq_len in enumerate(seq_lens):
            preds.append(pred[seq_idx, 0:seq_len])
            binary_preds.append(binary_pred[seq_idx, 0:seq_len])
            targets.append(target_answers[seq_idx, 0:seq_len])

    # compute metrics
    # valid_loss = overall_loss / valid_step

    preds = torch.cat(preds).detach().cpu().numpy()
    binary_preds = torch.cat(binary_preds).detach().cpu().numpy()
    targets = np.array(np.concatenate(targets, -1), dtype=np.int64)

    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)

    print("valid auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))
    write_log(args, args.model_dir, auc_value, accuracy, epoch, name='valid_')

    return auc_value


@torch.no_grad()
def test_impl(args, model):
    model.eval()

    test_data_generator = DataGenerator(args.test_seqs, args.max_step, batch_size=args.batch_size,
                                        feature_size=args.feature_answer_size - 2,
                                        hist_num=args.hist_neighbor_num)
    test_data_generator.reset()
    preds, binary_preds, targets = list(), list(), list()
    i = 0
    while not test_data_generator.end:
        [skill_quest_answer, target_answers, seq_lens, hist_neighbor_index] = test_data_generator.next_batch()
        binary_pred, pred = model(skill_quest_answer, hist_neighbor_index)

        # if i == 0:
        #     utils.draw_heatmap(pred.detach().cpu().numpy(), 1)
        #     i += 1

        # overall_loss += loss
        for seq_idx, seq_len in enumerate(seq_lens):
            preds.append(pred[seq_idx, 0:seq_len])
            binary_preds.append(binary_pred[seq_idx, 0:seq_len])
            targets.append(target_answers[seq_idx, 0:seq_len])

    # compute metrics
    # test_loss = overall_loss / test_step

    preds = torch.cat(preds).detach().cpu().numpy()
    binary_preds = torch.cat(binary_preds).detach().cpu().numpy()
    targets = np.array(np.concatenate(targets, -1), dtype=np.int64)

    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)

    print("test auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))
    write_log(args, args.model_dir, auc_value, accuracy, 1, name='test_')


def write_log(args, model_dir, auc, accuracy, epoch, name='train_'):
    log_path = os.path.join(args.log_dir, name + model_dir + '.csv')
    if not os.path.exists(log_path):
        log_file = open(log_path, 'w')
        log_file.write('Epoch\tAuc\tAccuracy\n')
    else:
        log_file = open(log_path, 'a')

    log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\n')
    log_file.flush()

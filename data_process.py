import os

import numpy as np


def data_process(args):
    save_file = os.path.join(args.cache_dir, format_save_file(args))
    if os.path.exists(save_file):
        return data_process_loading(args, save_file)
    else:
        if not os.path.exists(args.cache_dir):
            os.mkdir(args.cache_dir)
        return data_process_initial(args, save_file)


def format_save_file(args):
    return '{}_fs{}_ms{}_sn{}_qn{}.npy'.format(args.dataset,
                                               args.field_size,
                                               args.max_step,
                                               args.skill_neighbor_num,
                                               args.question_neighbor_num)


def data_process_loading(args, save_file):
    [args.train_seqs, args.test_seqs, args.valid_seqs, args.skill_num, args.question_num, args.feature_answer_size,
     args.skill_matrix, args.question_neighbors, args.skill_neighbors] = np.load(save_file, allow_pickle=True)
    return args


def data_process_initial(args, save_file):
    train_data_file = os.path.join(args.data_dir, args.dataset, args.dataset + '_train.csv')
    args.train_seqs, train_max_skill_id, train_max_question_id, feature_answer_id = load_data(train_data_file,
                                                                                              args.field_size,
                                                                                              args.max_step)
    print("original train seqs num:%d" % len(args.train_seqs))

    test_data_file = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')
    args.test_seqs, test_max_skill_id, test_max_question_id, _ = load_data(test_data_file, args.field_size,
                                                                           args.max_step)
    print("original test seqs num:%d" % len(args.test_seqs))

    valid_data_file = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')
    args.valid_seqs, _, _, _ = load_data(valid_data_file, args.field_size, args.max_step)
    print("original valid seqs num:%d" % len(args.valid_seqs))

    # (技能)数量 = 最大技能id + 1 // 训练 + 测试
    args.skill_num = max(train_max_skill_id, test_max_skill_id) + 1
    print('skill num: %s' % args.skill_num)

    # (技能 + 问题)数量 = 最大问题id + 1 // 训练 + 测试
    skill_quest_num = max(train_max_question_id, test_max_question_id) + 1

    # (问题)数量 = (技能 + 问题)数量 - (技能)数量 // 训练 + 测试
    args.question_num = skill_quest_num - args.skill_num
    print('quest num: %s' % args.question_num)

    # (技能 + 问题 + 答案)的数量 = 最大答案id + 1 //训练 + 测试
    args.feature_answer_size = feature_answer_id + 1

    # 构建技能矩阵的路径
    # multi-skill， # 加载技能矩阵
    skill_matrix_file = os.path.join(args.data_dir, args.dataset, args.dataset + '_skill_matrix.txt')
    args.skill_matrix = np.loadtxt(skill_matrix_file)

    # 构建邻接表和交互列表
    skill_question_adj = build_skill_question_adj(args.train_seqs, args.test_seqs, args.skill_matrix, skill_quest_num)

    # 提取问题和技能的关系
    args.question_neighbors, args.skill_neighbors = extract_qs_relations(skill_question_adj, args.skill_num,
                                                                         skill_quest_num, args.question_neighbor_num,
                                                                         args.skill_neighbor_num)

    # save processed file
    save_data = [args.train_seqs, args.test_seqs, args.valid_seqs, args.skill_num, args.question_num,
                 args.feature_answer_size, args.skill_matrix, args.question_neighbors, args.skill_neighbors]
    np.save(save_file, save_data)
    return args


def build_skill_question_adj(train_seqs, test_seqs, skill_matrix, skill_quest_num):
    skill_question_adj = [[] for _ in range(skill_quest_num)]
    for seqs in [train_seqs, test_seqs]:
        for seq in seqs:
            for [skill, question, answer] in seq:
                skill_neighbors = np.reshape(np.argwhere(skill_matrix[skill] == 1), [-1]).tolist()
                skill_question_adj[question] = skill_neighbors
                for skill_neighbor in skill_neighbors:
                    if question not in skill_question_adj[skill_neighbor]:
                        skill_question_adj[skill_neighbor].append(question)

    return skill_question_adj


def extract_qs_relations(skill_question_adj, skill_num, skill_quest_num, question_neighbor_num, skill_neighbor_num):
    # 创建一个大小为skill_quest_num行，question_neighbor_num列的整型数组，初始值为0
    question_neighbors = np.zeros([skill_quest_num, question_neighbor_num], dtype=np.int32)

    # 创建一个大小为skill_num行，skill_neighbor_num列的整型数组，初始值为0
    skill_neighbors = np.zeros([skill_num, skill_neighbor_num], dtype=np.int32)

    # 遍历skill_question_adj的索引和邻居列表
    for index, neighbors in enumerate(skill_question_adj):
        if len(neighbors) > 0:
            if index < skill_num:  # 技能
                if len(neighbors) >= skill_neighbor_num:
                    skill_neighbors[index] = np.random.choice(neighbors, skill_neighbor_num, replace=False)
                else:
                    skill_neighbors[index] = np.random.choice(neighbors, skill_neighbor_num, replace=True)
            else:  # 问题
                if len(neighbors) >= question_neighbor_num:
                    question_neighbors[index] = np.random.choice(neighbors, question_neighbor_num, replace=False)
                else:
                    question_neighbors[index] = np.random.choice(neighbors, question_neighbor_num, replace=True)

    return question_neighbors, skill_neighbors


def load_data(dataset_path, field_size, max_seq_len):
    seqs = []
    max_skill = -1
    max_question = -1
    feature_answer_size = -1
    with open(dataset_path, 'r') as f:
        feature_answer_list = []
        for lineid, line in enumerate(f):
            fields = line.strip().strip(',')
            i = lineid % (field_size + 1)
            if i != 0:  # i==0 new student==>student seq len
                feature_answer_list.append(list(map(int, fields.split(","))))
            if i == 1:  # 技能
                if max(feature_answer_list[-1]) > max_skill:
                    max_skill = max(feature_answer_list[-1])
            elif i == 2:  # 问题
                if max(feature_answer_list[-1]) > max_question:
                    max_question = max(feature_answer_list[-1])
            elif i == field_size:  # 答案
                if max(feature_answer_list[-1]) > feature_answer_size:
                    feature_answer_size = max(feature_answer_list[-1])
                if len(feature_answer_list[0]) > max_seq_len:
                    n_split = len(feature_answer_list[0]) // max_seq_len
                    if len(feature_answer_list[0]) % max_seq_len:
                        n_split += 1
                else:
                    n_split = 1
                for k in range(n_split):
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:
                        end_index = len(feature_answer_list[0])
                    else:
                        end_index = (k + 1) * max_seq_len
                    split_list = []

                    for i in range(len(feature_answer_list)):
                        split_list.append(feature_answer_list[i][k * max_seq_len:end_index])
                        # if i == len(feature_answer_list)-2:#before answer
                        # split_list.append([student_id]*(end_index-k*args.seq_len)) #student id

                    split_list = np.stack(split_list, 1).tolist()  # [seq_len,field_size]

                    seqs.append(split_list)
                feature_answer_list = []

    return seqs, max_skill, max_question, feature_answer_size


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    # print(np.shape(sequences))
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    # [batch_size, max_step, field_size]
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':  # maxlen!=none may need to truncating
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen + 1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# select same skill index
def sample_hist_neighbors(seqs_size, max_step, hist_num, skill_index):
    # skill_index:[batch_size,max_step]

    # [batch_size,max_step,M]
    hist_neighbors_index = []

    for i in range(seqs_size):
        seq_hist_index = []
        seq_skill_index = skill_index[i]
        # [max_step,M]
        for j in range(1, max_step):
            same_skill_index = [k for k in range(j) if seq_skill_index[k] == seq_skill_index[j]]

            if hist_num != 0:
                # [0,j] select M
                if len(same_skill_index) >= hist_num:
                    seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=False))
                else:
                    if len(same_skill_index) != 0:
                        seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=True))
                    else:
                        seq_hist_index.append(([max_step - 1 for _ in range(hist_num)]))
            else:
                seq_hist_index.append([])
        hist_neighbors_index.append(seq_hist_index)
    return hist_neighbors_index


def format_data(seqs, max_step, feature_size, hist_num):
    # [batch_size]
    seq_lens = np.array(list(map(lambda seq: len(seq), seqs)))

    # [batch_size,max_len,feature_size]
    features_answer_index = pad_sequences(seqs, maxlen=max_step, padding='post', value=0)

    target_answers = np.array([[j[-1] - feature_size for j in i[1:]] for i in seqs])
    target_answers = pad_sequences(target_answers, maxlen=max_step - 1, padding='post', value=0)

    skills_index = features_answer_index[:, :, 0]

    # [batch_size,max_step,M]
    hist_neighbor_index = sample_hist_neighbors(len(seqs), max_step, hist_num, skills_index)

    return features_answer_index, target_answers, seq_lens, hist_neighbor_index


class DataGenerator(object):

    def __init__(self, seqs, max_step, batch_size, feature_size, hist_num):  # feature_dkt
        np.random.seed(42)
        self.seqs = seqs
        self.max_step = max_step
        self.batch_size = batch_size
        self.batch_i = 0
        self.end = False
        self.feature_size = feature_size
        self.n_batch = int(np.ceil(len(seqs) / batch_size))
        self.hist_num = hist_num

    def next_batch(self):
        batch_seqs = self.seqs[self.batch_i * self.batch_size:(self.batch_i + 1) * self.batch_size]
        self.batch_i += 1

        if self.batch_i == self.n_batch:
            self.end = True

        # [feature_index,target_answers,sequences_lens,hist_neighbor_index]
        format_data_list = format_data(batch_seqs, self.max_step, self.feature_size, self.hist_num)
        return format_data_list

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggregators import SumAggregator, ConcatAggregator


class FFN(nn.Module):

    def __init__(self, in_feat):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_feat, in_feat)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class EncoderEmbedding(nn.Module):

    def __init__(self, seq_len, n_dims, device):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.device = device

        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, exercises, categories):
        seq = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        position = self.position_embed(seq)
        return exercises + categories + position


class DecoderEmbedding(nn.Module):

    def __init__(self, seq_len, n_dims, device):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.device = device

        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, responses):
        seq = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        position = self.position_embed(seq)
        return responses + position


class StackedNMultiHeadAttention(nn.Module):

    def __init__(self, n_stacks, n_dims, n_heads, seq_len, device, n_multi_head=1, dropout=0.0):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.device = device
        self.n_multi_head = n_multi_head
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)

        # n_stacks has n_multi_heads each
        self.multi_head_layers = nn.ModuleList(n_stacks * [nn.ModuleList(
            n_multi_head * [nn.MultiheadAttention(embed_dim=n_dims, num_heads=n_heads, dropout=dropout), ]), ])

        # FFN
        self.ffn = nn.ModuleList(n_stacks * [FFN(n_dims)])

        # 生成上三角矩阵
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            for multi_head in range(self.n_multi_head):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, _ = self.multi_head_layers[stack][multi_head](query=norm_q.permute(1, 0, 2),
                                                                            key=norm_k.permute(1, 0, 2),
                                                                            value=norm_v.permute(1, 0, 2),
                                                                            attn_mask=self.mask.to(self.device))
                heads_output = heads_output.permute(1, 0, 2)
                # assert encoder_output != None and break_layer is not None
                if encoder_output is not None and multi_head == break_layer:
                    assert break_layer <= multi_head, " break layer should be less than multi_head layers and postive integer"
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output


class SaintPlus(nn.Module):

    def __init__(self, args):
        super(SaintPlus, self).__init__()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=args.NUM_DECODER,
                                                        n_dims=args.EMBED_DIMS,
                                                        n_heads=args.DEC_HEADS,
                                                        seq_len=args.MAX_SEQ,
                                                        device=args.device,
                                                        n_multi_head=1, dropout=0.0)

        self.decoder_layer = StackedNMultiHeadAttention(n_stacks=args.NUM_ENCODER,
                                                        n_dims=args.EMBED_DIMS,
                                                        n_heads=args.ENC_HEADS,
                                                        seq_len=args.MAX_SEQ,
                                                        device=args.device,
                                                        n_multi_head=2, dropout=0.0)

        self.encoder_embedding = EncoderEmbedding(seq_len=args.MAX_SEQ,
                                                  n_dims=args.EMBED_DIMS,
                                                  device=args.device)

        self.decoder_embedding = DecoderEmbedding(seq_len=args.MAX_SEQ,
                                                  n_dims=args.EMBED_DIMS,
                                                  device=args.device)

        self.fc = nn.Linear(args.EMBED_DIMS, 1)

    def forward(self, params):
        enc = self.encoder_embedding(exercises=params['input_questions_embedding'],
                                     categories=params['input_skills_embedding'])

        encoder_output = self.encoder_layer(input_k=enc, input_q=enc, input_v=enc)

        dec = self.decoder_embedding(responses=params['input_answers_embedding'])

        decoder_output = self.decoder_layer(input_k=dec, input_q=dec, input_v=dec,
                                            encoder_output=encoder_output, break_layer=1)

        # fully connected layer
        out = self.fc(decoder_output)
        return out.squeeze()


# ----------------------------------------------------------------------------------------------------------------------


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, keep_prob):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=1 - keep_prob)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out)


class MultiLayerLSTM(nn.Module):
    def __init__(self, hidden_neurons, input_size, keep_prob):
        super(MultiLayerLSTM, self).__init__()
        self.lstm = nn.ModuleList()
        for hidden_size in hidden_neurons:
            self.lstm.append(LSTM(input_size, hidden_size, keep_prob))
            input_size = hidden_size

    def forward(self, x):
        for hidden_layer in self.lstm:
            x = hidden_layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_size, 1)))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class GIKT(nn.Module):
    def __init__(self, args):
        super(GIKT, self).__init__()

        self.args = args
        self.device = args.device

        # LSTM时间步数
        self.lr = args.lr
        self.n_hop = args.n_hop
        self.max_step = args.max_step - 1
        self.select_index = args.select_index

        self.feature_answer_size = args.feature_answer_size
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size

        self.hidden_neurons = args.hidden_neurons
        self.hidden_size = self.hidden_neurons[-1]

        self.dropout_keep_probs = eval(args.dropout_keep_probs)
        self.keep_prob = self.dropout_keep_probs[0]
        self.keep_prob_gnn = self.dropout_keep_probs[1]

        self.hist_neighbor_num = args.hist_neighbor_num  # ei
        self.next_neighbor_num = args.next_neighbor_num  # si

        self.question_neighbor_num = args.question_neighbor_num
        self.question_neighbors = torch.from_numpy(args.question_neighbors).long().to(self.device)

        self.skill_neighbor_num = args.skill_neighbor_num
        self.skill_neighbors = torch.from_numpy(args.skill_neighbors).long().to(self.device)

        self.input_feature_linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.next_feature_linear = nn.Linear(self.embedding_size, self.hidden_size)

        # lstm
        self.lstm_input_embedding = nn.Linear(self.embedding_size * 3, self.hidden_size)
        self.lstm = MultiLayerLSTM(self.hidden_neurons, self.embedding_size, self.keep_prob)

        # 交互模块
        self.attention1 = Attention(self.hidden_size)
        self.attention2 = Attention(self.hidden_size)

    def forward(self, params):
        """
        :param params: parameters
        :return:
        """
        self.questions_index = params['questions_index']
        self.next_questions_index = params['next_questions_index']

        self.input_skills_embedding = params['input_skills_embedding']
        self.next_skills_embedding = params['next_skills_embedding']
        self.input_questions_embedding = params['input_questions_embedding']
        self.next_questions_embedding = params['next_questions_embedding']
        self.input_answers_embedding = params['input_answers_embedding']

        self.batch_size = params['batch_size']
        self.feature_embedding = params['feature_embedding']
        self.hist_neighbor_index = params['hist_neighbor_index']

        # todo 将原输入转化为嵌入向量
        if self.n_hop > 0:
            if self.args.aggregator == 'sum':
                self.aggregator_class = SumAggregator
            elif self.args.aggregator == 'concat':
                self.aggregator_class = ConcatAggregator
            else:
                raise Exception("Unknown aggregator: " + self.args.aggregator)

            # [[batch_size, max_step],[batch_size, max_step, question_neighbor_num] * n_hop]
            input_neighbors = self.get_neighbors(self.n_hop, self.questions_index)

            aggregate_embedding, self.aggregators = self.aggregate(input_neighbors, self.input_questions_embedding)

            next_input_neighbors = self.get_neighbors(self.n_hop, self.next_questions_index)

            next_aggregate_embedding, self.aggregators = self.aggregate(next_input_neighbors,
                                                                        self.next_questions_embedding)

            # [batch_size, max_step, hidden_size]
            feature_trans_embedding = self.linear_relu(aggregate_embedding[0], self.input_feature_linear)

            # [batch_size, max_step, hidden_size]
            next_trans_embedding = self.linear_relu(next_aggregate_embedding[0], self.next_feature_linear)
        else:
            # [batch_size, max_step, hidden_size]
            feature_trans_embedding = self.linear_relu(self.input_questions_embedding, self.input_feature_linear)

            # [batch_size, max_step, hidden_size]
            next_trans_embedding = self.linear_relu(self.next_questions_embedding, self.next_feature_linear)

            # [[batch_size, max_len],[batch_size, max_len, question_neighbor_num]]
            input_neighbors = self.get_neighbors(1, self.questions_index)

            # [[batch_size, max_step, hidden_size], [batch_size, max_step, -1, embedding_size]]
            neighbor_embedding = self.feature_embedding[input_neighbors[-1], :]
            neighbor_embedding = neighbor_embedding.view([self.batch_size, self.max_step, -1, self.embedding_size])
            aggregate_embedding = [feature_trans_embedding, neighbor_embedding]

            # [[batch_size, max_len],[batch_size, max_len, question_neighbor_num]]
            next_input_neighbors = self.get_neighbors(1, self.next_questions_index)

            # [[batch_size, max_step, hidden_size], [batch_size, max_step, -1, embedding_size]]
            next_neighbor_embedding = self.feature_embedding[next_input_neighbors[-1], :]
            next_neighbor_embedding = next_neighbor_embedding.view(
                [self.batch_size, self.max_step, -1, self.embedding_size])
            next_aggregate_embedding = [next_trans_embedding, next_neighbor_embedding]

        # todo LSTM
        # [batch_size * max_step, embedding_size * 3]
        input_fa_embedding = torch.cat([
            self.input_questions_embedding,
            self.input_answers_embedding,
            self.input_skills_embedding], dim=-1)
        input_fa_embedding = input_fa_embedding.view(-1, self.max_step, self.embedding_size * 3)

        # [batch_size, max_step, hidden_size]
        input_trans_embedding = self.lstm_input_embedding(input_fa_embedding)
        input_trans_embedding = input_trans_embedding.view(-1, self.max_step, self.hidden_size)

        # [batch_size, max_step, hidden_size]
        output_series = self.lstm(input_trans_embedding)

        # todo recap模块
        if self.args.model == "hssi":  # hard selection, use hi  (HS)
            # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            self.hist_neighbors_features = self.hist_neighbor_sampler_hard(output_series)
        elif self.args.model == "hsei":  # hard selection, use ei  (HE)
            # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            self.hist_neighbors_features = self.hist_neighbor_sampler_hard(input_trans_embedding)
        elif self.args.model == "ssei":  # soft selection, use si/qi, ei
            if self.args.sim_emb == "skill_emb":
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(self.input_skills_embedding,
                                                                               self.next_skills_embedding,
                                                                               input_trans_embedding)
            elif self.args.sim_emb == "question_emb":
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(self.input_questions_embedding,
                                                                               self.next_questions_embedding,
                                                                               input_trans_embedding)
            else:
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(feature_trans_embedding,
                                                                               next_trans_embedding,
                                                                               input_trans_embedding)
        else:  # soft selection, use si/qi, ht
            if self.args.sim_emb == "skill_emb":
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(self.input_skills_embedding,
                                                                               self.next_skills_embedding,
                                                                               output_series)
            elif self.args.sim_emb == "question_emb":
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(self.input_questions_embedding,
                                                                               self.next_questions_embedding,
                                                                               output_series)
            else:
                # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
                self.hist_neighbors_features = self.hist_neighbor_sampler_soft(feature_trans_embedding,
                                                                               next_trans_embedding,
                                                                               output_series)
        # todo <qt, si>
        if self.next_neighbor_num != 0:
            # [batch_size, max_step, N+1, embedding_size]
            Nn = self.next_neighbor_sampler(next_aggregate_embedding)
            Nn = torch.cat([torch.unsqueeze(next_trans_embedding, 2), Nn], -2)
            next_neighbor_num = self.next_neighbor_num + 1
        else:
            # [batch_size, max_step, N+1, embedding_size]
            Nn = torch.unsqueeze(next_trans_embedding, 2)
            next_neighbor_num = 1

        # todo <ht, ei>
        if self.hist_neighbor_num != 0:
            # [batch_size, max_step, M+1, embedding_size]
            Nh = torch.cat([torch.unsqueeze(output_series, 2), self.hist_neighbors_features], 2)
        else:
            # [batch_size, max_step, M+1, embedding_size]
            Nh = torch.unsqueeze(output_series, 2)

        # todo g(fi, fj)
        # [-1, max_step, Nh, 1, embedding_size] * [-1, max_step, 1, Nn, embedding_size]
        logits = torch.sum(torch.unsqueeze(Nh, 3) * torch.unsqueeze(Nn, 2), dim=4)
        # [batch_size, max_step, Nu*Nv]
        logits = logits.view(-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num)

        # todo a(fi, fj)
        # [batch_size * max_step * (hn + 1)]
        # [batch_size, max_step, hn + 1, 1]
        f1 = self.attention1(Nh.view(-1, self.hidden_size))
        f1 = f1.view(-1, self.max_step, self.hist_neighbor_num + 1, 1)

        # [batch_size * max_step * nn]
        # [batch_size, max_step, 1, nn]
        f2 = self.attention2(Nn.contiguous().view(-1, self.hidden_size))
        f2 = f2.view(-1, self.max_step, 1, next_neighbor_num)

        # [batch_size, max_step, nn *(hn + 1)]
        f1_f2 = (f1 + f2).view(-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num)
        coefs = F.softmax(torch.tanh(f1_f2), dim=2)

        # todo pt
        # [batch_size, max_step]
        logits = torch.sum(logits * coefs, dim=-1)

        return logits

    def linear_relu(self, feature_embedding, linear):
        """
        :param feature_embedding: [batch_size, max_step, hidden_size]
        :param linear:  线性变换
        :return: [batch_size, max_step, hidden_size]
        """
        # [batch_size * max_step, hidden_size]
        embedding = feature_embedding.view(-1, self.embedding_size)

        # [batch_size * max_step, hidden_size]
        embedding = linear(embedding)

        # [batch_size * max_step, hidden_size]
        embedding = F.relu(embedding)

        # [batch_size, max_step, hidden_size]
        embedding = embedding.view(-1, self.max_step, self.hidden_size)
        return embedding

    def get_neighbors(self, n_hop, question_index):
        """
        :param question_index: [batch_size,seq_len]
        :return: [[batch_size, max_len],[batch_size, max_len, question_neighbor_num] * n_hop]
        """
        seeds = [question_index.contiguous()]

        for i in range(n_hop):
            if i % 2 == 0:
                neighbor = self.question_neighbors[seeds[i].view(-1), :]
                neighbor = neighbor.view(-1, self.max_step, self.question_neighbor_num)
            else:
                neighbor = self.skill_neighbors[seeds[i].view(-1), :]
                neighbor = neighbor.view(-1, self.max_step, self.skill_neighbor_num)
            seeds.append(neighbor)

        return seeds

    def aggregate(self, input_neighbors, input_questions_embedding):
        """
        :param input_neighbors: [[batch_size, max_step],[batch_size, max_step, question_neighbor_num] * n_hop]
        :param input_questions_embedding: [batch_size, max_step, embedding_size]
        :return: [[batch_size, max_step],[batch_size, max_step, question_neighbor_num]]
        """
        sq_neighbor_vectors = []
        for hop_i, neighbors in enumerate(input_neighbors):
            if hop_i % 2 == 0:  # question
                temp_neighbors = self.feature_embedding[neighbors, :]
                temp_neighbors = temp_neighbors.view(self.batch_size, self.max_step, -1, self.embedding_size)
                sq_neighbor_vectors.append(temp_neighbors)
            else:  # skill
                temp_neighbors = self.feature_embedding[neighbors, :]
                temp_neighbors = temp_neighbors.view(self.batch_size, self.max_step, -1, self.embedding_size)
                sq_neighbor_vectors.append(temp_neighbors)

        aggregators = []
        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size,
                                                   act=nn.Tanh(), dropout=self.keep_prob_gnn)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size,
                                                   act=nn.Tanh(), dropout=self.keep_prob_gnn)
            aggregator.to(self.args.device)
            aggregators.append(aggregator)

            # vectors_next_iter = []
            for hop in range(self.n_hop - i):  # aggregate from outside to inside#layer
                if hop % 2 == 0:
                    shape = [self.batch_size, self.max_step, -1, self.question_neighbor_num, self.embedding_size]
                    # [batch_size,seq_len, -1, dim]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=sq_neighbor_vectors[hop + 1].view(shape),
                                        question_embeddings=sq_neighbor_vectors[hop], )
                else:
                    shape = [self.batch_size, self.max_step, -1, self.skill_neighbor_num, self.embedding_size]
                    # [batch_size,seq_len, -1, dim]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=sq_neighbor_vectors[hop + 1].view(shape),
                                        question_embeddings=sq_neighbor_vectors[hop], )
                # shape = [self.batch_size, self.max_step, -1, self.sample_neighbor_num, self.embedding_size]

                # vectors_next_iter.append(vector)
                sq_neighbor_vectors[hop] = vector
            # sq_neighbor_vectors = vectors_next_iter

        # res = tf.reshape(sq_neighbor_vectors[0], [self.batch_size,self.max_step, self.embedding_size])
        # [[batch_size,max_step,-1,embedding_size]...]
        res = sq_neighbor_vectors

        return res, aggregators

    def hist_neighbor_sampler_hard(self, input_embedding):
        """
        :param input_embedding: [batch_size, max_step, hidden_size]
        :return: [batch_size, max_step, hist_neighbor_num, hidden_size]
        """
        # [batch_size,1,hidden_size]定义一个全零嵌入
        zero_emb = torch.zeros(self.batch_size, self.hidden_neurons[-1], dtype=torch.float32, device=self.device)
        zero_emb = zero_emb.unsqueeze(1)

        # [batch_size,max_step+1,hidden_size]
        input_embedding = torch.cat([input_embedding, zero_emb], 1)

        # [self.batch_size, max_step*M]
        temp_hist_index = self.hist_neighbor_index.view(-1, self.max_step * self.hist_neighbor_num)
        temp_hist_index = temp_hist_index.unsqueeze(-1).expand(-1, -1, input_embedding.shape[-1])

        hist_neighbors_features = torch.gather(input_embedding, 1, temp_hist_index)
        hist_neighbors_features = hist_neighbors_features.view(-1, self.max_step, self.hist_neighbor_num,
                                                               input_embedding.shape[-1])
        return hist_neighbors_features

    def hist_neighbor_sampler_soft(self, input_q_emb, next_q_emb, qa_emb):
        """
        :param input_q_emb: [batch_size, max_step, embedding_size]
        :param next_q_emb: [batch_size, max_step, embedding_size]
        :param qa_emb: [batch_size, max_step, hidden_size]
        :return: [batch_size, max_step, hist_neighbor_num, hidden_size]
        """
        # [batch_size, max_step]
        next_q_mod = torch.sqrt(torch.sum(next_q_emb * next_q_emb, dim=-1))

        # [batch_size, max_step]
        input_q_mod = torch.sqrt(torch.sum(input_q_emb * input_q_emb, dim=-1))

        # [batch_size, max_step, max_step]
        q_similarity = torch.sum(next_q_emb.unsqueeze(2) * input_q_emb.unsqueeze(1), -1)

        q_similarity = q_similarity / (next_q_mod.unsqueeze(2) * input_q_mod.unsqueeze(1))

        # [batch_size, 1, hidden_size]
        zero_emb = torch.zeros([self.batch_size, self.hidden_neurons[-1]], dtype=torch.float32, device=self.device)
        zero_emb = zero_emb.unsqueeze(1)

        # answer, [batch_size, max_step + 1, hidden_size]
        qa_emb = torch.cat([qa_emb, zero_emb], 1)

        # mask future position
        seq_mask = torch.arange(1, self.max_step + 1)

        # [batch_size, max_step, max_step]
        similarity_seqs = (seq_mask < self.max_step).float().unsqueeze(0).to(self.device)
        similarity_seqs = torch.tile(similarity_seqs, [self.batch_size, 1, 1])
        # mask_seqs = tf.tile(tf.expand_dims(similarity_seqs,-1),[1,1,1,self.embedding_size])
        # input_qa_emb = mask_seqs*input_qa_emb

        # only history q none zero
        # [batch_size, max_step, max_step]
        q_similarity = q_similarity * similarity_seqs

        # setting lower similarity bound
        lower_bound = torch.zeros([self.batch_size, self.max_step, self.max_step], device=self.device)
        q_similarity = torch.where(q_similarity > self.args.att_bound, q_similarity, lower_bound)

        # [batch_size, max_step, hist_neighbor_num]
        hist_attention_value, temp_hist_index = torch.topk(q_similarity, self.hist_neighbor_num, dim=-1)

        replace_index = torch.ones([self.batch_size, self.max_step, self.hist_neighbor_num], dtype=torch.int64,
                                   device=self.device)
        replace_index = (qa_emb.shape[-2] - 1) * replace_index
        temp_hist_index = torch.where(hist_attention_value > 0, temp_hist_index, replace_index)
        temp_hist_index = temp_hist_index.view([-1, self.max_step * self.hist_neighbor_num])
        temp_hist_index = temp_hist_index.unsqueeze(-1).expand(-1, -1, qa_emb.shape[-1])

        hist_neighbors_features = torch.gather(qa_emb, 1, temp_hist_index)
        hist_neighbors_features = hist_neighbors_features.view(
            [-1, self.max_step, self.hist_neighbor_num, qa_emb.shape[-1]])
        return hist_neighbors_features

    def next_neighbor_sampler(self, aggregate_embedding):
        """
        :param aggregate_embedding: [[batch_size, max_step, hidden_size], [batch_size, max_step, question_neighbor_num, hidden_size]]
        :return: [batch_size, max_step, next_neighbor_num, hidden_size]
        """
        # [batch_size, max_step, question_neighbor_num, hidden_size]
        embedding = aggregate_embedding[1]

        # [batch_size * max_step, question_neighbor_num, hidden_size]
        embedding = embedding.view(-1, self.question_neighbor_num, self.embedding_size)

        # [question_neighbor_num, batch_size * max_step, hidden_size]
        embedding = embedding.permute(1, 0, 2)

        # [question_neighbor_num, batch_size * max_step, hidden_size]
        embedding = embedding[torch.randperm(embedding.size(0)), :, :]

        # [batch_size * max_step, question_neighbor_num, hidden_size]
        embedding = embedding.permute(1, 0, 2)

        # 问题的邻居数量小于采样的邻居数量，则进行复制操作增加邻居
        if self.question_neighbor_num < self.next_neighbor_num:
            embedding = embedding.repeat(1, -(-self.next_neighbor_num // embedding.size(0)), 1)

        # [batch_size * max_step, next_neighbor_num, hidden_size]
        embedding = embedding[:, :self.next_neighbor_num, :]

        # [batch_size, max_step, next_neighbor_num, hidden_size]
        embedding = embedding.view(self.batch_size, self.max_step, self.next_neighbor_num, self.embedding_size)
        return embedding


# ----------------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)

    def forward(self, x):
        out = x.view(-1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class CMAKT(nn.Module):
    def __init__(self, args):
        super(CMAKT, self).__init__()

        self.args = args
        self.device = args.device
        self.max_step = args.max_step - 1
        self.select_index = args.select_index

        # (skill, quest, answer) embedding, [feature_answer_size, embedding_size]
        initial_value = torch.rand(args.feature_answer_size, args.embedding_size)
        self.feature_embedding = nn.Parameter(initial_value, requires_grad=True)

        self.gikt = GIKT(args)
        self.saint_plus = SaintPlus(args)

        self.fc = nn.Linear(self.max_step, self.max_step)

    def forward(self, features_answer_index, hist_neighbor_index):
        features_answer_index = torch.from_numpy(features_answer_index).long().to(self.device)
        hist_neighbor_index = torch.from_numpy(np.array(hist_neighbor_index, dtype=np.int64)).to(self.device)
        batch_size = features_answer_index.size(0)

        # 三元组索引
        # [batch_size, max_step + 1, field_size]
        select_feature_index = features_answer_index[:, :, self.select_index]

        # 技能索引
        # [batch_size, max_step]
        # [batch_size, max_step, embedding_size]
        skill_index = select_feature_index[:, :-1, 0]
        input_skills_embedding = self.feature_embedding[skill_index, :]

        # [batch_size, max_step]
        # [batch_size, max_step, embedding_size]
        next_skill_index = select_feature_index[:, 1:, 0]
        next_skills_embedding = self.feature_embedding[next_skill_index]

        # 问题索引
        # [batch_size, max_step]
        # [batch_size, max_step, embedding_size]
        questions_index = select_feature_index[:, :-1, 1]
        input_questions_embedding = self.feature_embedding[questions_index]

        # [batch_size, max_step]
        # [batch_size, max_step, embedding_size]
        next_questions_index = select_feature_index[:, 1:, 1]
        next_questions_embedding = self.feature_embedding[next_questions_index]

        # 答案索引
        # [batch_size, max_step]
        # [batch_size, max_step, embedding_size]
        input_answers_index = select_feature_index[:, :-1, -1]
        input_answers_embedding = self.feature_embedding[input_answers_index]

        params = {
            "skill_index": skill_index,
            "next_skill_index": next_skill_index,
            "questions_index": questions_index,
            "next_questions_index": next_questions_index,
            "input_answers_index": input_answers_index,

            "input_skills_embedding": input_skills_embedding,
            "next_skills_embedding": next_skills_embedding,
            "input_questions_embedding": input_questions_embedding,
            "next_questions_embedding": next_questions_embedding,
            "input_answers_embedding": input_answers_embedding,

            "batch_size": batch_size,
            "feature_embedding": self.feature_embedding,
            "hist_neighbor_index": hist_neighbor_index,
        }

        # [batch_size, max_step]
        h_gikt = self.gikt(params)

        # [batch_size, max_step]
        h_saint_plus = self.saint_plus(params)

        # [batch_size, max_step]
        logits = self.fc(h_gikt + h_saint_plus)

        pred = torch.sigmoid(logits.view(-1, self.max_step))
        binary_pred = (pred >= 0.5)
        return binary_pred, pred

import collections
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertPreTrainedModel, BertConfig, BertLMPredictionHead

from common.utils import is_equal
from common.data_helper import load_data, to_infix


pretrained_model = 'hfl/chinese-bert-wwm'
max_question_length = 128
max_equation_length = 64
max_length = max_question_length + max_equation_length
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained(pretrained_model)


def calc_attention_mask(token_type_ids):
    # bert_seq2seq中Causal attention mask的生成
    # seq_len = token_type_ids.shape[1]
    # ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32).to(device)
    # a_mask = ones.tril() # 下三角矩阵
    # s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(2).float()
    # s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3).float()
    # a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask 
    # return a_mask
    
    # bert4keras中Causal attention mask的生成
    idxs = torch.cumsum(token_type_ids, dim=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = mask[:, None]
    return mask


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        # 直接删除Pooler层，减少计算量和显存开销
        # self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None
    ):  
        extended_attention_mask = (input_ids > 0).float()
        # 注意力矩阵mask: [batch_size, 1, 1, seq_length]
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        if attention_mask is not None :
            ## 如果传进来的注意力mask不是null，那就直接用传进来的注意力mask 乘 原始mask
            # 注意 原始mask是extended_attention_mask，这个是用来把pad部分置为0，去掉pad部分影响
            extended_attention_mask = attention_mask * extended_attention_mask

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_layers = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )

        return encoder_layers[0]


class UniLMModel(nn.Module):
    
    def __init__(self, pretrained_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        config = BertConfig(vocab_size=vocab_size)
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.decoder = BertLMPredictionHead(config)
        self.decoder.decoder.weight.data = self.bert.embeddings.word_embeddings.weight.data
        for param in self.bert.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, token_type_ids, compute_loss=False):
        """
        参数：
            input_ids：一个batch的输入token id，用[PAD]作为填充符号
            token_type_ids：指示这个句子归属于setence1还是sentence2
            compute_loss：如果为True，则在forward的最后计算loss，计算方式为交叉熵
        
        使用方法：
            pretrained_model = 'hfl/chinese-bert-wwm'
            tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            model = UniLMModel(pretrained_model, tokenizer.vocab_size)
            
            question = ['小明有3个苹果，小红的苹果是小明的3倍，小红有几个苹果？', '正方形的周长为12，它的边长是多少']
            equation = ['3*5', '12/4']
            
            tokenized = tokenizer(question, equation, return_tensors='pt', padding=True)
            input_ids = tokenized['input_ids']
            token_type_ids = tokenized['token_type_ids']
            
            # 直接调用forward进行推导
            logits = model(input_ids, token_type_ids, compute_loss=False)
            
            # 用于训练
            optimizer = nn.optimizer.Adam(model.parameters(), lr=2e-5)
            loss, logits = model(input_ids, token_type_ids, compute_loss=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        """
        mask = calc_attention_mask(token_type_ids)
        encoder_layers = self.bert(input_ids,token_type_ids=token_type_ids, attention_mask=mask)
        logits = self.decoder(encoder_layers)
        if compute_loss:
            labels = input_ids.view(-1)[token_type_ids.view(-1)==1]
            logits = logits[:, :-1].contiguous()
            target_mask = token_type_ids[:, 1:].contiguous().view(-1)
            predictions = logits.view(-1, self.vocab_size)
            predictions = predictions[target_mask==1]
            loss_function = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_function(predictions, labels)
            return loss, logits
        return logits
    
    def generate(self, text, beam_size=1):
        """根据问题生成表达式
        
        输入：
            text：问题的文本
            beam_size：beam search过程中的top k值
        """
        tokenized = tokenizer(text, max_length=max_question_length, truncation=True, return_tensors='pt')
        token_ids = tokenized['input_ids'].to(device)
        token_type_ids = tokenized['token_type_ids'].to(device)
        
        token_ids = token_ids.view(1, -1)
        token_type_ids = token_type_ids.view(1, -1)
        out_puts_ids = self.beam_search(token_ids, token_type_ids, beam_size=beam_size, max_length=max_equation_length)
        # 去掉最后的[SEP]
        out_puts_ids = out_puts_ids[:-1]
        decoding_text = tokenizer.decode(out_puts_ids.cpu().numpy(), spaces_between_special_tokens=False)
        decoding_text = decoding_text.replace(' ', '')
        return decoding_text

    def beam_search(self, token_ids, token_type_ids, max_length, beam_size=1):
        """beam-search操作
        
        这里的max_length指的是生成序列的最大长度
        """
        
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, dtype=torch.long).to(device)
        # 用来保存累计得分
      
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0]).to(device)
            for step in range(max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)
                
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
               
                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids).to(device)], dim=1)

                end_counts = (output_ids == tokenizer.sep_token_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
    
            return output_ids[output_scores.argmax()]


def evaluate(model, valid_loader, tokenizer, show_process_bar=True):
    """在验证集上进行一轮准确率的评估
    
    输入：
        model：经过训练的模型
        valid_loader：一个可迭代对象，能够迭代出（问题文本，表达式文本，答案）三元组
        show_process_bar：是否展示进度条
    """
    model.eval()
    total, right = 0, 0
    for question, equation, answer in valid_loader:
        total += 1
        pred_equation = model.generate(question)
        try:
            pred_equation = to_infix(pred_equation)
            pred_equation = pred_equation.replace('^', '**')
            right += int(is_equal(eval(pred_equation), eval(answer)))
        except BaseException as e:
            pass
    acc = right / total
    return acc


class MathDataset(Dataset):

    def __init__(self, file_path):
        self._source = load_data(file_path)
    
    def __getitem__(self, idx):
        return self._source[idx]
    
    def __len__(self):
        return len(self._source)
    

def collate_fn(batch):
    """将一批训练集转换为输入到模型中的tensor
    
    输入的文本将被截断为max_length
    """
    questions, equations, answers = zip(*batch)
    tokenized = tokenizer(questions, equations, 
                          max_length=max_length, 
                          padding=True, truncation=True, 
                          return_tensors='pt')
    batch_input_ids = tokenized['input_ids']
    batch_token_type_ids = tokenized['token_type_ids']
    return batch_input_ids, batch_token_type_ids


train_dataset = MathDataset('dataset/ape210k/train.ape.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
small_valid_loader = load_data('dataset/test/test.json')
valid_loader = load_data('dataset/ape210k/valid.ape.json')


# 删除冗余的Position Embedding
def remove_redundant_position_embedding(model, max_length):
    embeddings = model.bert.embeddings
    original_embedding = embeddings.position_embeddings
    embeddings.position_embeddings = nn.Embedding(max_length, original_embedding.embedding_dim)
    embeddings.position_embeddings.weight.data = original_embedding.weight.data[:max_length]


# 简化词表
def _simplify_tokenizer_vocab(tokenizer, keep_set):
    new_vocab = collections.OrderedDict()
    new_id = 0
    for tok, ids in tokenizer.vocab.items():
        if ids in keep_set:
            new_vocab[tok] = new_id
            new_id += 1
    tokenizer.vocab = new_vocab
    tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])
    
    
def _remove_linear_weight(linear, keep_list):
    new_linear = nn.Linear(linear.in_features, len(keep_list))
    new_linear.weight.data = linear.weight.data[keep_list]
    new_linear.bias = nn.Parameter(torch.zeros(len(keep_list)))
    return new_linear
    

def _remove_embedding_weight(embedding, keep_list):
    new_embedding = nn.Embedding(len(keep_list), embedding.embedding_dim)
    new_embedding.weight.data = embedding.weight.data[keep_list]
    return new_embedding
    
    
def simplify_model_vocab(model, keep_list):
    model.decoder.decoder = _remove_linear_weight(model.decoder.decoder, keep_list)
    model.bert.embeddings.word_embeddings = _remove_embedding_weight(model.bert.embeddings.word_embeddings, keep_list)
    
    
def simplify_tokenizer_vocab(tokenizer, keep_vocab):
    """Keep vocab"""
    keep_list = tokenizer.convert_tokens_to_ids(keep_vocab)
    keep_set = set(keep_list)
    _simplify_tokenizer_vocab(tokenizer, keep_set)
    return keep_list
    

# 剔除掉不必要的词
def _is_cjk_character(ch):
    """CJK类字符判断（包括中文字符也在此列）
    参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F


def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
                58 <= code <= 64 or \
                91 <= code <= 96 or \
                123 <= code <= 126 or \
                unicodedata.category(ch).startswith('P')


def keep_token(t):
    keep = True
    if len(t) > 1:
        for c in t.lstrip('##'):
            if (
                _is_cjk_character(c) or
                _is_punctuation(c)
            ):
                return False
    return True


def get_keep_vocab():
    print(f'Original vocab size: {tokenizer.vocab_size}')
    original_keep = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
    keep_vocab = []
    for t in tokenizer.vocab.keys():
        if t not in original_keep and keep_token(t):
            keep_vocab.append(t)
    keep_vocab = original_keep + keep_vocab
    print(f'Simplified vocab size: {len(keep_vocab)}')
    return keep_vocab


keep_vocab = get_keep_vocab()
keep_list = simplify_tokenizer_vocab(tokenizer, keep_vocab)
model = UniLMModel(pretrained_model, vocab_size=tokenizer.vocab_size)
print(f'原参数数量：{sum(p.numel() for p in model.parameters())}')

simplify_model_vocab(model, keep_list)
# 截断Position Embedding
remove_redundant_position_embedding(model, max_length)
print(f'优化后参数数量：{sum(p.numel() for p in model.parameters())}')


model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 100
eval_per_epoch = 5

for epoch in range(epochs):
    model.train()
    total_iter = len(train_loader)
    # check_step = int(total_iter / eval_per_epoch)
    check_step = 0
    for i, (input_ids, token_type_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        loss, logits = model(input_ids, token_type_ids=token_type_ids, compute_loss=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if check_step != 0 and (i+1) % check_step == 0:
            acc = evaluate(model, small_valid_loader, tokenizer, show_process_bar=False)
            print(f'Epoch: {epoch}, Accuracy: {acc}')
    acc = evaluate(model, valid_loader, tokenizer)
    print(f'Epoch: {epoch}, Final Accuracy: {acc}')

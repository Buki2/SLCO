import json
from tqdm import tqdm
import string
from pytorch_pretrained_bert.tokenization import BertTokenizer

from modules import build_model_by_name
import options as options
import evaluation_metrics as evaluation_metrics
from utils import print_sentence_predictions, load_vocab
from nltk.corpus import stopwords
stopword = set(stopwords.words("english"))


def convert_to_segment(know_score_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    word_score_dict = {}
    for k, v in tqdm(know_score_data.items()):
        word_id = v[0]
        fact_score = v[1]
        tokens = tokenizer.convert_ids_to_tokens(word_id)
        word_score = []
        score = 0
        tmp = 1
        for i in range(len(tokens)):
            if tokens[i] == '[CLS]' or tokens[i] == '[SEP]' or tokens[i] == '[PAD]':
                if tokens[i] == '[PAD]':
                    if tmp > 1:
                        word_score.pop()
                        word_score.append(score / tmp)
                        tmp = 1
                    break
            else:
                if tokens[i][0] == '#':
                    score += fact_score[i]
                    tmp += 1
                else:
                    if tmp > 1:
                        word_score.pop()
                        word_score.append(score / tmp)
                        tmp = 1
                    score = fact_score[i]
                    word_score.append(score)
        word_score_dict[k] = word_score
    return word_score_dict


def prompt_retrieval(fact_pseudo_label):

    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    models = {}
    models['bert'] = build_model_by_name('bert', args)
    vocab_subset = None
    # Vocabulary: all knowledge categories in KB-Ref, obtaining from 3 files of knowledge bases
    args.common_vocab_filename = './knowledge_retrieval/data/kbref_knowledge_category_vocab.txt'
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    # Loading referring expressions
    data_path = './knowledge_retrieval/data/kbref_val_anno_format.json'
    with open(data_path, 'r') as f:
        data_val = json.load(f)
    data_path = './knowledge_retrieval/data/kbref_test_anno_format.json'
    with open(data_path, 'r') as f:
        data_test = json.load(f)
    data = data_val if len(fact_pseudo_label) == len(data_val) else data_test

    # Loading ground-truth knowledge category
    gt_fact_label_path = './knowledge_retrieval/data/gt_knowledge_category.json'
    with open(gt_fact_label_path, 'r') as f_fact:
        gt_fact_label = json.load(f_fact)

    # Loading all object names in images
    obj_label_path = './knowledge_retrieval/data/obj_label_frcn_conf025.json'
    with open(obj_label_path, 'r') as f:
        obj_labels = json.load(f)

    # Adding [MASK] into referring expressions
    Expression = {str(i[0][:-4] + '_' + i[1]): i[3] for i in data}
    NewExpression = {}
    print('Rewrite expressions ...')
    cnt_no_knowledge_required = 0
    score_thres = 0.35
    for k, v in tqdm(Expression.items()):
        v = v.lower()
        v = v.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        v_list = v.split()
        new_phrase = []
        new_phrase.append('A')
        new_phrase.append('[MASK]')
        new_phrase.append('is')
        old_phrase = []
        for i in range(len(fact_pseudo_label[k])):
            if fact_pseudo_label[k][i] > score_thres:
                new_phrase.append(v_list[i])
                old_phrase.append(v_list[i])

        # No knowledge required
        old_phrase_set = set(old_phrase)
        if len(old_phrase_set) == 0:
            cnt_no_knowledge_required += 1
            continue
        else:
            if old_phrase_set.issubset(stopword):
                cnt_no_knowledge_required += 1
                continue

        # For short expressions
        old_phrase_rm_sw = old_phrase_set - stopword
        if len(old_phrase_rm_sw) == 1:
            new_phrase = '[MASK] is a kind of ' + list(old_phrase_rm_sw)[0] + '.'
            NewExpression[k] = new_phrase
            continue

        # For sentence fluency
        if new_phrase[-1] == 'is':
            new_phrase.pop()

        new_phrase = " ".join(new_phrase)
        new_phrase += '.'
        NewExpression[k] = new_phrase

    # Prompt-based retrieval
    new_dict = {}
    cnt_all = 0
    cnt_match_fact_label = 0
    cnt_match_fact_label_top2 = 0
    cnt_match_fact_label_top3 = 0
    success_list = []
    fail_list = []
    print('Prompting ...')
    for k, v in tqdm(NewExpression.items()):
        text = v
        sentences = [text]
        for model_name, model in models.items():
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=True)
            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                id1, id2 = k.split('_')
                vocab_subset = obj_labels[id1]
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                if len(index_list) == 0:
                    filtered_log_probs_list = original_log_probs_list
                    index_list = None
                else:
                    filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                fact_obj_label_all = evaluation_metrics.get_ranking_return_all(filtered_log_probs_list[0],
                                                                                    masked_indices, model.vocab,
                                                                                    index_list=index_list,
                                                                                    original_text=text)

            gt = gt_fact_label.get(k, [])
            fact_obj_label = fact_obj_label_all[0]
            if fact_obj_label in gt:
                cnt_match_fact_label += 1
                success_list.append([k, sentences[0], gt, fact_obj_label_all])
            else:
                fail_list.append([k, sentences[0], gt, fact_obj_label_all])
            if len(fact_obj_label_all) >= 2:
                if fact_obj_label_all[0] in gt or fact_obj_label_all[1] in gt:
                    cnt_match_fact_label_top2 += 1
            if len(fact_obj_label_all) >= 3:
                if fact_obj_label_all[0] in gt or fact_obj_label_all[1] in gt or fact_obj_label_all[2] in gt:
                    cnt_match_fact_label_top3 += 1

            if len(fact_obj_label_all) >= 3:
                new_dict[k] = fact_obj_label_all[:3]
            elif len(fact_obj_label_all) == 2:
                new_dict[k] = fact_obj_label_all[:2]
                new_dict[k].append('')
            elif len(fact_obj_label_all) == 1:
                new_dict[k] = fact_obj_label_all[:1]
                new_dict[k].append('')
                new_dict[k].append('')
            else:
                new_dict[k] = ['', '', '']

            cnt_all += 1

    print('-----------------------------')
    print('Accuracy (knowledge retrieval) R@1:\t' + str(cnt_match_fact_label / cnt_all))
    print('Accuracy (knowledge retrieval) R@2:\t' + str(cnt_match_fact_label_top2 / cnt_all))
    print('Accuracy (knowledge retrieval) R@3:\t' + str(cnt_match_fact_label_top3 / cnt_all))

    return new_dict


if __name__ == '__main__':
    # Generated by segment detection module
    with open('./knowledge_retrieval/output_knowledge_score.json', 'r') as f:
        know_score_data = json.load(f)
    segment = convert_to_segment(know_score_data)
    result = prompt_retrieval(segment)
    with open('./knowledge_retrieval/knowledge_retrieval_results.json', 'w') as f:
        json.dump(result, f)

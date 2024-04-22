import argparse
import json
import os
from tqdm import tqdm
import string
import re
from collections import Counter
from collections import defaultdict
import concurrent.futures
import time
import datetime
import requests


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Computes ROUGE given one file of hypotheses and one file of references.
Lines should be aligned (e.g. hypothesis 1 corresponds to reference 1)
"""

from rouge import Rouge
from nltk import PorterStemmer
import argparse

stemmer = PorterStemmer()

def open_data(hypotheses, references):
	with open(hypotheses) as f:
		hypoth_data = f.readlines()
	with open(references) as f:
		ref_data = f.readlines()
	assert(len(ref_data) == len(hypoth_data))
	return hypoth_data, ref_data

def prepare(hypotheses, references):
	hypoth = [" ".join([stemmer.stem(i) for i in line.split()]) for line in hypotheses]
	ref = [" ".join([stemmer.stem(i) for i in line.split()]) for line in references]
	return hypoth, ref

def rouge_calculation(hypotheses, references):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    print(scores)
    return

parser = argparse.ArgumentParser(description='')
parser.add_argument("--data_path", type=str, required=True)
        
args = parser.parse_args()
prompt_acc = '''In the following task, you are given a Question, a model Prediction for the Question, and a Ground-truth Answer to the Question. You should decide whether the model Prediction implies the Ground-truth Answer.\n\nQuestion\n{question}\n\nPrediction\n{model_output}\n\nGround-truth Answer\n{answer}\n\nDoes the Prediction imply the Ground-truth Answer? Output Yes or No:'''

def send_post_request(prompt_str, model_name, num_return=1):

    url = "your url"
    headers = {
        "your headers"
    }
    if num_return > 1:
        top_p = 0.9
        presence_penalty = 1.0
        frequency_penalty = 1.0
    else:
        top_p = 1.0
        presence_penalty = 0.0
        frequency_penalty = 0.0
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_str}],
        "n": num_return,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "response_format": { "type": "json_object" }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        gpt_res = response.json()
        res = gpt_res['choices']
        res = [r['message']['content'] for r in res]
        
        return res
    else:
        return ''

def process_item_to_ask(item, model_name, num_return):
    try:
        prompt_to_ask = item['prompt_to_ask']

        res = send_post_request(prompt_to_ask, model_name, num_return)

        item['gpt_out'] = res

        return item

    except Exception as e:
        print(e)
        item['gpt_out'] = ''
        return item

def get_batch_request(all_data, num_workers, model_name, num_return=1):
    all_data_collect_list = []
    for k, v in all_data.items():
        all_data_collect_list.append({"ori_question": k, "prompt_to_ask": v})
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {executor.submit(process_item_to_ask, item, model_name, num_return): item for item in all_data_collect_list}

        progress = tqdm(total=len(future_to_item), desc="Processing items", ncols=75)

        for future in concurrent.futures.as_completed(future_to_item):
            result = future.result()
            all_results.append(result)
            progress.update(1)
        progress.close()
    return_results = {}
    for d in all_results:
        return_results[d['ori_question']] = d['gpt_out']
    return return_results

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
def isEM(label, output):
    map_dict = {"true": "yes", "false": "no"}
    if label == output:
        return True
    elif label.replace(" ", "") == output.replace(" ", ""):
        return True
    elif label in map_dict:
        if map_dict[label] == output:
            return True
        else:
            return False
    else:
        label = label.split(" ")
        output = output.split(" ")
        if not len(label) == len(output):
            return False
        for l in label:
            if l not in output:
                return False
        return True

if __name__ == "__main__":
    alias = open("../data/id_aliases.json").readlines()
    alias = [json.loads(d) for d in alias]
    alias_dict = {}
    for d in alias:
        alias_dict[d['Q_id']] = d['aliases']
    if os.path.exists(args.data_path + ".gpteval"):
        evaluated_res = json.load(open(args.data_path + ".gpteval"))
    else:
        evaluated_res = []
    try:
        res = open(args.data_path, 'r', encoding='utf-8').readlines()
        res = [json.loads(d) for d in res]
    except:
        res = json.load(open(args.data_path))
    evaluated_res_dict = {}
    for d in evaluated_res:
        q = d['question']
        evaluated_res_dict[q] = d
    for d in res:
        q = d['question']
        if q in evaluated_res_dict:
            for k, v in evaluated_res_dict[q].items():
                if k not in d:
                    d[k] = v
    print(len(res))
    not_strict_dict = []
    incorrect = []
    all_nums, nums, strict_num = 0, 0, 0
    all_acc_prompt = {}
    mismatch = []


    acc_eval = {}
    evaluated = False

    for idx_d, d in enumerate(res):
        q = d['question']
        if "task_type" in d:
            q = q + "\t" + d["task_type"]
        if "dataset" in d:
            dataset = d['dataset']
        if "gpt_eval" in d and d['gpt_eval'] != "":
            gpt_eval = d['gpt_eval']
        else:
            gpt_eval = None
        if "main_passages" in d:
            main_passages = d['main_passages']
        else:
            main_passages = []
        if "answer" in d:
            answer_key = "answer"
        elif "answers" in d:
            answer_key = "answers"
        else:
            answer_key = "short_answers"
        evidence_key = "evidences" if "evidences" in d else "question_decomposition"
        if evidence_key in d:
            evidence = d[evidence_key]
        else:
            evidence = []
        if "history" in d:
            history = d['history']
        else:
            history = None
        if "known information" in d:
            known_info = d['known information']
        else:
            known_info = None
        if "final_res" not in d:
            continue
        final_res = d['final_res']
        try:
            gpt_answer = str(final_res['answer']['text'])
        except:
            continue
        gpt_answer = normalize_answer(str(gpt_answer))
        if "confidence" in final_res["answer"]:
            confidence = final_res['answer']['confidence']
        elif "confidence" in final_res:
            confidence = final_res["confidence"]
        else:
            confidence = 5
        if isinstance(d[answer_key], str) or isinstance(d[answer_key], bool):
            if "dataset" in d and d['dataset'] == "trivia":
                labels = eval(d[answer_key])
            else:
                labels = [str(d[answer_key])]
        else:
            labels = d[answer_key]
        if "answer_aliases" in d:
            labels.extend(d['answer_aliases'])
        elif "answer_id" in d:
            label_id = d['answer_id']
            if label_id is not None and label_id in alias_dict:
                labels.extend(alias_dict[label_id])
        else:
            labels = labels
        labels = list(set(labels))
        
        answer_match = False
        strict = False
        all_nums += 1
        for label in labels:
            label = label.replace('''"''', "")
            label = normalize_answer(label)
            if isEM(label, gpt_answer):
                strict = True
            if label in gpt_answer:
                answer_match = True
            else:
                continue
        if "gpt_eval" not in d or d['gpt_eval'] == "":
            prompt_to_ask = prompt_acc.format(question=q, model_output=gpt_answer, answer=labels)
            all_acc_prompt[q] = prompt_to_ask
        else:
            acc_eval[q] = d['gpt_eval']
        if strict:
            d['strict'] = True
            strict_num += 1
            nums += 1
            d['answer_match'] = True
        elif answer_match:
            nums += 1
            d['strict'] = False
            d['answer_match'] = True
        else:
            d['strict'] = False
            d['answer_match'] = False
        if "gpt_eval" not in d:
            if not strict:
                prompt_to_ask = prompt_acc.format(question=q, model_output=gpt_answer, answer=labels)
                all_acc_prompt[q] = prompt_to_ask
            else:
                d["gpt_eval"] = ["yes"]
        else:
            acc_eval[q] = d['gpt_eval']
       
    if "strategy" not in args.data_path.lower(): 
        new_acc_eval = get_batch_request(all_acc_prompt, num_workers=50, model_name="gpt-3.5-turbo-1106", num_return=1)
        acc_eval = dict(acc_eval, **new_acc_eval)
        acc_nums = 0
        for d in res:
            q = d['question']
            if "strict" in d:
                strict = d["strict"]
            else:
                strict = False
            if "answer_match" in d:
                answer_match = d['answer_match']
            else:
                answer_match = False
            if "task_type" in d:
                q = q + "\t" + d["task_type"]
            if q not in acc_eval:
                continue
            this_eval = acc_eval[q]
            d["gpt_eval"] = this_eval
            if isinstance(this_eval, list):
                this_eval = this_eval[0]
            this_eval = this_eval.strip().rstrip().lower()
            if this_eval == "yes" or strict or ("odqa" in args.data_path and answer_match):
                acc_nums += 1
    else:
        acc_nums = strict_num
        
    print(f"Total valid nums: {all_nums}, cEM nums: {nums}, EM nums: {strict_num}, cEM score: {nums / all_nums}, EM score: {strict_num / all_nums}")
    print("GPT evaluation result: ", acc_nums, acc_nums / all_nums)
    result_save_path = args.data_path.split(".json")[0]
    if len(res) > len(evaluated_res):
        with open(args.data_path + ".gpteval", 'w', encoding='utf-8') as f:
            f.write(json.dumps(res, indent=5, ensure_ascii=False))

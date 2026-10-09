import argparse
import json
import ast
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
from openai import OpenAI


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
parser.add_argument("--api_model", default="gpt-3.5-turbo-1106")
parser.add_argument("--aliases_path", default=None, help="Optional JSONL answer-ID aliases")
        
args = parser.parse_args()
prompt_acc = '''In the following task, you are given a Question, a model Prediction for the Question, and a Ground-truth Answer to the Question. You should decide whether the model Prediction implies the Ground-truth Answer.\n\nQuestion\n{question}\n\nPrediction\n{model_output}\n\nGround-truth Answer\n{answer}\n\nDoes the Prediction imply the Ground-truth Answer? Output Yes or No:'''

def send_post_request(prompt_str, model_name, num_return=1):
    kwargs = {}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_str}],
        n=num_return, stream=False)
    return _response_process(response)

def _response_process(response):
    result = {
        'len': 0,
        'res': None
    }

    if response != None:
        choices = response.choices
        result['len'] = len(choices)
        result['res'] = choices[0].message.content
    else:
        result['len'] = -1
    
    return result['res']

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

def load_records(path):
    with open(path, encoding='utf-8') as source:
        text = source.read()
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(records, dict):
        records = records.get('data', [records])
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f'Expected JSON/JSONL records in {path}')
    return records

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
    alias_path = args.aliases_path or os.path.join(os.path.dirname(__file__), "../data/id_aliases.json")
    alias = load_records(alias_path) if os.path.isfile(alias_path) else []
    alias_dict = {}
    for d in alias:
        alias_dict[d['Q_id']] = d['aliases']
    if os.path.exists(args.data_path + ".gpteval"):
        evaluated_res = load_records(args.data_path + ".gpteval")
    else:
        evaluated_res = []
    res = load_records(args.data_path)
    evaluated_res_dict = {}
    for d in evaluated_res:
        key = (d['question'], d.get('task_type'))
        evaluated_res_dict[key] = d
    for d in res:
        key = (d['question'], d.get('task_type'))
        cached = evaluated_res_dict.get(key)
        # A cached judgement belongs to a particular prediction, not just a question.
        if cached and cached.get('final_res') == d.get('final_res'):
            if not d.get('gpt_eval') and cached.get('gpt_eval'):
                d['gpt_eval'] = cached['gpt_eval']
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
                labels = ast.literal_eval(d[answer_key])
            else:
                labels = [str(d[answer_key])]
        else:
            labels = list(d[answer_key])
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
        if d.get("gpt_eval"):
            acc_eval[q] = d["gpt_eval"]
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
        if not d.get("gpt_eval"):
            if not strict:
                prompt_to_ask = prompt_acc.format(question=q, model_output=gpt_answer, answer=labels)
                all_acc_prompt[q] = prompt_to_ask
            else:
                d["gpt_eval"] = "yes"
                acc_eval[q] = "yes"
        else:
            acc_eval[q] = d['gpt_eval']
       
    if "strategy" not in args.data_path.lower(): 
        new_acc_eval = get_batch_request(all_acc_prompt, num_workers=50, model_name=args.api_model, num_return=1)
        failed_eval = [q for q, result in new_acc_eval.items() if not result]
        if failed_eval:
            raise RuntimeError(f"GPT evaluation failed for {len(failed_eval)} questions; scores were not saved.")
        acc_eval = dict(acc_eval, **new_acc_eval)
        acc_nums = 0
        for d in res:
            if 'final_res' not in d:
                continue
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
        
    if all_nums == 0:
        raise ValueError("No records with valid final answers to evaluate")
    print(f"Total valid nums: {all_nums}, cEM nums: {nums}, EM nums: {strict_num}, cEM score: {nums / all_nums}, EM score: {strict_num / all_nums}")
    print("GPT evaluation result: ", acc_nums, acc_nums / all_nums)
    result_save_path = args.data_path.split(".json")[0]
    with open(args.data_path + ".gpteval", 'w', encoding='utf-8') as f:
        f.write(json.dumps(res, indent=5, ensure_ascii=False))


import json
import os
import requests
from tqdm import tqdm
import copy
import argparse
from utils.search_engine import *
from transformers import (
    BertTokenizer, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    AutoModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
from typing import List, Tuple, Dict, Iterator
from collections import defaultdict, Counter
import concurrent.futures
import time
import datetime
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
import torch
import deepspeed
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

world_size = int(os.getenv("WORLD_SIZE", '1'))
print("Total using GPUs: ", world_size)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def getRetrieval(path, data):
    ori_data = json.load(open(path))
    record = {}
    for d in data:
        record[d['question']] = d
    retrieval_type = data[0]['retrieval']
    for d in ori_data:
        if d['question'] in record:
            d['first_retrieval'] = record[d['question']]['first_retrieval']
        else:
            continue
    path_name = path.split(".json")[0]
    with open(path_name + f"_{retrieval_type}.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(ori_data, indent=5, ensure_ascii=False))
    return


def send_post_request(prompt_str):
    client = OpenAI(
        api_key = "",
        base_url = ""
    )
    response = None
    model = model_name
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": 'You are ChatGPT, a model trained by OpenAI.}'},
                {
                    "role": "user",
                    "content": prompt_str
                }
            ],
            # response_format={"type": "json_object"},
            stream=False
        )
    except Exception as e:
        print(e)
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

def gpt_batch_request(all_data, num_workers, model_name, num_return=1):
    all_data_collect_list = []
    for k, v in all_data.items():
        all_data_collect_list.append({"ori_question": k, "prompt_to_ask": v})
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {executor.submit(process_item_to_ask, item, model_name, num_return): item for item in tqdm(all_data_collect_list)}


        for future in concurrent.futures.as_completed(future_to_item):
            result = future.result()
            all_results.append(result)
    return_results = {}
    for d in all_results:
        return_results[d['ori_question']] = d['gpt_out']
    return return_results

def multi_return_process(inputs, type_module="main"):
    if type_module == "main":
        unanswerable_count, confidence_count = 0, 0
        answer_candidate = defaultdict(int)
        final_missing_info, explanation_answerable, explanation_unanswerable = "", "", ""
        for in_ in inputs:
            answer_text = in_['answer']['text']
            answer_confidence = in_['answer']['confidence']
            missing_info = in_["missing_information"]
            explanation = in_['explanation']
            if "unanswerable" not in answer_text.lower() and len(missing_info) < 10 and answer_confidence >= 4:
                explanation_answerable = explanation
                answer_candidate[answer_text] += 1
            else:
                unanswerable_count += 1
                explanation_unanswerable = explanation
                if len(missing_info) > 10:
                    final_missing_info = missing_info
            confidence_count += answer_confidence
        if unanswerable_count >= len(inputs) / 2 or confidence_count / len(inputs) < 4:
            outputs = {
                "answer": {"text": "unanswerable", "confidence": confidence_count // len(inputs)}, 
                "explanation": explanation_unanswerable,
                "missing_information": final_missing_info
            }
        else:
            ranged = sorted(answer_candidate.items(), key=lambda x:x[1], reverse=True)
            final_answer = ranged[0][0]
            outputs = {
                "answer": {"text": final_answer, "confidence": confidence_count // len(inputs)},
                "explanation": explanation_answerable,
                "missing_information": ""
            }
    elif type_module == "leaf":
        info_psg_map = {}
        for in_ in inputs:
            info = in_['info']
            support_idx = in_["support_passages"]
            info_psg_map[info] = support_idx
        infos = list(info_psg_map.keys())
        infos = clusterFilter(infos)
        outputs = [{"info": info, "support_passages": info_psg_map[info]} for info in infos]
    return outputs

def clusterFilter(inputs, num_cluster=3):
    if len(inputs) < num_cluster + 1:
        return inputs
    assert isinstance(inputs, list) and isinstance(inputs[0], str)
    final_res = []
    tfidf = TfidfVectorizer()
    text_vectors = tfidf.fit_transform(inputs)
    text_vectors = text_vectors.toarray()
    clf = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)
    clf.fit(text_vectors)
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_
    labels_count = Counter(labels)
    labels_index = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)
    if len(labels_index) < 3:
        remain_labels = [i for i in range(len(labels_index))]
    else:
        remain_labels = [labels_index[0][0]] if labels_index[1][1] == labels_index[2][1] else [labels_index[0][0], labels_index[1][0]]

    distances = cdist(text_vectors, cluster_centers, 'euclidean')
    for l in remain_labels:
        closest_distances = distances[np.arange(text_vectors.shape[0]), l * np.ones_like(labels)]
        closest_points_indice = np.argsort(closest_distances)[:1]
        choosen_text = inputs[int(closest_points_indice)]
        final_res.append(choosen_text)
    return final_res


class RetGBatched:
    def __init__(self, data, args, searcher, model_name, world_size, wikiData=None, nli_component=None):
        self.data = data
        self.args = args
        self.wikiData = wikiData
        self.world_size = world_size
        self.step = 0
        self.history_passages = {}
        self.known_info = {}
        self.question_list = {}
        self.extra_call = 0
        self.missing_info = defaultdict(str)
        if self.args.demon_path is not None and self.args.add_demon:
            self.demons = json.load(open(self.args.demon_path))
        if nli_component is not None:
            self.nli_model = nli_component[0]
            self.nli_tokenizer = nli_component[1]
        else:
            self.nli_model = None
            self.nli_tokenizer = None
        if self.args.model_name_or_path is not None:
            self.chatModel = ModelEngine(self.args, self.world_size)
        else:
            self.chatModel = None
        self.relationship_info = {
                "maternal grandmother": "maternal grandmother is mother's mother",
                "maternal grandfather": "maternal grandfather is mother's father",
                "paternal grandmother": "paternal grandmother is father's mother",
                "paternal grandfather": "paternal grandfather is father's father",
            }
        for d in self.data:
            q = d['question']
            d['history'] = {}
            d['not json'] = []
            d['retrieval'] = self.args.search_type
            d['model_name'] = model_name
            d["Missing info & query generated"] = []
            self.history_passages[d['question']] = defaultdict(int)
            self.known_info[d['question']] = []
            self.question_list[d['question']] = []
            for k, v in self.relationship_info.items():
                if k in q.lower():
                    self.known_info[q].append(v)
        self.known_info_backup = copy.deepcopy(self.known_info)

        self.instruction_main_head = "Generate the information needed to answer the following Question."
        self.instruction_main_tail = ""
        self.instruction_main_iter_head = '''Answer the Question based solely on the provided Information. If the Information is insufficient to answer the Question, the answer should be "unanswerable", and you should provide your explanation and a summary of missing information. Otherwise, write an accurate and concise answer to the Question with a confidence score varying from 1 (not confident) to 5 (very confident) then explain. Your response should be under the format {"answer": {"text": your answer, "confidence": confidence score}, "explanation": your explanation, "missing_information": the summary of missing information}. Please generate a dict format response.'''
        self.instruction_main_iter_tail = ""

        if self.args.max_iter == 0:
            self.instruction_main_last = '''Answer the Question based on the provided Passages. Provide an accurate and concise answer with a confidence score varying from 1 (not confident) to 5 (very confident) and list the index of support passages, then provide your explanation. If the Question can not be answered with the provided Passages, answer it based on your own knowledge. Your response should be under the format {"answer": {"text": your concise answer, "confidence": confidence score}, "support_passages": [index of support passages], "explanation": your explanation}. Please generate a dict format response.'''
        else:
            self.instruction_main_last = '''Answer the Question based on the provided Information. If the Information is not sufficient to answer the question, answer it based on your own knowledge. Please provide an accurate and concise answer with a confidence score varying from 1 (not confident) to 5 (very confident) and provide your explanation. Your response should be under the format {"answer": {"text": your concise answer, "confidence": confidence score}, "explanation": your explanation}.'''

        self.instruction_summ = '''Summarize the following document within 50 words with the question of interest "{QUESTION}"
Return "irrelevant" if the document is irrelevant to the question. Try to keep all the important
dates, numbers, and names.
Title: {TITLE}
Text: {TEXT}
Summary:'''
        self.instruction_snippet = '''Given the follow passage and the question "{QUESTION}", extract a useful span from the passage
that can answer the question. Resolve all the coreference issues to make the extracted span
understandable standalone. If the passage is not helpful for answering the question, return
"irrelevant".
Title: {TITLE}
Text: {TEXT}
Extracted span:'''

        if "strategy" in self.args.data_path.lower():
            self.instruction_main_head = '''Given the following Question and Information, your task is to reason from the Information whether the answer to the Question is "yes" or "no".'''
            self.instruction_main_tail = '''If you can make your judgment based solely on the provided Passages, give your answer to the question with a confidence score varying from 1 (not confident) to 5 (very confident), list the index of Passages that support your answer then provide an explanation; your response should be under the format {"answer": {"text": Yes or No, "confidence": confidence score}, "support_passages": [index of support passages], "explanation": your explanation}. Otherwise, extract useful information from the Passages and give a summary of the missing information, then provide an answer to the question based on your own knowledge with a confidence score; your response should be under the format {"useful_information": [{"info": useful information in passages for answering the question, "support_passages": [index of passage where the information is located]}], "missing_information": the missing information, "answer": {"text": True or False, "confidence": confidence score}}. Please generate a dict format response.'''
            self.instruction_main_iter_head = '''Given the following Question and Information, your task is to reason from the Information whether the answer to the Question is "yes" or "no".'''
            self.instruction_main_iter_tail = '''If the Information provided is not sufficient to reason an answer, the answer should be "unanswerable", and you should provide your explanation and a summary of missing information. Otherwise, write an accurate and concise answer to the Question with a confidence score varying from 1 (not confident) to 5 (very confident) then explain. Your response should be under the format {"answer": {"text": Yes or No, "confidence": confidence score}, "explanation": your explanation, "missing_information": the summary of missing information}. Please generate a dict format response.'''
            if self.args.max_iter == 0:
                self.instruction_main_last = '''Given the following question and passages, determine whether the statement of the question is true or false with a confidence score varying from 1 (not confident) to 5 (very confident) and provide your explanation. If the question can not be answered with the provided passages, answer it based on your own knowledge. Your response should be under the format {"answer": {"text": true or false, "confidence": confidence score}, "support_passages": [index of support passages], explanation": your explanation}.'''
            else:
                self.instruction_main_last = '''Given the following Question and Information, your task is to reason from the Information whether the answer to the Question is "yes" or "no". If the Information is not sufficient to reason an answer, answer it based on your own knowledge. Please provide an accurate and concise answer with a confidence score varying from 1 (not confident) to 5 (very confident) and provide your explanation. Your response should be under the format {"answer": {"text": Yes or No, "confidence": confidence score}, "explanation": your explanation}.'''
       

        self.instruction_leaf_head = '''Given the following Question and Passages, please distillate useful information from the Passages to address the Question effectively and list the support passage index for each distilled information.'''
        self.instruction_leaf_tail = '''Your response should be under the format {"useful_information": [{"info": statement of distilled useful information combining the question, "support_passages": [indexes of support passages]}]}. Not provided information should not appear in your response. Please generate a dict format response.'''

        self.instruction_query_forward = "Generate relevant information to the given Question."

        self.instruction_query_head = '''Based on the Original Question, Historical Questions, Known Information and Missing Information, write no more than 3 queries that ask for Missing Information to solve the Original Question. If the missing information is multi-hop, decompose it into several simple and single-hop queries. The new queries should not contain redundant information and should differ from the Original Question and Historical Questions. Separate each query with \"\n\".'''
        self.instruction_query_tail = ""
             
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.subscription_key = "your subscription_key"
       
        ## instruction for each single prompt
        self.main_text = defaultdict(str)
        self.leaf_text = defaultdict(str)
        self.leaf_forward_text = defaultdict(str)
        self.query_text = defaultdict(str)
        self.candidate_information = defaultdict(str)

        self.max_iter = self.args.max_iter
        self.reranker = Reranker(self.args)
        
        self.model_name = model_name
        self.searcher = searcher
        
        self.retrieval_res = {}
        
        ## always be maintained
        self.main_passages = defaultdict(list)
        ## refresh each time
        self.leaf_passages = defaultdict(list)
        self.retrieval_trace = defaultdict(list)
        
        
    def promptPassages(self, ori_question, first_question, passage_list):
        snippet = []
        num_ret = 0
        this_question = [first_question[0]]
        previous_question, ori_context = None, None
        new_main_passages, question_list = [], set()
        for i, p in enumerate(passage_list):
            title = None
            drop_ = False
            if isinstance(p, dict):
                context = p['snippet'] if 'snippet' in p else p['text']
                if "name" in p or "title" in p:
                    title = p['name'] if 'name' in p else p['title']
            elif isinstance(p, list):
                ## retrieval from external corpus
                context, score, question, title, ori_context = p
                if previous_question is not None:
                    question_list.add(question)
                else:
                    previous_question = question
                    question_list.add(question)
                if self.args.summ_snippet == "summ":
                    prompt_to_refine = self.instruction_summ.format(QUESTION=question, TITLE=title, TEXT=context)
                    context = self.askGPT({question: prompt_to_refine})
                    new_main_passages.append(context[question][0])
                    self.extra_call += 1
                elif self.args.summ_snippet == "snippet":
                    prompt_to_refine = self.instruction_snippet.format(QUESTION=question, TITLE=title, TEXT=context)
                    context = self.askGPT({question: prompt_to_refine})
                    new_main_passages.append(context[question][0])
                    self.extra_call += 1
                else:
                    pass
            else:
                # GPT knowledge
                context = p
                filter_str = ['sorry', '2022', "have no specific iinformation", "no publicly available information"]
                for str_ in filter_str:
                    if str_ in context.lower():
                        drop_ = True
                question = this_question[0]
            if drop_:
                continue
            if self.args.summ_snippet is not None:
                this_snippet = context[question][0]
            else:
                this_snippet = context
            if ori_context is None:
                ori_context = this_snippet
            if self.history_passages[ori_question][ori_context] >= 2:
                continue
            else:
                self.history_passages[ori_question][ori_context] += 1
            if len(this_snippet.split(" ")) < 5:
                continue
            this_snippet = f'''(Title: {title}) {this_snippet}'''
            snippet.append(this_snippet)
            num_ret += 1
            if num_ret == self.args.top_k:
                break
        if len(new_main_passages) > 0:
            self.main_passages[ori_question] = new_main_passages
        if len(question_list) == 0:
            question_list = this_question
        self.question_list[ori_question].extend(list(question_list))
        res = ""
        for i, s in enumerate(snippet):
            res += f'''Passage {i}: {s}\n'''
        res = res.replace("<b>", "")
        res = res.replace("</b>", "")
        res = res.replace("<br>", "").strip().rstrip()
        return res, question_list
    
    def promptMain(self, num_iter=0):
        self.main_text = defaultdict(str)
        head_instruct = self.instruction_main_iter_head
        tail_instruct = self.instruction_main_iter_tail
        this_instruct = head_instruct + "\n" + tail_instruct if num_iter < self.args.max_iter else self.instruction_main_last
        if self.args.add_demon:
            main_demons = self.demons["main"] if num_iter < self.args.max_iter else self.demons['final_main']
        for index, d in enumerate(self.data):
            if "final_res" in d or "fail" in d:
                continue
            ori_question = d['question']
            if len(self.known_info[ori_question]) == len(self.known_info_backup[ori_question]) and self.step > 1:
                ## no new information is added
                d['history'][f"step {self.step}-main"] = "None"
                continue
            if num_iter == 0 or self.max_iter == 0:
                passages, question = self.promptPassages(ori_question, ori_question, self.main_passages[ori_question])
                if passages == "":
                    passages = "Passages 0: None"
                inputs = f'''{passages}\nQuestion: {ori_question}'''
            else:
                if len(self.known_info[ori_question]) == 0:
                    known_info = "None"
                else:
                    known_info = "; ".join(self.known_info[ori_question])
                inputs = f'''Information: {known_info}\nQuestion: {ori_question}\nYour response:'''
            if self.args.add_demon:
                inputs = f'''{main_demons}\n\n{inputs}'''
            inputs = f'''{this_instruct}\n\n{inputs}'''
            self.main_text[ori_question] = inputs
        return
    
    def promptLeaf(self, questions):
        self.leaf_text = defaultdict(str)
        # backup the current known information, 
        # if no new information is extracted, directly fed to QueryGenerator for new queries generation
        self.known_info_backup = copy.deepcopy(self.known_info)
        this_instruct = self.instruction_leaf_head + "\n" + self.instruction_leaf_tail
        for index, d in enumerate(self.data):
            ori_question = d['question']
            if "final_res" in d or "fail" in d or ori_question not in questions:
                continue
            if ori_question not in self.leaf_passages:
                continue
            question = questions[ori_question]
            passages, question = self.promptPassages(ori_question, question, self.leaf_passages[ori_question])
            if isinstance(question, str):
                question = question
            elif isinstance(question, set) or isinstance(question, list):
                question = " ".join(question)
            if passages == "":
                d['history'][f"step {self.step}-leaf"] = "None"
                continue
            inputs = f'''{passages}\nQuestion: {question}\nYour response:'''
            if self.args.add_demon:
                inputs = f'''{self.demons["leaf"]}\n\n{inputs}'''
            inputs = f'''{this_instruct}\n\n{inputs}'''
            self.leaf_text[ori_question] = inputs
        return
    def promptLeafForward(self, questions):
        self.leaf_forward_text = {}
        for index, d in enumerate(self.data):
            if "final_res" in d or "fail" in d:
                continue
            ori_question = d['question']
            question = questions[ori_question]
            inputs = f'''{self.instruction_leaf_forward}\nQuestion: {question}'''
            self.leaf_forward_text[ori_question] = inputs
        return
    
    def promptQuery(self, missing_info):
        self.query_text = defaultdict(str)
        this_instruct = f"{self.instruction_query_head} {self.instruction_query_tail}"
        for index, d in enumerate(self.data):
            ori_question = d['question']
            if "final_res" in d or "fail" in d:
                continue
            if ori_question not in missing_info:
                continue
            inputs = f'''Original Question: {ori_question}'''
            if len(self.question_list[ori_question]) > 0:
                historical_q = " ".join(self.question_list[ori_question])
            else:
                historical_q = "None"
            inputs = f'''{inputs}\nHistorical Questions: {historical_q}'''
            if len(self.known_info[ori_question]) == 0:
                known_info = "None"
            else:
                known_info = "; ".join(self.known_info[ori_question])
            inputs = f'''{inputs}\nKnown Information: {known_info}'''
            inputs = f'''{inputs}\nMissing Information: {missing_info[ori_question]}\nNew queries:'''
            if self.args.add_demon:
                inputs = f'''{self.demons["query"]}\n\n{inputs}'''
            inputs = f'''{this_instruct}\n\n{inputs}'''
            self.query_text[ori_question] = inputs
        return
    def QueryForward(self, ret_res, questions, raw_ret_res=None):
        if self.args.gpt_knowledge:
            query_missed = {}
            for ori_q, p in ret_res.items():
                q = questions[ori_q][0]
                if len(p) < 1:
                    prompt_query_forward = f"Question: {q}\nInformation:"
                    if self.args.add_demon:
                        prompt_query_forward = f'''{self.demons['forward']}\n\n{prompt_query_forward}'''
                    prompt_query_forward = f"{self.instruction_query_forward}\n\n{prompt_query_forward}"
                    query_missed[ori_q] = prompt_query_forward
                    if raw_ret_res is not None:
                        ## recording the retrieval res, even though it's filtered
                        self.retrieval_res[q] = raw_ret_res[ori_q]
            if len(query_missed) > 0:
                print("Prompt GPT to generate relevant information.")
                res = self.askGPT(query_missed) #res: {ori_question: [gpt_generated_res]}
                for d in self.data:
                    if d['question'] in res:
                        d["history"][f"step {self.step}-QueryForward"] = {"input": query_missed[d['question']], 'output': res[d['question']][0]}
                    else:
                        d["history"][f"step {self.step}-QueryForward"] = "None"
                self.step += 1
                for ori_q, p in ret_res.items():
                    if ori_q in res:
                        ret_res[ori_q].extend(res[ori_q])
        else:
            remain_ret_res = defaultdict(list)
            for ori_q, p in ret_res.items():
                if len(p) < 1:
                    continue
                else:
                    remain_ret_res[ori_q] = p
            ret_res = remain_ret_res
        return ret_res

    def search_bm25(self, question, searcher_index=0, rerank=True):
        if isinstance(self.searcher, list):
            searcher = self.searcher[searcher_index]
        else:
            searcher = self.searcher
        map_dict = {}
        original_questions = question.copy()
        for k, v in question.items():
            map_dict_sub = {v[i]: k for i in range(len(v))}
            map_dict = dict(map_dict, **map_dict_sub)

        question = [{'query': v} for k, v in question.items()]

        num_process = self.args.num_process
        pool = multiprocessing.pool.ThreadPool(processes=num_process)
        sampleData = [x for x in range(num_process)]
        search_all_part = partial(search_all,
                                searcher=searcher,
                                num_process=num_process,
                                args=self.args,
                                data=question)
        results = pool.map(search_all_part, sampleData)
        pool.close()

        output_data = []
        for result in results:
            output_data.extend(result)
        output_data_new = {}
        for d in output_data:
            new_query = d['query']
            if isinstance(new_query, list) and len(new_query) > 0:
                new_query = new_query[0]
            ori_question = map_dict[new_query]
            output_data_new[ori_question] = d['ctxs']
        if rerank:
            ret_res = self.reranker.sentenceRerank(output_data_new, original_questions, self.history_passages)
            ret_res = self.QueryForward(ret_res, original_questions, output_data_new)
        else:
            ret_res = None

        return ret_res, output_data_new
        
    def search_bing(self, questions, searcher_index=0, rerank=True):
        all_web_knowledge = {}
        for ori_question, search_term in tqdm(questions.items()):
            headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
            params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
            response = requests.get(self.search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            web_knowledge = search_results['webPages']['value']
            all_web_knowledge[ori_question] = web_knowledge

        return all_web_knowledge

    def search_dense(self, questions, searcher_index=0, rerank=True):
        if isinstance(self.searcher, list):
            searcher = self.searcher[searcher_index]
        else:
            searcher = self.searcher
        ori_questions = list(questions.keys())
        new_queries = list(questions.values())
        queries_length = [len(query) for query in new_queries]
        all_psg_ids = []
        for i in range(0, len(new_queries), self.args.num_process):
            batch = []
            batch_list = new_queries[i: i + self.args.num_process]
            for b in batch_list:
                batch.extend(b)
            psg_ids = searcher.get_top_docs(batch, top_docs=args.num_dense)
            all_psg_ids.extend(psg_ids)
        passages = defaultdict(list)
        new_all_psg_ids = []
        previous_index = 0
        for length in queries_length:
            psg_extended = []
            psg_ids_question = all_psg_ids[previous_index: previous_index + length]
            for ids in psg_ids_question:
                psg_extended.extend(ids)
            new_all_psg_ids.append(psg_extended)
            previous_index += length
            
        assert len(new_all_psg_ids) == len(ori_questions)
        for index, ids in enumerate(new_all_psg_ids):
            ori_question = ori_questions[index]
            record_passages = set()
            for id_ in ids:
                id_ = str(id_)
                if isinstance(self.wikiData, dict):
                    title = self.wikiData[id_]['title']
                    if "text" in self.wikiData[id_]:
                        text = self.wikiData[id_]['text'].strip().rstrip()
                    else:
                        text = self.wikiData[id_]["contents"].strip().rstrip()
                else:
                    id_ = max(int(id_) - 1, 0)
                    title = self.wikiData.loc[id_]['title']
                    if "text" in self.wikiData.loc[id_]:
                        text = self.wikiData.loc[id_]['text'].strip().rstrip()
                    else:
                        text = self.wikiData.loc[id_]["contents"].strip().rstrip()
                if text not in record_passages:
                    record_passages.add(text)
                    passages[ori_question].append({'title': title, 'text': text})
        if rerank:
            ret_res = self.reranker.sentenceRerank(passages, questions, self.history_passages) if rerank else passages
            ret_res = self.QueryForward(ret_res, questions)
        else:
            ret_res = None
        return ret_res, passages
    def search_mixture(self, questions, rerank=True):
        psgs_bm25, raw_bm25 = self.search_bm25(questions, searcher_index=0, rerank=False)
        psgs_dense, raw_dense = self.search_dense(questions, searcher_index=1, rerank=False)
        psgs = {}
        for k, v in raw_bm25.items():
            psg_set = set()
            psg_list = []
            v_all = v + raw_dense[k]
            for ctx in v_all:
                if isinstance(ctx, dict):
                    text = ctx['text']
                else:
                    text = ctx
                if text in psg_set:
                    continue
                else:
                    psg_set.add(text)
                    psg_list.append(ctx)
            psgs[k] = psg_list
        if rerank:
            psgs_final = self.reranker.sentenceRerank(psgs, questions, self.history_passages) if rerank else psgs
            psgs_final = self.QueryForward(psgs_final, questions)
        else:
            psgs_final = None
        return psgs_final, psgs

    def search(self, questions, rerank=True):
        ## question: {ori_question: new_query}
        if self.args.search_type == "bm25":
            return self.search_bm25(questions, rerank=rerank)
        elif self.args.search_type == "bing":
            return self.search_bing(questions, rerank=rerank)
        elif self.args.search_type == "mixture":
            return self.search_mixture(questions, rerank=rerank)
        else:
            return self.search_dense(questions, rerank=rerank)
    
    def askGPT(self, inputs, num_return=1):
        if self.args.model_name_or_path is None:
            all_gpt_outs = gpt_batch_request(inputs, num_workers=50, model_name=self.model_name, num_return=num_return)
            ## gpt might fail to call api, we need to filter it out
            for d in self.data:
                ori_question = d['question']
                if ori_question not in all_gpt_outs:
                    continue
                if all_gpt_outs[ori_question] == "":
                    all_gpt_outs.pop(ori_question)
                    d['fail'] = True
        else:
            all_gpt_outs = self.chatModel.inference(inputs)
        return all_gpt_outs
    
    def extractNewinfo(self, all_gpt_out, info_type="main", strict=True, iteration=1):
        # all_gpt_out = {ori_question: gpt_out}
        fail_save_path = self.args.save_path.split(".json")[0] + "_fail.json"
        f = open(fail_save_path, 'a', encoding='utf-8')
        nums, leaf_nums = 0, 0
        all_queries = {}
        for data_index, d in enumerate(self.data):
            ori_question = d['question']
            if ori_question not in all_gpt_out:
                continue
            gpt_outs = all_gpt_out[ori_question]
            try:
                if info_type not in ['main', 'leaf', 'query']:
                    raise ValueError
                if info_type == "main":
                    main_record = []
                    for gpt_out in gpt_outs:
                        sub_extract_res, not_json = self.extractMain(gpt_out)
                        main_record.append(sub_extract_res)
                    if self.args.num_return == 1:
                        extract_res = main_record[0]
                    else:
                        extract_res = multi_return_process(main_record, type_module="main")
                    answer_text = extract_res['answer']['text']
                    answer_text = str(answer_text)
                    if "confidence" in extract_res['answer']:
                        confidence = extract_res['answer']['confidence']
                    else:
                        confidence = 1
                        extract_res['answer']['confidence'] = confidence
                    missing_info = extract_res["missing_information"]
                    explanation = extract_res["explanation"]
                    if "unanswerable" not in answer_text.lower():
                        d['history']["step-last-main"] = {"input": self.main_text[ori_question], "output": gpt_outs, "voted": extract_res}
                        d['final_res'] = extract_res
                        d['extra_call'] = self.extra_call
                        d['known information'] = self.known_info[ori_question]
                        d['asked_questions'] = self.question_list[ori_question]
                        d['retrieval_trace'] = self.retrieval_trace[ori_question]
                        d['main_passages'] = self.main_passages[ori_question]
                    else:
                        missing_info = explanation if len(missing_info) < 10 else missing_info
                        self.missing_info[ori_question] = missing_info
                    all_queries[ori_question] = extract_res
                    
                elif info_type == "leaf":
                    leaf_records = []
                    useful_passage_index = []
                    for gpt_out in gpt_outs:
                        if "useful_information" not in gpt_out:
                            continue
                        sub_info_extracted = self.extractLeaf(gpt_out)
                        leaf_records.extend(sub_info_extracted)
                    if self.args.num_return == 1:
                        useful_info = leaf_records
                    else:
                        useful_info = multi_return_process(leaf_records, type_module="leaf")
                    for info in useful_info:
                        if "info" in info and "support_passages" in info:
                            if len(info['support_passages']) > 0:
                                nli_score = 1
                                if self.nli_model is not None:
                                    nli_score = 0
                                    for i, p in enumerate(self.leaf_passages[ori_question]):
                                        if i in info["support_passages"]:
                                            if isinstance(p, list):
                                                context, score, question, title, ori_context = p
                                                p = f"(Title: {title}) {context}"
                                            nli_score = verify_info(info["info"], p, model=self.nli_model, tokenizer=self.nli_tokenizer)
                                        if nli_score == 1:
                                            break
                                if nli_score == 1:
                                    if str(info["info"]) not in self.known_info[ori_question]:
                                        self.known_info[ori_question].append(str(info["info"]))
                                        useful_passage_index.extend(info["support_passages"])
                    useful_passage_index = set(useful_passage_index)
                    for i, p in enumerate(self.leaf_passages[ori_question]):
                        if i in useful_passage_index:
                            # print("*****    Main passages added *****")
                            self.main_passages[ori_question].append(p)
                            if isinstance(p, dict):
                                text = p['snippet'] if 'snippet' in p else p['text']
                            elif isinstance(p, list):
                                text = p[-1]
                            else:
                                text = p
                            self.history_passages[ori_question][text] += 1
                    all_queries[ori_question] = useful_info
                else:
                    query = [out.strip().rstrip() for out in gpt_outs]
                    valid_query = []
                    query = query[0].split("\n")
                    query = [q.strip().rstrip() for q in query]
                    for q in query:
                        if len(q) > 10 and q not in self.question_list[ori_question]:
                            valid_query.append(q)
                    if len(valid_query) == 0:
                        valid_query.append(ori_question)
                    all_queries[ori_question] = valid_query
            except:
                nums += 1
                d['fail'] = True
                if info_type == "main":
                    this_type = self.main_text
                elif info_type == "leaf":
                    this_type = self.leaf_text
                else:
                    this_type = self.query_text
                fail_record = {'question': ori_question, "type": info_type, "input": this_type[ori_question], "output": gpt_outs, "main_passages": self.main_passages[ori_question]}
                f.write(json.dumps(fail_record, indent=5, ensure_ascii=False))
        f.close()
        return all_queries
                          
    def extractUseful(self, span_list, strict=True):
        # res_dict = {'useful info': [], "support_passages": []}
        res = []
        for index, t in enumerate(span_list[: -1]):
            useful_info = {"info": None, "support_passages": []}
            find = False
            support_passages = span_list[index + 1]
            if '''"info:"''' in t:
                index = t.find('''"info":''') + 8
            else:
                index = t.find('''"useful_information"''') + 22
            if index != -1:
                useful_info["info"] = t[index:].replace('''"''', "").strip().rstrip()
            if not useful_info["info"]:
                continue
            index = support_passages.find(''"support_passages"'')
            if index != -1:
                for str_ in support_passages[index + 18: ]:
                    if str_.isdigit():
                        find = True
                        useful_info["support_passages"].append(int(str_))
                    if str_ == "]":
                        break
            if find:
                res.append(useful_info)
            elif not strict:
                remain = True
                keywords_refuse = ["not provided", "not given", "not mentioned", "not known"]
                for k in keywords_refuse:
                    if k in useful_info["info"].lower():
                        remain = False
                        break
                if remain:
                    res.append(useful_info)
        return res
    def extractMain(self, gpt_out, key='explanation'):
        if isinstance(gpt_out, list):
            gpt_out = gpt_out[0]
        not_json = False
        useful_passage_index = []
        missing_info, answer_text, answer_confidence, explanation_info = "", "", "", ""
        try:
            record_res = eval(gpt_out)
            if "missing_information" not in record_res:
                record_res["missing_information"] = "None"
        except:
            not_json = True
            sequence_dict, sequence_key = {}, ['''"missing_information":''', '''"answer":''', '''"useful_information":''', '''"explanation":''']
            for sub_key in sequence_key:
                try:
                    sequence_dict[sub_key] = gpt_out.find(sub_key)
                except:
                    pass
            sequence_dict = sorted(sequence_dict.items(), key=lambda x: x[1])
            sequence_order = [seq[0] for seq in sequence_dict]
            if "missing_information" in gpt_out:
                missing_info_split = sequence_order[(sequence_order.index('''"missing_information":''') + 1) % len(sequence_order)]
                missing_info = gpt_out.split('''"missing_information":''')[-1].split(missing_info_split)[0].replace('''"''', "").strip().rstrip()
            else:
                missing_info = ""
            if "answer" in gpt_out:
                answer_text_split = sequence_order[(sequence_order.index('''"answer":''') + 1) % len(sequence_order)]
                answer_span = gpt_out.split('''"answer":''')[-1].split(answer_text_split)[0]
                answer_text = answer_span.split('''"text":''')[-1].split('''"confidence":''')[0].strip().rstrip()
                try:
                    answer_confidence = int(answer_span.split('''"confidence":''')[-1].split('''}''')[0])
                except:
                    answer_confidence = -1
            if "explanation" in gpt_out:
                explanation_info_split = sequence_order[(sequence_order.index('''"explanation":''') + 1) % len(sequence_order)]
                explanation_info = gpt_out.split('''"explanation":''')[-1].split(explanation_info_split)[0].replace('''"''', "").strip().rstrip()
            
            if answer_text.endswith(","):
                answer_text = answer_text[:-1]
            if missing_info.endswith(","):
                missing_info = missing_info[:-1]
            
            if explanation_info.endswith(","):
                explanation_info = explanation_info[:-1]
                
            if "support_passages" in gpt_out:
                support_passages =  gpt_out.split('''"support_passages":''')[-1]
                for str_ in support_passages:
                    if str_ == "]":
                        break
                    if str_.isdigit():
                        useful_passage_index.append(int(str_))
            explanation = gpt_out.split(f'''"{key}":''')[-1][:-1]
            record_res = {'answer': {"text": answer_text, 'confidence': answer_confidence}, "missing_information": missing_info, "explanation": explanation, "support_passages": useful_passage_index}
        return record_res, not_json

    def extractLeaf(self, gpt_out):
        try:
            gpt_out = eval(gpt_out)
            gpt_out = gpt_out["useful_information"]
            record_res = []
            for info in gpt_out:
                if "support_passages" not in info:
                    continue
                elif len(info["support_passages"]) == 0:
                    continue
                else:
                    record_res.append(info)
        except:
            gpt_out = gpt_out.split('''}, {''')
            record_res = self.extractUseful(gpt_out, strict=True)
        return record_res
        
    def _iteration(self):
        ## initialize the main text
        ori_questions = {d['question']: [d['question']] for d in self.data}
        num_iter = 0

        if self.args.max_iter > 0:
            #self.args.add_demon or self.args.leaf_first:
            ranked_leaf_passages, raw_leaf_passages = self.search(ori_questions)
            for k, v in ranked_leaf_passages.items():
                self.leaf_passages[k] = v
                self.retrieval_trace[k].extend(v)
            self.promptLeaf(ori_questions)
            leaf_gpt_out = self.askGPT(self.leaf_text, num_return=self.args.num_return)
            leaf_extraction = self.extractNewinfo(leaf_gpt_out, info_type='leaf')
            for d in self.data:
                ori_question = d['question']
                if "final_res" in d or "fail" in d:
                    continue
                if ori_question not in self.leaf_text:
                    continue
                d['history']["step 0-leaf"] = {
                    "input": self.leaf_text[ori_question],
                    "output": leaf_gpt_out[ori_question],
                    "voted": leaf_extraction[ori_question]
                }
            self.promptMain(num_iter=1)
        else:
            if "first_retrieval" in self.data[0]:
                for d in self.data:
                    self.main_passages[d['question']] = d['first_retrieval']
            else:
                ranked_main_passages, raw_main_passages = self.search(ori_questions)
                for k, v in ranked_main_passages.items():
                    self.main_passages[k] = v
                    self.retrieval_trace[k].extend(v)
                for d in self.data:
                    if self.args.search_type == "bing":
                        d['first_retrieval'] = raw_main_passages[d['question']]
            self.promptMain()
        leaf_retrieval_record = []
        # recording history
        self.step = 1
        
        while True:
            # print(f"Current iteration self.step: {num_iter}")
            if num_iter >= self.max_iter:
                main_gpt_out = self.askGPT(self.main_text, num_return=self.args.num_return)
                for d in self.data:
                    if "final_res" in d or "fail" in d:
                        continue
                    ori_question = d['question']
                    if ori_question not in main_gpt_out:
                        continue
                    if self.args.num_return == 1:
                        extract_res, not_json = self.extractMain(main_gpt_out[ori_question], key='explanation')
                    else:
                        main_record = []
                        for sub_gpt_out in main_gpt_out[ori_question]:
                            sub_extract_res, not_json = self.extractMain(sub_gpt_out, key='explanation')
                            main_record.append(sub_extract_res)
                        extract_res = multi_return_process(main_record, type_module="main")
                    d['history']["step-last-main"] = {
                        "input": self.main_text[ori_question], 
                        "output": main_gpt_out[ori_question], 
                        "voted": extract_res
                    }
                    d['extra_call'] = self.extra_call
                    d['final_res'] = extract_res
                    d['known information'] = self.known_info[ori_question]
                    if self.max_iter == 0:
                        support_passages = extract_res['support_passages']
                        new_main_passages = []
                        main_passages = self.main_passages[ori_question]
                        for i, p in enumerate(main_passages):
                            if i in support_passages:
                                new_main_passages.append(p)
                        d['main_passages'] = new_main_passages
                    else:
                        d['main_passages'] = self.main_passages[ori_question]
                    d['asked_questions'] = self.question_list[ori_question]
                    d['retrieval_trace'] = self.retrieval_trace[ori_question]
                break
            else:
                num_iter += 1
            print(f"step: {self.step}: Prompt Main")
            main_gpt_out = self.askGPT(self.main_text, num_return=self.args.num_return)
            main_extraction = self.extractNewinfo(main_gpt_out, info_type='main', strict=num_iter==1, iteration=num_iter)
            for d in self.data:
                ori_question = d['question']
                if "final_res" in d or "fail" in d:
                    continue
                if ori_question not in self.main_text:
                    continue
                d['history'][f"step {self.step}-main"] = {
                    "input": self.main_text[ori_question],
                    "output": main_gpt_out[ori_question],
                    "voted": main_extraction[ori_question]
                }

            self.step += 1
            ## lack of knowledge, we first generate a query to perform retrieval
            self.promptQuery(self.missing_info)
            print(f"step: {self.step}: Generate query for knowledge retrieval")
            query_gpt_out = self.askGPT(self.query_text)
            new_query = self.extractNewinfo(query_gpt_out, info_type='query')
            
            for d in self.data:
                if d['question'] in self.missing_info and d['question'] in new_query:
                    d['Missing info & query generated'].append({"missing info": self.missing_info[d['question']], "query": new_query[d['question']], "known info": self.known_info[d['question']]})

            for d in self.data:
                ori_question = d['question']
                if "final_res" in d or "fail" in d:
                    continue
                if ori_question not in self.query_text:
                    continue
                d['history'][f"step {self.step}-query"] = {"input": self.query_text[ori_question], "output": query_gpt_out[ori_question]}
            self.step += 1
            print(f"step: {self.step}: Jump to the leaf for knowledge retrieval")
            ranked_leaf_passages, raw_leaf_passages = self.search(new_query)
            for k, v in ranked_leaf_passages.items():
                self.leaf_passages[k] = v
                self.retrieval_trace[k].extend(v)

            self.promptLeaf(new_query)
            leaf_gpt_out = self.askGPT(self.leaf_text, num_return=self.args.num_return)
            leaf_extraction = self.extractNewinfo(leaf_gpt_out, info_type='leaf', strict=True)
            for d in self.data:
                ori_question = d['question']
                if "final_res" in d or "fail" in d:
                    continue
                if ori_question not in self.leaf_text:
                    continue
                d['history'][f"step {self.step}-leaf"] = {
                    "input": self.leaf_text[ori_question], 
                    "output": leaf_gpt_out[ori_question], 
                    "voted": leaf_extraction[ori_question]
                }
            self.step += 1
            self.promptMain(num_iter)

        return
    def retrieval(self):
        ori_questions = {d['question']: [d['question']] for d in self.data}
        print(f"Processing retrieval, total data {len(self.data)}")
        self.main_passages, raw_main_passages = self.search(ori_questions, rerank=False)
        ## get retrieval results after rerank
        for d in self.data:
            d['retrieval_res'] = raw_main_passages[d['question']]
        with open(self.args.save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.data, indent=5, ensure_ascii=False))
        return

parser = argparse.ArgumentParser(description='')
parser.add_argument("--index_dir", type=str,
                    default=None,
                    help="directory to store the search index built with BM25")
parser.add_argument("--num_process", type=int, default=1,
                    help="number of processes to use for multi-threading")
parser.add_argument("--top_k", type=int, default=2,
                    help="number of passages to be retrieved for each query")
parser.add_argument("--bm25_b", type=float, default=0.4,
                    help="parameter of BM25")
parser.add_argument("--k1", type=float, default=0.9,
                    help="parameter of BM25")
parser.add_argument('--local_rank', default=-1)
parser.add_argument('--local-rank', default=-1)
parser.add_argument("--search_type", type=str, default="bm25",
                    help="default search method, choose between bm25, bing search and dense retrieval",
                    choices=['bm25', 'bing', 'dense', 'mixture'])
parser.add_argument("--data_path", type=str,
                    required=True, help="dev data path")
parser.add_argument("--query_prompt", type=str, default="")
parser.add_argument("--num_process_data", type=float, default=float('inf'))
parser.add_argument("--save_path", type=str,
                    required=True, help="save results path")
parser.add_argument("--retrieval", action="store_true")
parser.add_argument("--entail_judge", action="store_true")
parser.add_argument("--encoder_path", type=str, default=None)
parser.add_argument("--encoded_data_path", type=str, default=None)
parser.add_argument("--model_name", type=float, default=3.5)
parser.add_argument("--num_return", type=int, default=1)
parser.add_argument("--ranker_model_path", type=str, default="BAAI/bge-rerank-base")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--add_demon", action="store_true")
parser.add_argument("--sentRerank", action="store_true")
parser.add_argument("--gpt_knowledge", action="store_true")
parser.add_argument("--aug", action="store_true", help="Utilizing the oracle retrieval corpus.")
parser.add_argument("--summ_snippet", type=str, default=None)
parser.add_argument("--max_iter", type=int, default=5)
parser.add_argument("--num_dense", type=int, default=30)
parser.add_argument("--num_bm25", type=int, default=50)
parser.add_argument("--model_name_or_path", type=str, default=None)
parser.add_argument("--demon_path", type=str, default=None)
parser.add_argument("--relevance", type=float, default=3.0, help="The relevance threshold delta.")
parser.add_argument("--ctx_sources", type=str, default=None)

args = parser.parse_args()

if __name__ == "__main__":
    if args.aug:
        bm25_dict = {
            "wikimultihop": "bm25_index_of_augmented_corpus",
            "odqa": "bm25_index_of_augmented_corpus",
            'hotpot': "bm25_index_of_augmented_corpus"",
            "musique": "bm25_index_of_augmented_corpus",
            "strategy": "bm25_index_of_augmented_corpus"
        }
    else:
        bm25_dict = {
            "wikimultihop": "bm25_index_of_original_corpus",
            "odqa": "bm25_index_of_original_corpus",
            'hotpot': "bm25_index_of_original_corpus"",
            "musique": "bm25_index_of_original_corpus",
            "strategy": "bm25_index_of_original_corpus"
        }
    dense_dict = {
        "wikimultihop": "",
        "odqa": "",
        "hotpot": "",
        "musique": "",
        "strategy": ""
    }
    source_dict = {
        "wikimultihop": "2018_wikipedia_dump",
        "odqa": "2018_wikipedia_dump",
        "hotpot": "2017_wikipedia_dump",
        "musique": "2021_wikipedia_dump",
        "strategy": "2021_wikipedia_dump"
    }
    for k in bm25_dict.keys():
        if k in args.data_path:
            args.index_dir = bm25_dict[k]
            args.encoded_data_path = dense_dict[k]
            args.ctx_sources = source_dict[k]
    print(args)
    if args.add_demon:
        assert args.demon_path is not None
    args.query_prompt = "Generate a representation for this sentence to be used to retrieve related documents: " if "bge" in args.encoder_path else args.query_prompt
    choosed_model_name = "gpt-4-0613" if args.model_name == 4 else "gpt-3.5-turbo-1106"
    print("Backbone model: ", choosed_model_name)
    if not args.retrieval:
        save_dir = args.save_path.split("output_")[0]
        os.makedirs(save_dir, exist_ok=True)
    finished_data =[]
    try:
        data = json.load(open(args.data_path))
    except:
        data = open(args.data_path).readlines()
        data = [json.loads(d) for d in data]
    if isinstance(data, dict):
        data = data["data"]
    args.num_process_data = int(min(args.num_process_data, len(data)))
    data = data[: args.num_process_data]
    if os.path.exists(args.save_path):
        try:
            res_finished = json.load(open(args.save_path))
        except:
            res_finished = open(args.save_path).readlines()
            res_finished = [json.loads(d) for d in res_finished]
        finished_id = set()
        for d in res_finished:
            if "final_res" in d:
                q = d['question']
                if "fail" in d:
                    fail = d.pop("fail")
                finished_id.add(q)
                finished_data.append(d)
            if "fail" in d:
                fail = d.pop("fail")
    else:
        finished_id = []
    # f = open(args.save_path, 'a', encoding='utf-8')
    unfinished_data = []
    for d in data:
        q = d['question']
        if q in finished_id:
            continue
        if "prompt_to_ask" in d:
            text = d.pop("prompt_to_ask")
        unfinished_data.append(d)
    print("Total data: ", len(unfinished_data))
    if len(unfinished_data) < 1:
        print("All queries solved")
        exit()
    all_passages = None
    searcher = []
    if args.search_type == "mixture":
        args.num_bm25 = 20
        args.num_dense = 30
    if args.search_type == "bm25" or args.search_type == "mixture":
        print("Building bm25 search engine")
        searcher.append(Bm25Searcher(args.index_dir, args))
        print("bm25 search engine built")
    if args.search_type == "dense" or args.search_type == "mixture":
        assert args.encoded_data_path is not None and args.encoder_path is not None
        print("Loading Wikipedia passages")
        all_passages = get_all_passages_multi(args.ctx_sources)
        print("Wikipedia passages loaded")
        list_dirs = os.listdir(args.encoded_data_path)
        paths = []
        for p in list_dirs:
            if p.endswith(".pkl"):
                paths.append(os.path.join(args.encoded_data_path, p))
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)
        encoder = AutoModel.from_pretrained(args.encoder_path, device_map='cuda')
        index = DenseFlatIndexer()
        vector_size = 768
        index.init_index(vector_size)
        index_buffer_sz = index.buffer_size
        dense_searcher = LocalFaissRetriever(args, encoder, 64, tokenizer, index=index)
        dense_searcher.index_encoded_data(paths, index_buffer_sz)
        searcher.append(dense_searcher)

    if args.entail_judge:
        AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        nli_component = [autoais_model, autoais_tokenizer]
    else:
        nli_component = None
    iter_data = RetGBatched(unfinished_data, args, searcher=searcher, model_name=choosed_model_name, wikiData=all_passages, nli_component=nli_component, world_size=world_size)
    if args.retrieval:
        iter_data.retrieval()
    else:
        iter_data._iteration()
    iter_finished_data = []
    finished_data.extend(iter_data.data)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(finished_data, indent=5, ensure_ascii=False))

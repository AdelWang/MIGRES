from pyserini.search.lucene import LuceneSearcher
import multiprocessing.pool
from functools import partial
from collections import defaultdict
from multiprocessing import Process, Pool
import json
import nltk
import faiss
import logging
import numpy as np
import os
import pickle
import socket
import torch.nn as nn

from typing import List, Tuple, Dict, Iterator
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import pickle
import time
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import deepspeed

from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
import torch
import torch.nn.functional as F

logger = logging.getLogger()

def get_all_passages(all_passages_paths):
    all_passages = {}
    for p in tqdm(all_passages_paths):
        passages_data = open(p).readlines()
        passages_data = [json.loads(d) for d in passages_data]
        for d in passages_data:
            id_ = str(d['id'])
            title = d['title']
            text = d['text'] if "text" in d else d["contents"]
            all_passages[id_] = {'title': title, 'text': text}
    return all_passages

def get_all_passages_multi(all_data, num_workers=32):
    if os.path.isdir(all_data):
        paths = os.listdir(all_data)
        paths = [os.path.join(all_data, p) for p in paths]
        all_data = paths
    else:
        return pd.read_table(all_data)
    sub_data = []
    res_all = []
    sub_length = len(all_data) // num_workers
    for w in range(num_workers):
        sub_wikidata = all_data[w * sub_length: (w + 1) * sub_length]
        sub_data.append(sub_wikidata)
    pool = Pool(num_workers)
    for j in range(num_workers):
        res = pool.apply_async(func=get_all_passages, args=(sub_data[j],))
        res_all.append(res)
    final_dict = {}
    for r in res_all:
        dict_sub = r.get()
        final_dict = dict(final_dict, **dict_sub)
    pool.close()
    pool.join()
    return final_dict

def verify_info(info, context, model, tokenizer):
    text = f"premise: {context} hypothesis: {info}"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference

class Bm25Searcher:
    def __init__(self, index_dir, args):
        self.index_dir = index_dir
        self.args = args
        try:
            self.searcher = LuceneSearcher(index_dir)
        except:
            print("index dir not found")
            self.searcher = LuceneSearcher.from_prebuilt_index(index_dir)
        self.searcher.set_bm25(args.k1, args.bm25_b)
        if len(args.ignore_string) > 0:
            self.ignore_list = args.ignore_string.split(',')
            print(f'ignore list: {self.ignore_list}')
        else:
            self.ignore_list = []

    def perform_search(self, data_i, top_k):
        # queries: List[str]
        queries = data_i["query"]
        all_results = []
        if isinstance(queries, str):
            queries = [queries]
        for query in queries:
            for string in self.ignore_list:
                query = query.replace(string, ' ')
            query = query.strip()
            results = self.searcher.search(query, k=top_k, strip_segment_id=False, remove_dups=False)
            if len(results) < top_k:
                query = 'Find information about ' + query
                results = self.searcher.search(query, k=top_k, strip_segment_id=False, remove_dups=False)
            all_results.append(results)

        ctxs = []
        context_set = set()
        for results in all_results:
            for result in results:
                doc_dict = json.loads(result.raw)
                ctx_text = doc_dict["contents"]
                if ctx_text not in context_set:
                    ctx = {"title": doc_dict["title"], "text": ctx_text, "score": result.score}
                    ctxs.append(ctx)
                    context_set.add(ctx_text)
        output_i = data_i.copy()
        output_i["ctxs"] = ctxs
        return output_i


def search_all(process_idx, num_process, searcher, data, args):
    output_data = []
    for i, data_i in enumerate(data):
        if i % num_process != process_idx:
            continue
        if i > args.num_queries and args.num_queries != -1:
            break

        output_i = searcher.perform_search(data_i, args.num_bm25)
        output_data.append(output_i)
    return output_data

def iterate_encoded_files(vector_files: list) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                yield doc

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)
        logger.info("Serializing Finished")

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        
    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            # self.index.add_with_ids(vectors, db_ids)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"

class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tokenizer: BertTokenizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tokenizer,
            questions,
            bsz,
            query_token=query_token,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        args,
        question_encoder: nn.Module,
        batch_size: int,
        tokenizer: BertTokenizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tokenizer)
        self.index = index
        self.args = args

    def index_encoded_data(
        self,
        vector_files: str,
        buffer_size: int,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        print("Indexing encoded document.")
        for item in tqdm(iterate_encoded_files(vector_files)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, queries, top_docs=5):
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        if isinstance(queries, str):
            queries = [queries]
        prompt = self.args.query_prompt
        queries = [prompt + q for q in queries]
        inputs = self.tokenizer(queries, max_length=64, truncation=True, padding=True, return_tensors='pt').to(self.question_encoder.device)
        with torch.no_grad():
            query_vectors = self.question_encoder(**inputs).last_hidden_state[:, 0, :]
        if self.args.normalize:
            query_vectors = F.normalize(query_vectors, p=2, dim=1)
        query_vectors = query_vectors.cpu().numpy()
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        psg_ids = [results[i][0] for i in range(len(results))]
        # scores = results[0][1]
        # print("*********  dense res check:", psg_ids, '\n', scores)
        return psg_ids

class Reranker:
    def __init__(self, args):
        self.args = args
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.ranker_model_path, device_map="auto")
        # self.model, *_ = deepspeed.initialize(model=model, config=deepspeed_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.ranker_model_path)
        self.sent_tokenizer = nltk.data.load('tokenization/punkt/english.pickle')
 
    def process_data(self, this_question, passages, new_questions):
        pairs = []
        questions = new_questions[this_question]
        passages = [ctx['title'] + ctx['text'] for ctx in passages]
        for p in passages:
            for q in questions:
                pairs.append([q, p])
        return pairs

    def calRelevance(self, pairs):
        all_scores = []
        all_sorted_indices = []
        for pair in tqdm(pairs):
            inputs = self.tokenizer(pair, max_length=512, padding=True, truncation=True, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = outputs.logits.view(-1, ).float()
            sorted_indices = torch.argsort(scores, descending=True)
            all_scores.append(scores)
            all_sorted_indices.append(sorted_indices)
        return all_scores, all_sorted_indices

    def rank(self, psgs, questions):
        new_psgs = defaultdict(list)
        for q, p in tqdm(psgs.items()):
            num_questions = len(questions[q])
            input_pairs = self.process_data(q, p, questions)
            inputs = self.tokenizer(input_pairs, max_length=512, truncation=True, padding=True, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                scores = self.model(**inputs).logits.view(-1, ).float()
            scores = scores.reshape(num_questions, -1).sum(0)
            sorted_indices = torch.argsort(scores, descending=True)
            for index in sorted_indices:
                index = int(index)
                ranked_psg = psgs[q][index]
                new_psgs[q].append(ranked_psg)
        return new_psgs

    def sentenceRerank(self, ret_res, questions, historical_choosed):
        all_choosed = defaultdict(list)
        print("Reranking the retrieval doc.")
        for ori_question, raw_context in tqdm(ret_res.items()):
            context = [
                {"title": c['title'], "text": c["text"].replace("\n", "").replace("<br>", "").replace("<b>", "").replace("</b>", "")}
                for c in raw_context
            ]
            # print(len(context))
            context_title_map = {}
            for c in context:
                context_title_map[c["text"]] = c["title"]
            generated_questions = questions[ori_question]
            all_choosed[ori_question] = []
            choosed_context, choosed_snippet = {}, {}
            snippet_context_map = {}
            for question in generated_questions:
                pairs = [[question, f"{c['title']} {self.tokenizer.sep_token} {c['text']}"] for c in context]
                inputs = self.tokenizer(pairs, max_length=512, padding=True, truncation=True,return_tensors='pt').to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                scores = outputs.logits.view(-1, ).float()
                sorted_indices = torch.argsort(scores, descending=True)
                for idx in sorted_indices:
                    idx = int(idx)
                    score = float(scores[idx])
                    ## We filter the knowledge with relevance smaller than threshold outside the rerank
                    ## outside the rerank (in pipeline.py) to record the retrieval trace
                    if score < self.args.relevance:
                        break
                    this_context = context[idx]['text']
                    this_title = context[idx]['title']
                    if historical_choosed[ori_question][this_context] >= 2:
                        continue
                    choosed_context[this_context] = [score, question]
            if not self.args.sentRerank:
                choosed_snippet = sorted(choosed_context.items(), key=lambda x: x[1][0], reverse=True)
                choosed_snippet = [[snippet[0]] + snippet[1] for snippet in choosed_snippet]
                choosed_snippet = [snippet + [context_title_map[snippet[0]], snippet[0]] for snippet in choosed_snippet]
                all_choosed[ori_question] = choosed_snippet
            else:
                text_question_map = {}
                for c, score in choosed_context.items():
                    this_title = context_title_map[c]
                    score, question = score
                    cutted_sentences = self.sent_tokenizer.tokenize(c)
                    cutted_sentences.append("T")
                    raw_sentences, sentences = [], []
                    previous_sentence = cutted_sentences[0]
                    for s in cutted_sentences[1:]:
                        s = s.strip().rstrip()
                        if not s[0].isupper():
                            previous_sentence = previous_sentence + " " + s
                        else:
                            raw_sentences.append(previous_sentence)
                            previous_sentence = s
                    for s in raw_sentences:
                        if s.lower() in question.lower():
                            continue
                        else:
                            context_title_map[s] = this_title
                            sentences.append(s)
                    if len(sentences) == 0:
                        continue
                    sents_pairs = [[question, f"{this_title} {self.tokenizer.sep_token} {s}"] for s in sentences]
                    inputs = self.tokenizer(sents_pairs, max_length=128, padding=True, truncation=True,return_tensors='pt').to(self.model.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    scores = outputs.logits.view(-1, ).float()
                    sorted_indices = torch.argsort(scores, descending=True)
                    gathered_sentences, gathered_score = [], []
                    idx = int(sorted_indices[0])
                    this_score = float(scores[idx])
                    if this_score < score:
                        if historical_choosed[ori_question][c] < 2:
                            choosed_snippet[c] = score
                            text_question_map[c] = question
                            snippet_context_map[c] = c
                    else:
                        for this_sentence, s_score in zip(sentences, scores):
                            s_score = float(s_score)
                            if s_score < self.args.relevance:
                                continue
                            record = True
                            for snippet in choosed_snippet:
                                if this_sentence in snippet or snippet in this_sentence:
                                    record = False
                            if record:
                                gathered_sentences.append(this_sentence)
                                gathered_score.append(score)
                    if len(gathered_sentences):
                        gathered_sentences = " ".join(gathered_sentences)
                        context_title_map[gathered_sentences] = this_title
                        gathered_score = gathered_score[0]
                        choosed_snippet[gathered_sentences] = gathered_score
                        text_question_map[gathered_sentences] = question
                        snippet_context_map[gathered_sentences] = c
                choosed_snippet = sorted(choosed_snippet.items(), key=lambda x: x[1], reverse=True)
                choosed_snippet = [list(snippet) for snippet in choosed_snippet]
                choosed_snippet = [snippet + [text_question_map[snippet[0]], context_title_map[snippet[0]], snippet_context_map[snippet[0]]] for snippet in choosed_snippet]
                all_choosed[ori_question] = choosed_snippet
        return all_choosed

class Seq2SeqDataset(Dataset):
    def __init__(self, all_data):
        ## data: {}
        all_data_collect_list = []
        for k, v in all_data.items():
            all_data_collect_list.append({"ori_question": k, "prompt_to_ask": v})
        self.examples = all_data_collect_list

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args    

    def __call__(self, batch):
        ori_questions = [b['ori_question'] for b in batch]
        input_text = [b["prompt_to_ask"] for b in batch]
        inputs = self.tokenizer(input_text, max_length=4096, padding=True, truncation=True, return_tensors='pt')

        return inputs, ori_questions
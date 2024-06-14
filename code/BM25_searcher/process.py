import json
import multiprocessing
from multiprocessing import Process, Pool
import os
from tqdm import tqdm
import math
import pandas as pd

def process_data(data, index):
    f = open(os.path.join(save_dir, f"psg_{index}.jsonl"), 'a', encoding='utf-8')
    if isinstance(data, list):
        data = [json.loads(d) for d in data]
        for d in tqdm(data):
            id_ = d['id']
            title = d['title']
            context = d['text']
            new_d = {'id': id_, "title": title, "contents": context}
            f.writelines([json.dumps(new_d, ensure_ascii=False), '\n'])
    elif isinstance(data, pd.DataFrame):
        for row in tqdm(data.index):
            context = data.loc[row]["text"]
            id_ = data.loc[row]["id"]
            title = data.loc[row]["title"]
            new_d = {'id': id_, "title": title, "contents": context}
            f.writelines([json.dumps(new_d, ensure_ascii=False), '\n'])
    f.close()
    return "Finish"

def multi_process(data_path, num_workers=32):
    if data_path.endswith("jsonl"):
        all_data = open(data_path).readlines()
    elif data_path.endswith("tsv") or data_path.endswith("csv"):
        all_data = pd.read_table(data_path)
    sub_length = len(all_data) // num_workers + 1
    pool = Pool(num_workers)
    res_all = []
    for w in range(num_workers):
        sub_wikidata = all_data[w * sub_length: (w + 1) * sub_length]
        res = pool.apply_async(func=process_data, args=(sub_wikidata, w))
        res_all.append(res)
    final_dict = []
    for r in res_all:
        sub_res = r.get()
        final_dict.append(sub_res)
    pool.close()
    pool.join()
    return final_dict

def data_split(paths, save_dir):
    if isinstance(paths, str):
        paths = [paths]
    for index, p in enumerate(paths):
        save_names = p.split(".jsonl")[0].split("/")[-1]
        data = open(p).readlines()
        data = [json.loads(d) for d in data]
        lengths = len(data) // 10 + 1 
        for i in range(10):
            this_save_name = save_names + f"_{i}.jsonl"
            f = open(os.path.join(save_dir, this_save_name), 'a', encoding='utf-8')
            for idx in range(0, len(data), lengths):
                sub_data = data[idx: idx + lengths]
                for d in tqdm(sub_data):
                    f.writelines([json.dumps(d, ensure_ascii=False), '\n'])
            f.close()
    return "Finish"

def multi_process_split(data_paths, save_dir, num_workers=32):
    pool = Pool(num_workers)
    res_all = []
    sub_length = math.ceil(len(data_paths) / num_workers)
    for w in range(num_workers):
        sub_wikidata_paths = data_paths[w * sub_length: (w + 1) * sub_length]
        res = pool.apply_async(func=data_split, args=(sub_wikidata_paths, save_dir))
        res_all.append(res)
    final_dict = []
    for r in res_all:
        sub_res = r.get()
        final_dict.append(sub_res)
    pool.close()
    pool.join()
    return final_dict

if __name__ == "__main__":
    data_path = "./wiki18/psg_100_1812.tsv"
    save_dir = "./wiki18/processed"
    os.makedirs(save_dir, exist_ok=True)
    multi_process(data_path)
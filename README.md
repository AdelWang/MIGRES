# MIGRES

### Preparation
Download retrieval corpus from https://github.com/facebookresearch/DPR (for wiki-18) and https://github.com/facebookresearch/atlas?tab=readme-ov-file#corpora (for wiki-17 & wiki-21)
```
wiki18/
 |psgs_w100.tsv
wiki17/
 |*.jsonl
wiki21/
 |*.jsonl
```
Install pyserini package (pyserini might require JAVA adaptation, plz refer to https://github.com/castorini/pyserini for details)  
`pip install pyserini`  
Utilize BM25 to index the retrieval corpus (Change the file/folder name to your corresponding path first.)  
`cd code/BM25_seacher/ && python process.py && sh BM25_build_index.sh`  

### Run MIGRES
Please use your own api_key to call OpenAI GPT (refers to def send_post_request in pipeline.py), and change the "bm25_dict" and "source_dict" in pipeline.py to your own path. Then  
`cd code/ && bash migres.sh`

### Evaluation
We utilize gpt-3.5-turbo-1106 to perform a more robust evaluation as discussed in [ITER-GEN](https://aclanthology.org/2023.findings-emnlp.620/).  
Change the api_key to call OpenAI GPT in `evaluate.py`, then run  
`python evaluate.py --data_path "Your saving path" `

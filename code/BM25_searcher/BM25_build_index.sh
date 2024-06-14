
# Build the index for the general knowledge base using pyserini.
## pyserini might require JAVA adaptation, plz refer to https://github.com/castorini/pyserini for details.

# export JAVA_HOME=$USER_PATH$/programs/jdk-17.0.7
# export PATH=$JAVA_HOME/bin:$PATH

SOURCE_DIR=./wiki18/processed
TARGET_DIR=./wiki18/processed_bm25

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${SOURCE_DIR} \
  --index ${TARGET_DIR} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw
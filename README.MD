# Sentence similarity example#
## 功能 ##
  * 按照不同算法查询相似的句子
  * 基于gensim的 lda ，lsi， rp算法
  * 基于gensim构建索引

## 程序说明
  * dataset.py 构建了字典和语料库（按照bagofword向量存储）
  * gensim_model_main.py 是根据语料库结果建立句子的模型（LDA，LSI，RP）
  * RP是向量的压缩
  * gensim_similarity_main.py 里面是针对所有的语料根据不同的模型计算出句子的向量，并且可以根据计算出的矩阵做相似查询 


  





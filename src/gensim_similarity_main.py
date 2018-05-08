# coding:utf-8  

import os
import logging
import time
import jieba

from gensim import corpora, models, similarities
from gensim_model_main import load_lad_model,load_lsi_model,load_tfidf_model,load_rp_model
from dataset import load_corpora,get_question
from utility import write_to_file

def lsi_similarity_main(dictionary,tfidf_save_path,lsi_save_path,corpora_path,index_path):
    
    questions= get_question()
    
    corpus = load_corpora(corpora_path)

    tfidfmodel = load_tfidf_model(tfidf_save_path)
    lsimodel = load_lsi_model(lsi_save_path)

    corpus_tfidf = tfidfmodel[corpus]
    
    if os.path.exists(index_path):
        index_sim = similarities.MatrixSimilarity.load(index_path)
    else:
        index_sim = similarities.MatrixSimilarity(lsimodel[corpus_tfidf]) 
        index_sim.save(index_path)
        
    write_to_file('../data/query.lsi.txt',  ''.encode('utf-8'),mode='wb+')
    for i in range(1000):
        querydoc =questions[i]
        vec_bow = dictionary.doc2bow(jieba.lcut(querydoc))
        vectfidf = tfidfmodel[vec_bow]
        vec_lsi = lsimodel[vectfidf]
        sims = index_sim[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        for sim in sims[:10]:
            index = sim[0]
            distance = sim[1]
            txt = '{} {} {} {} {}\n'.format(i,querydoc,index,questions[index],distance)
            write_to_file('../data/query.lsi.txt', txt.encode('utf-8'))
        write_to_file('../data/query.lsi.txt',  '\n'.encode('utf-8'))
    
        
def lda_similarity_main(dictionary,lda_save_path,corpora_path,index_path):
    
    questions= get_question()
    
    corpus = load_corpora(corpora_path)

    ldamodel = load_lad_model(lda_save_path)
 
    if os.path.exists(index_path):
        index_sim = similarities.MatrixSimilarity.load(index_path)
    else:
        index_sim = similarities.MatrixSimilarity(ldamodel[corpus]) 
        index_sim.save(index_path)
        
    write_to_file('../data/query.lda.txt', ''.encode('utf-8'),mode='wb+')
    for i in range(1000):
        querydoc = questions[i]
        vec_bow = dictionary.doc2bow(jieba.lcut(querydoc))
        vec_lda= ldamodel[vec_bow]
        
        sims = index_sim[vec_lda]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        for sim in sims[:10]:
            index = sim[0]
            distance = sim[1]
            txt='{} {} {} {} {}\n'.format(i,querydoc,index,questions[index],distance)
            write_to_file('../data/query.lda.txt', txt.encode('utf-8'))
        write_to_file('../data/query.lda.txt',  '\n'.encode('utf-8'))
        
def rp_similarity_main(dictionary,tfidf_save_path,rp_save_path,corpora_path,index_path):
    
    questions= get_question()
    
    corpus = load_corpora(corpora_path)

    rpmodel = load_rp_model(lda_save_path)
    tfidfmodel = load_tfidf_model(tfidf_save_path)
 
    if os.path.exists(index_path):
        index_sim = similarities.MatrixSimilarity.load(index_path)
    else:
        print('build matrix similarity')
        corpus_tfidf = tfidfmodel[corpus]
        index_sim = similarities.MatrixSimilarity(rpmodel[corpus_tfidf]) 
        index_sim.save(index_path)
        
    write_to_file('../data/query.rp.txt', ''.encode('utf-8'),mode='wb+')
    for i in range(1000):
        querydoc = questions[i]
        vec_bow = dictionary.doc2bow(jieba.lcut(querydoc))
        vectfidf = tfidfmodel[vec_bow]
        vec_rp= rpmodel[vectfidf]
        
        sims = index_sim[vec_rp]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        for sim in sims[:10]:
            index = sim[0]
            distance = sim[1]
            txt='{} {} {} {} {}\n'.format(i,querydoc,index,questions[index],distance)
            write_to_file('../data/query.rp.txt', txt.encode('utf-8'))
        write_to_file('../data/query.rp.txt',  '\n'.encode('utf-8'))
    
if __name__ == '__main__':
    lsi_save_path = '../data/yuer.lsi'
    lda_save_path = '../data/yuer.lda'
    rp_save_path = '../data/yuer.rp'
    tfidf_save_path = '../data/yuer.tfidf'
    dicsavepath = '../data/yuer.dict'
    corpora_path = '../data/yuer.mm'
    index_lsi_path ='../data/yuer.lsi.index'
    index_lda_path ='../data/yuer.lda.index'  
    index_rp_path ='../data/yuer.rp.index'      
    dictionary = corpora.Dictionary.load(dicsavepath)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    rp_similarity_main(dictionary,tfidf_save_path,rp_save_path,corpora_path,index_rp_path)
#     lsi_similarity_main(dictionary,tfidf_save_path,lsi_save_path,corpora_path,index_lsi_path)
#     lda_similarity_main(dictionary,lda_save_path,corpora_path,index_lda_path)
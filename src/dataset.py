# coding:utf-8  

import csv
import jieba
from six import iteritems
from gensim import corpora
from docutils.parsers.rst.directives import path

def build_dictionary(corppath,dicsavepath,stoplist = []):    
    dict = corpora.Dictionary(jieba.lcut(line) for line in open(corppath,'r',encoding='utf-8'))
    print(dict)
    stop_ids = [dict.token2id[stopword] for stopword in stoplist if stopword in dict.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dict.dfs) if docfreq == 1]
    
    print('stop id and once id len',len(stop_ids),len(once_ids))
    dict.filter_tokens(stop_ids + once_ids)
    dict.compactify() 
    dict.save(dicsavepath)
    print(dict)

def load_dictionary(dicsavepath):
    dict = corpora.Dictionary.load(dicsavepath)    
    return dict
    
class MyCorpus(object):
    def __init__(self,dic,corp_path):
        self.dict = dic
        self.corp_path = corp_path
        
    def __iter__(self):
        for line in open(self.corp_path,'r',encoding='utf-8'):
            yield self.dict.doc2bow(jieba.lcut(line))
    
    
def get_question(qpath=r'../data/question.csv'):
    rows = csv.reader(open(qpath,'r',encoding='utf-8'))#
    corps = []
    for r in rows:
        corps.append(r[0])

    return corps

def get_stop_words(spath = '../data/stopword.txt'):
    stopwords=[]
    with open(spath) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line:
                stopwords.append(line)
    return stopwords

def build_corpora(dicsavepath,raw_corps_path,corps_path):
    dicts = load_dictionary(dicsavepath)
    corps = MyCorpus(dicts,raw_corps_path)
    corpus  = []
    for corp in corps:
        corpus.append(corp)
    corpora.MmCorpus.serialize(corps_path, corpus)
    
def load_corpora(corps_path):
    corpus = corpora.MmCorpus(corps_path)
    return corpus

if __name__ == '__main__':
    dicsavepath = '../data/yuer.dict'
    raw_corps_path = '../data/question.csv'
    crops_path = '../data/yuer.mm'
    stopwords=get_stop_words()
    print('load stop words',len(stopwords))
    build_dictionary(raw_corps_path,dicsavepath,stopwords)
    build_corpora(dicsavepath,raw_corps_path,crops_path)
    print(load_corpora(crops_path))

    
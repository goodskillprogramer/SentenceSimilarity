# coding:utf-8  
import os
import gensim
from gensim.models import doc2vec
from dataset import get_question
import jieba
import numpy as np
import time
from utility import write_to_file

def gen_d2v_corpus(lines,savemodel,istran=False):

    total_examples = len(lines)
    with open("../data/ques2_result.txt", "wb") as fw:
        for line in lines:
            txt = " ".join(jieba.lcut(line)) + "\n"
            txt = txt.encode('utf-8')
            fw.write(txt)

    sents = doc2vec.TaggedLineDocument("../data/ques2_result.txt")
    
    model = None
    if os.path.exists(savemodel):
        print('loading model',savemodel,time.asctime())
        model = doc2vec.Doc2Vec.load(savemodel)
        print('loaded model',savemodel,time.asctime())
        if istran:
            count = 0
            while(True):
                count+=1
                epoches =20
                model.train(sents,total_examples=total_examples,epochs=epoches)
                if count % 10:
                    model.save(savemodel+"."+str(count))
                    model.save(savemodel)
                print('trained ',count*epoches)
    else:
        print('train new model')
        model = doc2vec.Doc2Vec(sents, size=300, window=12, min_count=2, workers=4,dm=0)
        
        print('train',time.asctime())
        model.train(sents,total_examples=total_examples,epochs=200)
        print('train',time.asctime())
        model.save(savemodel)

    save_path = '../data/query.doc2vec.txt'
    write_to_file(save_path,"".encode('utf-8'),mode='wb+')
    for i in range(100):
        vs=model.docvecs.most_similar(i)
        for v in vs[:10]:
            result_indx = v[0]
            distance=v[1]
            txt ='{} {} {} {} {} {}\n'.format(i,lines[i],"->",result_indx,lines[result_indx],distance)
            write_to_file(save_path,txt.encode('utf-8'))
        write_to_file(save_path,"\n".encode('utf-8'))
   
def main():
    questions = get_question()
    savemodel = '../data/gensim.model'
    gen_d2v_corpus(questions,savemodel,istran=False)

if __name__ == "__main__":
    main()
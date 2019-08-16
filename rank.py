
# data preperation - the colab version
# the following lines were added for this code to run on colab
#import os
#os.chdir('/content/drive/My Drive/NLP_ProjeFiles')
#import sys
#sys.argv = ['-f']

import data_constants
import data_loader
import data.uri as uri_helper
import utils
import json
import sys
import logging
import pandas as pd
import argparse
import pathlib
import re
import gzip
import numpy as np
from numpy import dot
from numpy.linalg import norm

class Rank:

    def loadInputData(self,inputPath):
        inputFile = open(inputPath,"r")
        counter=0
        edges = []
        sentences = []
        for line in inputFile:
            target = line
            scentence = inputFile.readline()
            sentences.append(scentence.split("\n")[0])
            edges.append(line.split("\n")[0])
            # for debug
            # print(target + " -> " + scentence)
        return edges,sentences

    def preProcess(self,filteredEdges):
        subjectRelationObjectTupleArr = []
        for val in filteredEdges:
            words = val.split(" ")
            index = 0
            tmpTuple = ([],[],[])
            for word in words:
                if word[0].isupper():
                    tmpTuple[1].append(word)
                    index = 2
                else:
                    tmpTuple[index].append(word)
            subjectRelationObjectTupleArr.append(tmpTuple)
        return subjectRelationObjectTupleArr

    def checkForEmbedding(self,subjectList,numberBatchDF,preFix):
        embedding = []
        for val in subjectList:
            if preFix+val in numberBatchDF.index:
                embedding.append(numberBatchDF.loc[preFix+val])
            else:
                return 100
        return embedding
    
    def getCosineSimilarity(self,triplet,numberBatchDF,preFix):
        a = rank.checkForEmbedding(triplet[0],numberBatchDF,preFix)
        if a==100:
            return 100
        b = rank.checkForEmbedding(triplet[2],numberBatchDF,preFix)
        if b==100:
            return 100
        a = np.asarray(a)
        b = np.asarray(b)
        if len(a)>1:
            a = np.mean(a,axis=0)
        if len(b)>1:
            b = np.mean(b,axis=0)
        cos_sim = dot(a, b.T)/(norm(a)*norm(b))
        return cos_sim

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Preprocessing ConceptNet')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=data_constants.VERSION))
    parser.add_argument('-i', '--inputScentences', default=data_constants.DEFAULT_GENERATED_SENTENCES_PATH,help='Path to generated input scentences')
    parser.add_argument('-o', '--outputScentences', default=data_constants.DEFAULT_FILTERED_SENTENCES_PATH,help='Path to filtered output scentences')
    parser.add_argument('--override', action='store_true', help='Override outputs if already exist')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    # create the filter object
    rank = Rank()
    edges,sentences = rank.loadInputData(args.inputScentences)

    # now calculte the cosine similarity betweend nodes and rank the results from -1 to 1, which means from less likely to most
    # 100 means that a word was not found in the embedding space
    # first prepare the data for embedding
    subjectRelationObjectTupleArr = rank.preProcess(edges)
    # if the word "mini" appears in the filename, load the mini version, otherwise load the full one
    preFix = ""
    if "mini" in data_constants.DEFAULT_NUMBERBATCH_PATH:
        # light version - less accurate but faster
        preFix = "/c/en/"
        numberBatchDF = data_loader.load_numberbatch(data_constants.DEFAULT_NUMBERBATCH_PATH, lang="en")
        logging.info("loaded mini version")
    else:
        # heavy version - more accurate but slower
        numberBatchDF = pd.read_csv('data/numberbatch-en-19.08.txt.gz',compression="gzip", sep=" ",skiprows=[0], header=None,index_col=0)
        logging.info("loaded full version")

    scoreArr = []
    with open(args.outputScentences, 'w', encoding='utf-8') as f:
        for i in range(len(subjectRelationObjectTupleArr)):
            score = rank.getCosineSimilarity(subjectRelationObjectTupleArr[i],numberBatchDF,preFix)
            if type(score) is np.ndarray:
                score = score[0]
            if type(score) is np.ndarray:
                score = score[0]
            scoreArr.append(score)
    tmpNPArray = np.array([sentences,scoreArr]).T
    tmpNPArray = sorted(tmpNPArray, key=lambda row: row[1],reverse=True)
    f = open(data_constants.DEFAULT_FILTERED_SENTENCES_PATH,"w")
    for val in tmpNPArray:
        f.write(val[0] + "_" + str(val[1])+ "\n")
    logging.info('Done')
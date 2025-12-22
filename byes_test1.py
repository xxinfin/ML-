import numpy as np
import re
import random



def textParse(input_string):
    print(input_string)
    listtoToken=re.split(r"\W+",input_string)
    return [tok for tok in listtoToken if len(listtoToken)>2]
def createVocablist(doclist):
    #语料表的构建
    vocabSet=set([])
    for document  in doclist:
        vocabSet=vocabSet|set (document)
    return list(vocabSet)
def trainNB(trainMat,trainClass):
    """
    训练模块
    :param data: 数据
    :param label: 标签
    :return:正常邮件中词出现的概率,垃圾邮件中词出现的概率,邮件是垃圾邮件的概率
    """
    numTrainDocs=len(trainMat)
    numwords=len(trainMat[0])
    #p1 垃圾邮件的概率值
    p1=sum(trainClass)/float(numTrainDocs)
    #np.ones进行初始化，做了一个平滑处理，防止分子变为0
    p0Num=np.ones((numwords))
    p1Num=np.ones((numwords))
    #定义分母个数，通常情况下，设置为分类的个数
    p0Denom=2
    p1Denom=2
    for i in range(numTrainDocs):
        if trainClass[i]==1:
            p1Num+=trainMat[i]
            p1Denom+=sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    #求概率值
    p1Vec=np.log(p1Num/p1Denom)
    p0Vec=np.log(p0Num/p0Denom)
    return p1Vec,p0Vec,p1
def classifyNB(wordVec,p1Vec,p0Vec,p1_class):
    """
    预测函数
    :param word_vec: 文章中的词
    :param p0_vec: 词出现在正常邮件中的概率
    :param p1_vec: 词出现在垃圾邮件中的概率
    :param p1_class: 邮件是垃圾邮件的概率
    :return:0 正常邮件 、1 垃圾邮件
    """
    #sum(wordVec * p1Vec)词频相加
    p1=np.log(p1_class)+sum(wordVec*p1Vec)
    p0=np.log(1.0-p1_class)+sum(wordVec*p0Vec)
    #判断是否为正常邮件
    if p0>p1:
        return 0
    else:
        return 1

def setOfWord2Vec(vocablist,inputSet):
    returnVec=[0]*len(vocablist)
    for word in inputSet:
        returnVec[vocablist.index(word)]=1
    return returnVec
def spam():
    #数据的读取
    doclist=[]
    classlist=[]
    for i  in range(1,26):
       wordlist= textParse(open("email/sham16/%d.txt"%i,"r").read())
       doclist.append(wordlist)
       classlist.append(1)

       wordlist = textParse(open("email/ham16/%d.txt" % i, "r").read())
       doclist.append(wordlist)
       classlist.append(0)
    vocablist=createVocablist(doclist)
    trainSet=list(range(40))
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat=[]
    trainClass=[]
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist,doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    p1Vec,p0Vec,p1=trainNB(np.array(trainMat),np.array(trainClass))
    errorCount=0
    for docIndex in testSet:
        wordVec=setOfWord2Vec(vocablist, doclist[docIndex])
        if classifyNB(np.array(wordVec),p1Vec,p0Vec,p1)!=classlist[docIndex]:
            errorCount+=1
    print("当前10个样本错了，",errorCount)
if __name__=='__main__':
    spam()
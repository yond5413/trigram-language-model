import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""
#Yonathan Daniel 
#UNI id yd2696
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    gen = list(sequence)
    ret = []
    if n < 1:
        print("Must be greater than 1")
    else:
        ### maybe one case for n>len(sequence) and the reverse 
        ###TODO add branches for when 2d array then string list or 1 dimensional string plus string
        ## assume at least one list is passed 
        twoD_flag = False
        if type(gen[0]) == type([]):
            twoD_flag = True
        elif type(gen[0]) == type('s'):
            twoD_flag = twoD_flag
        ################################
        if n == 1 and twoD_flag:
            #ret.append(("START",))
            for i in range(0,len(gen)): 
                for j in range(0,len(gen[i])):    
                    if j==0:
                        ret.append(("START",))
                    storage = list()
                    storage = tuple([gen[i][j]])#[i])
                    ret.append(storage)
                    if j+1 == len(gen[i]):
                        ret.append(("STOP",))
        if n == 1 and twoD_flag == False:
            #ret.append(("START",))
            for i in range(0,len(gen)):             
                if i ==0:
                    ret.append(("START",))
                storage = list()
                storage = tuple([gen[i]])#[i])
                ret.append(storage)
                if i+1 == len(gen[i]):
                    ret.append(("STOP",))
            #ret.append(("STOP",))
        elif n==2 and twoD_flag:
            #storage = ()
            ## will double check if 100% what I want
            for i in range(0,len(gen)):#-1):
                for j in range(0,len(gen[i])-1):
                #TODO make 2 space tuples
                #storage  = ()
                    if j ==0:
                        firstInput = ("START",gen[i][j])
                        ret.append(firstInput)
                    if(len(gen)%n != 0) or n> len(gen):
                        if i == len(gen)-1 or n>len(gen):
                            break
                    storage = list()
                    storage.append(gen[i][j])
                    storage.append(gen[i][j+1])
                    #storage = (gen[i][j],gen[i][j+1])#list()#(gen[i],gen[i+1])
                #storage.append(gen[i])
                #storage.append(gen[i+1])
                    ret.append(tuple(storage))
                    if j+1 == len(gen)-1:
                        lastInput = (gen[i][-1],"STOP")
                        ret.append(lastInput)
        elif n==2 and twoD_flag == False:
            #storage = ()
            ## will double check if 100% what I want
            for i in range(0,len(gen)-1):#-1):
                    if i ==0:
                        firstInput = ("START",gen[i])
                        ret.append(firstInput)
                    if(len(gen)%n != 0) or n> len(gen):
                        if i == len(gen)-1 or n>len(gen):
                            break
                    storage = list()
                    storage.append(gen[i])
                    storage.append(gen[i+1])
                    #storage = (gen[i][j],gen[i][j+1])#list()#(gen[i],gen[i+1])
                #storage.append(gen[i])
                #storage.append(gen[i+1])
                    ret.append(tuple(storage))
                    if i+1 == len(gen)-1:
                        lastInput = (gen[i+1],"STOP")
                        ret.append(lastInput)
        elif n==3 and twoD_flag:
            #firstInput = ("START","START",gen[0][0])
            #secondInput = ("START",gen[0][0],gen[0][1])
            #ret.append(firstInput)
            #ret.append(secondInput)
            if len(gen)>n: #paranoia 
                for i in range(0,len(gen)):
                    #TODO make 3 space tuples
                    for j in range(0,len(gen[i])-2):
                        if j ==0:
                           firstInput = ("START","START",gen[i][j])
                           secondInput = ("START",gen[i][j],gen[i][j+1])
                           ret.append(firstInput)
                           ret.append(secondInput)
                        storage = list()
                        storage.append(gen[i][j])
                        storage.append(gen[i][j+1])
                        storage.append(gen[i][j+2])
                    #storage = (gen[i],gen[i+1],gen[i+2]) 
                        ret.append(tuple(storage))
                        if j+1 == len(gen)-2:
                            lastInput =(gen[i][-2],gen[i][-1],"STOP") #(gen[len(gen)-1][len(gen[len(gen)-1])-2],gen[len(gen)-1][len(gen[len(gen)-1])-1],"STOP")
                            ret.append(lastInput)
        elif n==3 and twoD_flag == False:
            if len(gen)>n: #paranoia 
                for i in range(0,len(gen)-2):
                    #TODO make 3 space tuples
                        if i ==0:
                           firstInput = ("START","START",gen[i])
                           secondInput = ("START",gen[i],gen[i+1])
                           ret.append(firstInput)
                           ret.append(secondInput)
                        storage = list()
                        storage.append(gen[i])
                        storage.append(gen[i+1])
                        storage.append(gen[i+2])
                    #storage = (gen[i],gen[i+1],gen[i+2]) 
                        ret.append(tuple(storage))
                        if i+1 == len(gen)-2:
                            lastInput =(gen[i+2],gen[i+1],"STOP") #(gen[len(gen)-1][len(gen[len(gen)-1])-2],gen[len(gen)-1][len(gen[len(gen)-1])-1],"STOP")
                            ret.append(lastInput)
    return ret 


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        ###########################
        self.totalWords =0# len()
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        # add total wordcount somewhere
        
    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        #generator = corpus_reader(corpus)
        #lexicon = get_lexicon(generator)
        #for i in corpus:
        #    print(i)
        
        self.unigramcounts = defaultdict(int)#{} # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)#{} 
        self.trigramcounts = defaultdict(int)#{} 
        ######################
        #self.wordcount = 0 ## may should do -2 because of START and END
        ##Your code here
        # don't like generators tbh
        cor = list(corpus)
        one_gram = list()
        two_gram =  list()
        three_gram = list()
        one_gram.append(get_ngrams(cor,1))
        two_gram.append(get_ngrams(cor,2))
        three_gram.append(get_ngrams(cor,3))
        #################################
        ### adding keys to dictionary
        for i in range(0,len(one_gram)):
            #print("unigram initalizing counts loop")
            for j in range(0,len(one_gram[i])):
                self.totalWords +=1#?
                if one_gram[i][j] not in self.unigramcounts:## delete if it doesn't work
                    self.unigramcounts[one_gram[i][j]] = 1
                else:
                    self.unigramcounts[one_gram[i][j]] +=1
            #self.lexicon.add[i]
        #self.totalWords -= self.unigramcounts[('START',)]
        for i in range(0,len(two_gram)):
            #print("bigram initalizing counts loop")
            for j in range(0,len(two_gram[i])):
                #if( i<5 and j < 5):
                #    print("BI:"+str(two_gram[i][j]))
                if two_gram[i][j] not in self.bigramcounts:## delete if it doesn't work
                    self.bigramcounts[two_gram[i][j]] = 1
                else:
                    self.bigramcounts[two_gram[i][j]] +=1
        #print("Done Bi")
        for i in range(0,len(three_gram)):
            #print("trigram initalizing counts loop")
            for j in range(0,len(three_gram[i])):
                #if( i<5 and j < 5):
                    #print("Tri:"+str(three_gram[i][j]))
                    #print(three_gram[i][j])
                if three_gram[i][j] not in self.trigramcounts:## delete if it doesn't work
                    self.trigramcounts[three_gram[i][j]] = 1
                else:
                    self.trigramcounts[three_gram[i][j]] +=1
        #################################
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        #P(w|u,v) = count(u,v,w)/count(u,v)
        #P(w|START,START) #Special case?
        #P(w|START,v) v not START
        ret =0.0
        if trigram not in self.trigramcounts:
            storage = list()
            if trigram[0] not in self.lexicon and trigram[1] not in self.lexicon and trigram[2] not in self.lexicon:
                storage.append("UNK")
                storage.append("UNK")
                storage.append("UNK")
                trigram = tuple(storage)
            elif trigram[0] not in self.lexicon and trigram[1] not in self.lexicon:
                storage.append("UNK")
                storage.append("UNK")
                storage.append(trigram[2])
                trigram = tuple(storage)
            elif trigram[0] not in self.lexicon and trigram[2] not in self.lexicon:
                storage.append("UNK")
                storage.append(trigram[1])
                storage.append("UNK")
                trigram = tuple(storage)
            elif trigram[1] not in self.lexicon and trigram[2] not in self.lexicon:
                storage.append(trigram[1])
                storage.append("UNK")
                storage.append("UNK")
                trigram = tuple(storage)
            elif trigram[0] not in self.lexicon:
                storage.append("UNK")
                storage.append(trigram[1])
                storage.append(trigram[2])
                trigram = tuple(storage)
            elif trigram[1] not in self.lexicon:
                storage.append(trigram[0])
                storage.append("UNK")
                storage.append(trigram[2])
                trigram = tuple(storage)
            elif trigram[2] not in self.lexicon:
                storage.append(trigram[0])
                storage.append(trigram[1])
                storage.append("UNK")
                trigram = tuple(storage)
        ####################################
        bi_storage = list()
        bi_storage.append(trigram[0])
        bi_storage.append(trigram[1])
        count_uv = self.bigramcounts[tuple(bi_storage)]
        if count_uv >0:
            ### should operate like normal
            ret = self.trigramcounts[trigram]/self.bigramcounts[tuple(bi_storage)]
        elif bi_storage == ('START','START'):
            #special case
            # number of times u starts a sentence/ number of sentences
            newTup = list()
            newTup.append(trigram[1])
            newTup.append(trigram[2])
            newTup = tuple(newTup)
            ret = self.trigramcounts[trigram]/self.unigramcounts[('STOP',)]##self.bigramcounts[newTup]/self.unigramcounts[('STOP',)]
        elif count_uv == 0:
            ### 2 cases
            #if trigram[2] not in self.lexicon:## I think at least 
                #print("does this ever happen")
                ret = 1/len(self.lexicon)
                #else:
                #ret = self.raw_unigram_probability(tuple(trigram[2]))
        else:
            print("idk what happened here")
            print("Trigram here: "+str(trigram))
        return ret
        # previous implemntation below 

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        #P(w|u) = count(u,w)/count(u)
        ret = 0.0
        #return ret
        '''count = 0
        bottom = 1

        if(bigram in self.bigramcounts):
            if tuple(bigram[1]) in self.unigramcounts:
                #probably shouldn't happen
                # regular case
                count = self.bigramcounts[bigram]
                uni =tuple([bigram[0],])
                #uni.append(bigram[0])
                bottom = self.unigramcounts[uni]
                #print("Should never happen tbh")
            #else:
            #    pass#count = self.unigramcounts[bigram[1]]
        '''#else:
            #if bigram[0] in self.lexicon:
               # ret = 0
         #       count = self.unigramcounts[bigram[1]]#count = 0
                #bottom = 1
          #  else:
        '''
        ### (u,v)
        ### if u been seen and v has been seen do reg
        ### if u has not been seen and v has do u/total
        ## if u has not  been been seen and v has not been seen
        
        return ret'''
        if bigram not in self.bigramcounts:
            storage = list()
            if bigram[0] not in self.lexicon and bigram[1] not in self.lexicon:
                storage.append("UNK")
                storage.append("UNK")
                bigram = tuple(storage)
                ### says should be 0
            if bigram[0] not in self.lexicon:
                storage.append("UNK")
                storage.append(bigram[1])
                bigram = tuple(storage)
                ## unigram 
            #elif bigram[1] not in self.lexicon:
            #    storage.append(bigram[0])
            #    storage.append("UNK")
            #    bigram = tuple(storage)
        #TODO check with the ("START","START") case
        ### p(u|START is another special case)
        ################################################ 
        ## check if potential case for division by zero 
        # if so divided by totalnumber of words 
        ########## special case #############
        #if bigram == ("START","START"):
        #    return ret 
        bi_storage = list()
        bi_storage.append(bigram[0])
        if bigram[0] == "START":
            #print("Start in bigram: "+ str(bigram))
            if bigram[1] == "START":    
                ret = ret
            else:
                ret = self.bigramcounts[bigram]/self.unigramcounts[("STOP",)]
            return ret 
        if self.unigramcounts[tuple(bi_storage)] == 0:
            #TODO
            #print("a division by 0?")
            ret = self.bigramcounts[bigram]/self.totalWords#self.unigramcounts[bigram[0]]
        else:
            # no division by 0
            ret = self.bigramcounts[bigram]/self.unigramcounts[tuple(bi_storage)]#bigram[0]]
        ################################################
        return ret #0.0'''
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        ret = 0.0
        starts = self.unigramcounts[("START",)]
        if unigram[0] not in self.lexicon:
            print("UNI fail")
            unigram = ("UNK",)
        #### special case 
        #starts = self.unigramcounts[("START",)] # THINK THIS HAS TO BE REMOVED FROM self.totalWords
        ## total number of word =>The number of tokens in the corpusfile after replace unseen tokens, right?
        if unigram == ("START",):
            return ret
        else:
            ret =   self.unigramcounts[unigram]/(self.totalWords-starts)#(self.totalWords)#-starts)
            #double check which is correct^
            # like should I remove START or assume it isn't included    
            # TODO check at OH      
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return ret #0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = ""#TODO
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        ###############
        ret = 0.0
        #count_star = 0.0 #?
        #P(w|u,v) = L1*P_mle(w|u,v) + L2*P_mle(w|v) +L3*P_mle(w)
        bi_storage = list()#list(trigram[0],trigram[1])
        #bi_storgae.append(trigram[0])
        #bi_storgae.append(trigram[1])
        bi_storage.append(trigram[1])
        bi_storage.append(trigram[2])
        uni_storage = list()
        uni_storage.append(trigram[2])#uni_storage.append(trigram[0])

        ret = lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(tuple(bi_storage)) + lambda3*self.raw_unigram_probability(tuple(uni_storage))
        return ret #0.0
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        ### check lecture notes 
        ret = float("-inf")
        trigrams = get_ngrams(sentence,3)
        prob = float("0.0")
        ret = float("0.0")#math.log2(prob)
        ####### think this makes sense ##
        for i in range(0,len(trigrams)):
            #print(trigrams[i])
            #for j in range(0,len(trigrams[i])):
                #print(trigrams[i])
                #print(trigrams[i][j])
            prob = self.smoothed_trigram_probability(trigrams[i])
            #if prob !=0:#trigrams[i] in self.trigramcounts:# to avoid 0 
            #    ret+= math.log2(prob)
            ret+= math.log2(prob)
        #################################
        return ret#float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        # corpus = generator object 
        ## M = total word tokens
        ## m = number of sentences
        m = 0
        M = 0
        l = 0
        for sentence in  corpus:
            m += 1
            #get_ngrams(sentence,3)
            #print(sentence)#?
            l += self.sentence_logprob(sentence)
            M +=1
            for word in sentence:
                if word != "START":
                    M+=1
        l = l/M
        #print(l)
        perp = float(2**(-l))
        return perp #float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
    #######################################
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total +=1
            ## want perplexity of the one it was trained on 
            if pp<p2:
                correct +=1
            #dir1.append(pp)
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            p2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total +=1
            if pp<p2:
                correct +=1
            
        ret = correct/total
        return ret#0.0

if __name__ == "__main__":
    #print("hello")
    #python -i trigram_model.py hw1_data/hw1_data/ets_toefl_data/test_high/brown_train.txt
    # actual one below
    ##python -i trigram_model.py hw1_data/hw1_data/brown_train.txt##
    model = TrigramModel(sys.argv[1]) 
    
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #_
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    og_corpus = corpus_reader(sys.argv[1],model.lexicon)
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)

    pp = model.perplexity(dev_corpus)
    og_pp = model.perplexity(og_corpus)
    print(pp) #for train/ test should be between 100-300 but not more than 400
    print(og_pp) # for  train/train should be around 10 
    ## i added the og ones


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('ets_toefl_data/train_high.txt','ets_toefl_data/train_low.txt', "ets_toefl_data/test_high", "ets_toefl_data/test_low")#essay_scoring_experiment('ets_toefl_data/train_high.txt', 'ets_toefl_data/train_low.txt', "ets_toefl_data/test_high", "ets_toefl_data/test_low")#essay_scoring_experiment('ets_toefl_data/train_high.txt', 'ets_toefl_data/train_low.txt', "ets_toefl_data/", "ets_toefl_data/")
    #essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")#essay_scoring_experiment('ets_toefl_data/train_high.txt', 'ets_toefl_data/train_low.txt', "ets_toefl_data/", "ets_toefl_data/")
    print(acc)
   # python -i trigram_model.py hw1_data/hw1_data/brown_train.txt hw1_data/hw1_data/brown_test.txt
   # run with above with training and test files. 
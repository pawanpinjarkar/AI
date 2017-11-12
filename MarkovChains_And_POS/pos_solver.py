#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
# Chandan Avdhut- cavdhut@iu.edu
# Sunil Agrawal - suniagra@iu.edu
# Pawan Pinjarkar - ppinjark@iu.edu
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
# The project contains two problem statements -
Part 1 - Marcov Chains
Part 2  - Part of Speech tagging
Part of Speech tagging is achieved by using the following algorithms:
# 1) Simplified
# 2) HMM Variable Elimination
# 3) HMM MAP
==============================================================================================================================
# How did we train the data?
We used the large corpus of labeled training and testing data provided my Professor David, consisting of nearly 1 million words and 50,000 sentences.
==============================================================================================================================
#Part 1.2  - Estimate the probabilities of the given Markov Chain
We calculated probabilities for first word of the sentence by counting the number of occurrences of that word at the start of the sentence divided by total number of sentences in the given training data. To calculate transition probability we used below approach.
For example, if the training sentence is "a b c d ", "a c b d ", then to calculate probability of b given a (P(b|a), we counted  the number of times 'b' appeared after 'a' and then divided that by total number of 'a'. i.e.  the number of times 'b' appeared after 'a'  = 1 and total number of 'a' = 2 . Therefore the transition probability is 1/2 = 0.5
We used python dictionary objects to populate and store the various probabilities. For example,
        self.word_initial_prob = {}# Initial probabilities for words P(W0)
        self.word_prob = {}		   # Probability of a given word
        self.trans_prob = {}       # Transition probabilities P(Wi+1 | Wi)
==============================================================================================================================
# Part 1.3
To implement grammar checker, we read the file containing data for the common confused words in English language. For the words in the given input sentence, we checked if an input word is in the list of confused words. After replacing confused words, we prepared sentences with all the combinations of present confused words. We also calculated the probability and we selected a sentence with the maximum probability. To calculate the probability of a sentence, we have used initial probability of initial word and multiplied by transition probabilities of all the remaining words.
Assumption - If a word is not present in the training set data then, for such words we are using a very negligible probability of 0.000000001.
==============================================================================================================================
Part 2.1
We calculated probabilities for first part of speech (pos) P(S0) of the sentence by counting the number of occurrences of that pos at the start of the sentence divided by  total number of sentences in the given training data.
We calculated probabilities for part of speech (pos) P(Si) by counting the number of occurrences of that pos divided by  total number of pos in the given training data.
Transition probability - For example, if the pos training sentence is " noun verb noun adjective ", "noun adjective verb noun ", then to calculate probability of verb given noun P(verb|noun), we counted  the number of times 'verb' appeared after 'noun' and then divided that by total number of 'noun'. i.e.  the number of times 'verb' appeared after 'noun'  = 1 and total number of 'noun' = 4 . Therefore the transition probability is 1/4 = 0.25.
Emission probability - To calculate emission probability for part of speech P(Wi | Si) i.e. probability of a word given pos, we counted number of times a given word appeared as pos divided by total number of times that pos appeared in the training data.
For part of speech tagging, we used python dictionary objects to populate and store the various probabilities. For example,
        self.pos_prob={}           # Probability for part of speech P(Si)
        self.pos_initial_prob ={}  # Initial probability for part of speech P(S0)
        self.pos_trans_prob = {}   # Transition probability for part of speech P(Si+1 | Si)
        self.pos_emision_prob= {}  # Emission probability for part of speech P(Wi | Si)
==============================================================================================================================
Part 2.2: Simplified algorithm: def simplified(self, sentence):
For each word in the given sentence, we did following -
	For each pos we calculated P(S)*P(W|S) i.e. probability of pos multiplied by probability of word given pos.

Note - As per given figure 1.b, each word is only dependent on corresponding pos hence we did not consider the transition probabilities.
==============================================================================================================================
Part 2.3: Variable Elimination algorithm: def hmm_ve(self, sentence)
To implement Variable Elimination algorithm, we considered following probabilities and dictionary objects.
		self.pos_initial_prob ={}   # P(S0) Initial distribution
        self.pos_trans_prob = {}  #  P(Si+1 | Si) Transition probability
        self.pos_emision_prob= {} # P(Wi | Si) Emission probability
		tao ={} # tao function as used in variable elimination algorithm

tao is initialized to 1 initially for all pos. 

For the initial value of tao i.e. tao 0, we calculated the product of the initial probabilities and emission probabilities for each row.

For next tao function i.e. tao 1 we did a summation of initial probabilities and emission probabilities for 12 pos .

For all further tao functions i.e. tao 2 onwards,  we did a summation of initial probabilities and emission probabilities for 12 X 12 = 144 pos.

Finally, we determined the maximum probabilities for each words.

Assumption : We considered emission probabilities to a very negligible value of 0.000000001 to handle the scenarios where the training data is unable to provide the emission probabilities. We  also considered transition probabilities to a very negligible value of 0.000000001 to handle the scenarios where the training data is unable to provide the transition probabilities. 

==============================================================================================================================
Part 2.4: HMM Viterbi algorithm: def hmm_viterbi(self, sentence):
To implement Viterbi algorithm, we considered following probabilities.
		self.pos_initial_prob ={}   # P(S0) Initial distribution
        self.pos_trans_prob = {}  #  P(Si+1 | Si) Transition probability
        self.pos_emision_prob= {} # P(Wi | Si) Emission probability
Node 0:  Initial distribution * Emission probability
	P(S0|W0) = P(S0) * P(W0|S0)
For the first node, we only have to consider Initial distribution * Emission probability.
Node 1 onwards:
At each node for the given pos, we calculated
	max(viterbi value for previous pos * Transition probability * Emission probability )
	P(Si|Wi) = P(Si) *  P(Si|previousPOS) * P(Wi|Si)
We then selected previousPOS with maximum probability and saved it in a dictionary variable so as to backtrack to previous nodes.
For the last node, we selected pos with maximum viterbi value and then we used previous nodes dictionary to backtrack to first node and get the most probable sequence of pos for a given sentence.
Assumption: We considered a negligible probability of 0.000000001 for missing words in the training data.
==============================================================================================================================
Results:

Part 1.2:
		$ ./label.py part1.2 bc.train
		Learning model...


		1. We have all her date being victimized by local church , a slight modification depended greatly uppon to carry out of peg laughing fit in the bandits .
		2. Refuses to 8-inch drain pipe for materials handling .
		3. Our domestic and f. w. pohly said , the acting with legs , for example , , and meltzer lightened occasionally muttering in the masterpiece .
		4. Mrs. frank lee , 944 persons found a 100-yard range of car was thrown in that prevailed on any loss , miss him , we would still in `` miss ada !
		5. `` favorable circumstances before eleven states pharmacopoeia reference , not said sunday after thanksgiving .

Part1.3:

		$./label.py part1.3 bc.train "he is to busy"
		Learning model...


		Probability of Input Sentence : 4.79165236472e-14
		probability of Suggested Sentence: 2.01411828212e-08

		Suggested sentence:
		He is too busy
		
		
Part 2:
	Part 2.1: $ ./label.py part2.1  bc.train bc.test
		....................
		....................
		Simplfied probablities for word "it's" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt = 1.0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "late" :
		  adv = 0.241610738255 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x= 0 conj= 0 adj = 0.758389261745
		Simplfied probablities for word "and" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x = 0.0001261829653 conj = 0.999873817035 adj= 0
		Simplfied probablities for word "you" :
		  adv= 0 noun= 0 adp= 0 pron = 1.0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "said" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb = 1.0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "they'd" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt = 1.0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "be" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb = 0.999644760213 x = 0.000355239786856 conj= 0 adj= 0
		Simplfied probablities for word "here" :
		  adv = 1.0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "by" :
		  adv = 0.0088368269626 noun= 0 adp = 0.990957665434 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x = 0.000205507603781 conj= 0 adj= 0
		Simplfied probablities for word "dawn" :
		  adv= 0 noun = 1.0 adp= 0 pron= 0 det= 0 num= 0 .= 0 prt= 0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "''" :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 . = 1.0 prt= 0 verb= 0 x= 0 conj= 0 adj= 0
		Simplfied probablities for word "." :
		  adv= 0 noun= 0 adp= 0 pron= 0 det= 0 num= 0 . = 1.0 prt= 0 verb= 0 x= 0 conj= 0 adj= 0
								  : it's late and  you  said they'd be   here by   dawn ''   .
		 0. Ground truth ( -79.00): prt  adv  conj pron verb prt    verb adv  adp  noun .    .
		   1. Simplified ( -77.89): prt  adj  conj pron verb prt    verb adv  adp  noun .    .
			   2. HMM VE ( -77.89): prt  adj  conj pron verb prt    verb adv  adp  noun .    .
			  3. HMM MAP ( -77.89): prt  adj  conj pron verb prt    verb adv  adp  noun .    .

		==> So far scored 2000 sentences with 29442 words.
						   Words correct:     Sentences correct:
		{'2. HMM VE': 27056, '3. HMM MAP': 27991, '0. Ground truth': 29442, '1. Simplified': 27646}
		   0. Ground truth:      100.00%              100.00%
			 1. Simplified:       93.90%               47.30%
				 2. HMM VE:       91.90%               38.20%
				3. HMM MAP:       95.07%               54.35%
==============================================================================================================================
Thoughts:
Out of the 3 algorithms we tested, we found that the performance of Viterbi (HMM MAP) algorithm was the best compared to others. The Variable elimination algorithm performed the worst. Implementation of simplified algorithm was quite easy and also it resulted a descent performance.
We had also considered a very small probability of 0.000000001 for the transition and emission probabilities for the words not found in the training data

Future scope:
Implementing grammar fun was quite a fun and we really enjoyed coding and testing the results with the trained data. We can know understand how the online tools for grammar check ( for example, Grammarly) works. The current implementation of grammar checker is as per the given problem statement however, we strongly believe that the code for grammar check can be improved easily to do a full fledge grammar. 

Further improvements can be done to handle unseen data i.e. words that are not present in the training data.
'''
####
import random
import math
import copy




def do_part12(data):
    solver = Solver()
    solver.train(data)
    for i in range(1, 6):
        print str(i) + ". " + str (solver.sample())

def do_part13(data, sentence ):
    solver = Solver()
    solver.train(data)
    solver.grammer_check(sentence.lower())




# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.word_initial_prob = {}  # Initial probabilities of words P(W1)
        self.word_prob = {} # Probabilities of word P(Wi)
        self.trans_prob = {}   # transition probabilities P(Wi+1 | Wi)
        self.pos_prob={}         #  P(S)
        self.pos_initial_prob ={}   # P(S0)
        self.pos_trans_prob = {}  #  P(Si+1 | Si)
        self.pos_emision_prob= {} # P(Wi | Si)

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):

        prob=0.0
        for i in range(len(sentence)):
            if sentence[i] in self.pos_emision_prob and label[i] in self.pos_emision_prob[sentence[i]]:
                prob = prob  + math.log(self.pos_emision_prob[sentence[i]][label[i]]) 
            else:
                prob = prob  + math.log(0.000000001)                           
            if i == 0:
                if label[i] in self.pos_initial_prob:
                    prob = prob + math.log(self.pos_initial_prob[label[i]])
                else:
                    prob = prob + math.log(0.000000001)
            else:
                if label[i] in self.pos_trans_prob and label[i-1] in self.pos_trans_prob[label[i]]:
                    prob = prob + math.log(self.pos_trans_prob[label[i]][label[i-1]])
                else:
                    prob = prob + math.log(0.000000001)

        return prob

    # Do the training!
    #
    def train(self, data):
        total_count = 0
        for d in data:
            total_count += len(d[0])
            word = d[0][0]
            # calculate initial probability for word
            if word in self.word_initial_prob:
                self.word_initial_prob[word] += 1
            else:
                self.word_initial_prob[word] = 1
            if word in self.word_prob:
                self.word_prob[word] += 1
            else:
                self.word_prob[word] = 1
            for i in xrange(1,len(d[0])):
                word = d[0][i]
                pre_word=d[0][i-1]

                if word in self.word_prob:
                    self.word_prob[word] += 1
                else:
                    self.word_prob[word] = 1
                # get actual probability P(Wi+1 | wi)
                if word not in self.trans_prob:
                    self.trans_prob[word] = {pre_word: 1}
                elif pre_word not in self.trans_prob[word]:
                    self.trans_prob[word][pre_word] = 1
                else:
                    self.trans_prob[word][pre_word] += 1

        for word in self.word_initial_prob:
            self.word_initial_prob[word] = float(self.word_initial_prob[word]) / float(len(data))

        for word in self.trans_prob:
            for pre_word in self.trans_prob[word]:
                self.trans_prob[word][pre_word] = float(self.trans_prob[word][pre_word]) / float(self.word_prob[pre_word])


    def train_part2(self, data):
        total_count = 0
        for d in data:
            total_count += len(d[1])
            pos = d[1][0]
            word=d[0][0]
            # calculate initial probability for pos
            if pos in self.pos_initial_prob:
                self.pos_initial_prob[pos] += 1
            else:
                self.pos_initial_prob[pos] = 1

            if pos in self.pos_prob:
                self.pos_prob[pos] += 1
            else:
                self.pos_prob[pos] = 1

            if word not in self.pos_emision_prob:
                self.trans_prob[word] = {pos: 1}
            elif pos not in self.pos_emision_prob[word]:
                self.pos_emision_prob[word][pos] = 1
            else:
                self.pos_emision_prob[word][pos] += 1

            for i in xrange(1,len(d[1])):
                pos = d[1][i]
                pre_pos=d[1][i-1]
                word = d[0][i]
                if pos in self.pos_prob:
                    self.pos_prob[pos] += 1
                else:
                    self.pos_prob[pos] = 1
                # get actual probability P(Si+1 | Si)
                if pos not in self.pos_trans_prob:
                    self.pos_trans_prob[pos] = {pre_pos: 1}
                elif pre_pos not in self.pos_trans_prob[pos]:
                    self.pos_trans_prob[pos][pre_pos] = 1
                else:
                    self.pos_trans_prob[pos][pre_pos] += 1

                if word not in self.pos_emision_prob:
                    self.pos_emision_prob[word] = {pos: 1}
                elif pos not in self.pos_emision_prob[word]:
                    self.pos_emision_prob[word][pos] = 1
                else:
                    self.pos_emision_prob[word][pos] += 1

        for pos in self.pos_initial_prob:
            self.pos_initial_prob[pos] = float(self.pos_initial_prob[pos]) / float(len(data))
        #P(Si+1| Si)

        for pos in self.pos_trans_prob:

            for pre_pos in self.pos_trans_prob[pos]:
                self.pos_trans_prob[pos][pre_pos] = float(self.pos_trans_prob[pos][pre_pos]) / float(self.pos_prob[pre_pos])
        # P(Wi|Si)
        for word in self.pos_emision_prob:
            for pos in self.pos_emision_prob[word]:
                self.pos_emision_prob[word][pos] = float(self.pos_emision_prob[word][pos]) / float(self.pos_prob[pos])

        #P(s)
        for pos in self.pos_prob:
            self.pos_prob[pos] = float(self.pos_prob[pos] ) / float(total_count)

        print "Initial POS prob" + str(self.pos_initial_prob)
        print "POS prob" + str(self.pos_prob)
        print "POS transition prob" + str(self.pos_trans_prob)
        print "Emission Prob" + str(self.pos_emision_prob)


    def sample(self):
        rand = random.random()
        psum = 0.0
        sample=[]
        for word in self.word_initial_prob:
            psum  += self.word_initial_prob[word]
            if rand < psum:
                sample.append(word)
                break
        while sample[-1]!= "." and sample[-1]!= "?" and sample[-1]!= "!":
            rand = random.random()
            #print rand
            psum = 0.0
            for word in self.trans_prob:
                if sample[-1] in self.trans_prob[word]:
                    psum += self.trans_prob[word][sample[-1]]
                    if rand <= psum:
                        sample.append(word)
                        break

        sample[0] =sample[0].title()
        #print "sample "  + " ".join(sample)
        return " ".join(sample)

    def grammer_check(self, input_line):

        file = open("confused_words.txt", 'r');
        confused_words=[]
        for line in file:
            data = tuple([w.lower() for w in line.split()])
            confused_words.append(data)

        input_words = input_line.split()
        variants=[]
        variants.append(input_words)
        for i in range(0, len(input_words)):
            for j in range(0, len(confused_words)):
                if input_words[i] in confused_words[j]:
                    for cword in confused_words[j]:
                        for variant in variants:

                            newvariant=copy.deepcopy(variant)
                            newvariant [i] = cword
                            if newvariant not in variants:
                                variants.append(newvariant)
                    break
        maxprob=0.0
        suggested_sent=input_words
        count=0

        for variant in variants:
            prob =0.000000001
            if variant[0] in self.word_initial_prob:
                prob=self.word_initial_prob[variant[0]]
            for i in range (1,len(variant)):
                if variant[i] in self.trans_prob and variant[i - 1] in self.trans_prob[variant[i]]:
                    prob *= self.trans_prob[variant[i]][variant[i - 1]]
                else:
                    prob *=0.000000001
            if maxprob < prob:
                maxprob=prob
                suggested_sent=variant
            if count == 0 :
                orginal_probability=prob
            count +=1
        suggested_sent[0] = suggested_sent[0].title()
        if suggested_sent != input_words:
            print "Probability of Input Sentence : " + str(orginal_probability)
            print "probability of Suggested Sentence: " + str(maxprob)
            print "\nSuggested sentence:\n", " ".join(suggested_sent)
        else:
            print "Probability of Input Sentence ( " + str(orginal_probability) + " ) is Highest So No Change Required: "
            print "\nFinal sentence:\n", " ".join(suggested_sent)




    # Functions for each algorithm.
    #
    def simplified(self, sentence):

        output_pos= [ "noun" ] * len(sentence)

        for i in range(len(sentence)):
            maxprob = 0
            probalitty={}
            for pos in self.pos_prob:
                #max si P(W|S) P(S)
                if sentence[i] in self.pos_emision_prob and pos in self.pos_emision_prob[sentence[i]]:
                    probalitty[pos] = self.pos_emision_prob[sentence[i]][pos] * self.pos_prob[pos]
                    if maxprob < self.pos_emision_prob[sentence[i]][pos] * self.pos_prob[pos]:
                        maxprob =self.pos_emision_prob[sentence[i]][pos] * self.pos_prob[pos]
                        output_pos[i]=pos
                else:
                    probalitty[pos] = 0
            prob_sum = sum(probalitty.values())
            print_prob = ""
            for pos in self.pos_prob:
                if probalitty[pos] != 0:
                    print_prob = print_prob +" " +pos +" = " + str(probalitty[pos]/prob_sum)
                else:
                    print_prob = print_prob + " " + pos + "= 0"
            print "Simplfied probablities for word \"" + sentence [i] + "\" :\n " + print_prob

        return output_pos

    def hmm_ve(self, sentence):
        output = ["noun"] * len(sentence)
        probability=[]
        tao={}

        for i in self.pos_prob:
            tao[i]=[1 for j in range(len(sentence))]

        for i in range(len(sentence)):
            if i==0:
                for j in self.pos_prob:
                    if sentence[i]in self.pos_emision_prob and j in self.pos_emision_prob[sentence[i]]:
                        tao[j][i]=self.pos_initial_prob[j]*self.pos_emision_prob[sentence[i]][j]
                    else:
                        tao[j][i] = self.pos_initial_prob[j] *0.000000001
            elif i==1:
                for j in self.pos_prob:
                    emi_prob= 0.000000001
                    if sentence[i] in self.pos_emision_prob and j in self.pos_emision_prob[sentence[i]]:
                        emi_prob= self.pos_emision_prob[sentence[i]][j]
                    prob = 0
                    for pos in self.pos_prob:
                        tran_prob=0.000000001
                        if j in self.pos_trans_prob and pos in self.pos_trans_prob[j]:
                            tran_prob=self.pos_trans_prob[j][pos]

                        prob += tao[pos][i-1] *tran_prob* emi_prob

                    tao[j][i]=prob
            else:
                for j in self.pos_prob:

                    emi_prob = 0.000000001
                    if sentence[i] in self.pos_emision_prob and j in self.pos_emision_prob[sentence[i]]:
                        emi_prob=self.pos_emision_prob[sentence[i]][j]
                    sum = 0
                    for pos in self.pos_prob:
                        sum=0
                        for pre_pos in self.pos_prob:

                            if pos not in self.pos_trans_prob:
                                self.pos_trans_prob[pos] = {pre_pos: 0.000000001}
                            elif pre_pos not in self.pos_trans_prob[pos]:
                                self.pos_trans_prob[pos][pre_pos] = 0.000000001

                            sum +=self.pos_trans_prob[j][pre_pos] * tao[pre_pos][i - 2]

                        sum +=self.pos_trans_prob[j][pos] * tao[pos][i - 1]

                    tao[j][i] =sum*emi_prob

        for i in range(len(sentence)):
            temp=[[tao[x][i],x] for x in self.pos_prob]
            maximum=max([tao[x][i] for x in self.pos_prob])
            for j in temp:
                if j[0]==maximum:
                    probability.append(j[0])
                    output[i]=j[1]

        return output

    def hmm_viterbi(self, sentence):
        node_prob={}
        previous_nodes={}
        for t in range(len(sentence)):
            for pos in self.pos_prob:
                if t==0:
                    prob=0.000000001
                    if pos in self.pos_initial_prob:
                        prob= self.pos_initial_prob[pos]
                    if sentence[t] in self.pos_emision_prob and pos in self.pos_emision_prob[sentence[t]]:
                        prob *=  self.pos_emision_prob[sentence[t]][pos]
                    else:
                        prob *= 0.000000001
                    node_prob[pos+ str(t)] = prob
                    previous_nodes[pos+ str(t)]="start"
                else:
                    max=0
                    for pre_pos in self.pos_prob:
                        prob=node_prob[pre_pos + str(t-1)]
                        if pos in self.pos_trans_prob and pre_pos in self.pos_trans_prob[pos]:
                            prob*= self.pos_trans_prob[pos][pre_pos]
                        else:
                            prob*=0.000000001
                        if sentence[t] in self.pos_emision_prob and pos in self.pos_emision_prob[sentence[t]]:
                            prob *= self.pos_emision_prob[sentence[t]][pos]
                        else:
                            prob *= 0.000000001
                        if max <= prob:
                            max=prob
                            node_prob[pos + str(t)] = prob
                            previous_nodes[pos + str(t)] = pre_pos

        max = 0
        t = len(sentence) - 1
        lastnode=""
        for pos in self.pos_prob:
            if max <=  node_prob[pos + str(t)]:
                max = node_prob[pos + str(t)]
                lastnode=pos
        node_list=[]
        for t in range( len(sentence)-1, -1, -1):
            node_list.append(lastnode)
            lastnode=previous_nodes[lastnode+ str(t)]
        node_list.reverse()


        return node_list


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

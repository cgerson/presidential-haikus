from spyre import server #spyre

import os
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import seaborn as sns
from nltk.probability import FreqDist
from nltk.probability import MLEProbDist
from nltk.probability import ConditionalFreqDist
from nltk import bigrams
from nltk.corpus import cmudict
from nltk import pos_tag


class MongoExample(server.App):

    def __init__(self):
        client = MongoClient(host="54.69.198.239",port=27017)
        self.col = client.Presidents.Docs
        self.d = cmudict.dict()
    
    title = "Presidential Documents- Analysis"

    inputs = [{     "type":'dropdown',
                    "label": 'President', 
                    "options" : [{"label": "Abraham Lincoln", "value":"Abraham Lincoln"},
                                 {"label": "Andrew Jackson", "value":"Andrew Jackson"},
                                 {"label": "Andrew Johnson", "value":"Andrew Johnson"},
                                 {"label": "Barack Obama", "value":"Barack Obama"},
                                 {"label": "Benjamin Harrison", "value":"Benjamin Harrison"},
                                 {"label": "Bill Clinton", "value":"William J. Clinton"},
                                 {"label": "Calvin Coolidge", "value":"Calvin Coolidge"},
                                 {"label": "Chester A. Arthur", "value":"Chester A. Arthur"},
                                 {"label": "Dwight D. Eisenhower", "value":"Dwight D. Eisenhower"},
                                 {"label": "Franklin D. Roosevelt", "value":"Franklin D. Roosevelt"},
                                 {"label": "Franklin Pierce", "value":"Franklin Pierce"},
                                 {"label": "George Bush", "value":"George Bush"},
                                 {"label": "George W. Bush", "value":"George W. Bush"},
                                 {"label": "George Washington", "value":"George Washington"},
                                 {"label": "Gerald R. Ford", "value":"Gerald R. Ford"},
                                 {"label": "Grover Cleveland", "value":"Grover Cleveland"},
                                  {"label": "Harry S. Truman", "value":"Harry S. Truman"},
                                 {"label": "Herbert Hoover", "value":"Herbert Hoover"},
                                 {"label": "James A. Garfield", "value":"James A. Garfield"},
                                  {"label": "James Buchanan", "value":"James Buchanan"},
                                 {"label": "James K. Polk", "value":"James K. Polk"},     
                                 {"label": "James Madison", "value":"James Madison"},
                                 {"label": "James Monroe", "value":"James Monroe"},
                                 {"label": "Jimmy Carter", "value":"Jimmy Carter"},
                                 {"label": "John Adams", "value":"John Adams"},
                                 {"label": "John F. Kennedy", "value":"John F. Kennedy"},
                                 {"label": "John Quincy Adams", "value":"John Quincy Adams"},
                                 {"label": "John Tyler", "value":"John Tyler"},
                                 {"label": "Lyndon B. Johnson", "value":"Lyndon B. Johnson"},
                                 {"label": "Martin van Buren", "value":"Martin van Buren"},
                                 {"label": "Millard Fillmore", "value":"Millard Fillmore"},
                                 {"label": "Richard Nixon", "value":"Richard Nixon"},
                                  {"label": "Ronald Reagan", "value":"Ronald Reagan"},
                                 {"label": "Rutherford B. Hayes", "value":"Rutherford B. Hayes"},
                                 {"label": "Theodore Roosevelt", "value":"Theodore Roosevelt"},
                                 {"label": "Thomas Jefferson", "value":"Thomas Jefferson"},
                                 {"label": "Ulysses S. Grant", "value":"Ulysses S. Grant"},
                                  {"label": "Warren G. Harding", "value":"Warren G. Harding"},   
                                  {"label": "William Henry Harrison", "value":"William Henry Harrison"},                      
                                  {"label": "William Howard Taft", "value":"William Howard Taft"},
                                  {"label": "William McKinley", "value":"William McKinley"},
                                  {"label": "Woodrow Wilson", "value":"Woodrow Wilson"},
                                  {"label": "Zachary Taylor", "value":"Zachary Taylor"}],
                    "key": 'president', #refer here for value chosen above
                    "action_id": "update_data"},
              {     "type":'radiobuttons',
                    "label": 'Speech', 
                    "options" : [ {"label": "State of the Union", "value":"SOU","checked":True},
                                  {"label": "Inaugural Address", "value":"Inaugurals"}],
                    "key": 'speech', #refer here for value chosen above
                    "action_id": "update_data"}]
              

    controls = [{   "type" : "hidden",
                    "id" : "update_data"},
                {    "type" : "button",
		     "id" : "button2",
		     "label" : "refresh"}]

    tabs = ["Plot","Table","Haiku","DrunkTopic"]

    outputs = [{ "type" : "plot",
                    "id" : "plot",
                    "control_id" : "update_data",
                    "tab" : "Plot"},
                { "type" : "table",
                    "id" : "table_id",
                    "control_id" : "update_data",
                    "tab" : "Table",
                    "on_page_load" : True },
               {"type" : "html",
		"id" : "html2",
		"control_id" : "button2",
		"tab" : "DrunkTopic"},
               {"type" : "html",
		"id" : "html1",
		"control_id" : "button2",
		"tab" : "Haiku"}]

    def loadData(self,params):
        """return filtered words according to inputs"""
        
        self.president = params['president']
        self.speech = params['speech']
        
        pipeline = [{"$match": {"name":self.president, "type":self.speech}},
                    {"$project": {"text": "$filtered_speech", "date":"$date"}},{"$limit":1}]

        filtered_words = []

        for i in self.col.aggregate(pipeline):
            filtered_words.extend(i['text'])

        return filtered_words

    def fDist(self,filtered_words):
        """return frequency distribution using filtered_words from loadData"""
        
        fdist = FreqDist() #frequency dist for number of words
        for word in filtered_words:
            fdist[word] += 1
        return fdist
            
    def speechCt(self,params):
        """return count of number of speeches for given president"""
        
        self.president = params['president']
        self.speech = params['speech']
        
        pipeline_ct = [{"$match":{"name":self.president,"type":self.speech}},{"$group":{"_id":"type",
                                                              "count":{"$sum":1}}}]
        for i in self.col.aggregate(pipeline_ct):
            ct = i['count']

        return ct

    def cDist(self,params):
        """return conditional freq distribution using filtered_words from loadData"""
        president = params['president']
        speech = params['speech']
        
        pipeline = [{"$match": {"name":president, "type":speech}},
                    {"$project": {"tags": "$filtered_speech_tags"}},{"$limit":1}]

        tags = []
        for i in self.col.aggregate(pipeline):
            tags.extend(i['tags'])
        
        cfdist = ConditionalFreqDist() #conditioned on pos_tag
        for word,tag in tags:
            condition =  tag #specify condition to group frequencies by 
            cfdist[condition][word] += 1

        VB = MLEProbDist(cfdist.get("VBP"))
        NN = MLEProbDist(cfdist.get("NN"))
        JJ = MLEProbDist(cfdist.get("JJ"))

        return VB,NN,JJ
    
    def getData(self,params):
        fdist = self.fDist(self.loadData(params))
        max_freq = fdist.most_common(20)
        words_unzipped,count_unzipped = zip(*max_freq)
        f = [fdist.freq(w) for w in words_unzipped]
        z = zip(words_unzipped,count_unzipped,f)
        df = pd.DataFrame(z)
        df.columns = ['words','count','frequency']
        return df

    def getPlot(self, params):
        ct = self.speechCt(params)
        speech = params['speech']
        if speech == "Inaugurals":
            speech = "Ingaugural"
            
        df = self.getData(params).set_index('words')
        plt_obj = df.plot(kind='bar',rot=45,legend=True,secondary_y=['frequency'])
        plt_obj.set_ylabel("Frequency in texts")
        if ct > 1:
            plt_obj.set_title("{0}: {1} {2} speeches".format(self.president,ct,speech))
        else:
            plt_obj.set_title("{0}: {1} {2} speech".format(self.president,ct,speech))
        fig = plt_obj.get_figure()
        return fig

    def html2(self,params):
        filtered_words = self.loadData(params)
        fdist = self.fDist(filtered_words)
        
        my_bigrams = bigrams(filtered_words)
        cfd = ConditionalFreqDist(my_bigrams)

        common_words,counts = zip(*fdist.most_common(20))
        random_word = np.random.choice(common_words)
        
        def generate_model(cfdist, word, num=15):
            words = []
            for i in range(num):
                words.append(word)
                word = cfdist[word].max()
            return words

        result =  "<p>{0}</p>".format(" ".join(generate_model(cfd,random_word)))
        return result
    
    def html1(self,params):
        VB,NN,JJ = self.cDist(params)
        
        def makeHaiku():    

            def generateLine(line):
                if line == 1:
                    return NN.generate(),VB.generate()
                if line == 2:
                    return VB.generate(),JJ.generate(),NN.generate()
                if line == 3:
                    return JJ.generate(),NN.generate()

            def sumSyl(*args):
                sum_syl = 0
                for word in args:
                    sum_syl+=self.nsyl(word)[0]
                return sum_syl
            
            sum1,sum2,sum3 = 0,0,0

            while sum1!=5:
                n1, v1 = generateLine(1)
                sum1 = sumSyl(n1,v1)
                if sum1 == 5:
                    break
            while sum2!=7:
                v2,a1,n2 = generateLine(2)
                sum2= sumSyl(v2,a1,n2)
                if sum2 == 7:
                    break
            while sum3!=5:
                a2,n3 = generateLine(3)
                sum3 = sumSyl(a2,n3)
                if sum3 == 5:
                    break

            l = ["{0} {1}".format(n1,v1),"{0} a {1} {2}".format(v2,a1,n2),"{0} {1}".format(a2,n3)]
            return l

        haiku = makeHaiku()
        return "<p>".join(haiku)

    def nsyl(self,w): #count syllables
        try: 
            result = [len(list(y for y in x if y[-1].isdigit())) for x in self.d[w]]
        except:
            result = [8] #so that it exceeds syllable limit and starts over
        return result
    
if __name__ == '__main__':
    app = MongoExample()
    app.launch(port=8000)
    #app.launch(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))

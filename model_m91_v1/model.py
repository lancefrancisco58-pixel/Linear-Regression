#DATE: 11/22/2026
#-my 1st ML model, buggy sometimes but hey it's my 1st one

import json as js 
import os
import random as r

class model_m91:
    def __init__(self, LR = 10):
        self.hours = []
        self.scores = []
        self.hour_score = {}
        self.m = 0
        self.b = 0
        self.learning_rate = LR/100
        
        #file
        base_folder = os.path.dirname(__file__)
        root_path = os.path.dirname(base_folder)
        self.data_file = os.path.join(root_path, "training data", "data.json")
        self.bmx_file = os.path.join(root_path, "training data", "bmx.json")
        
        self._load()
        
    def predict(self, hour):
        #predicts the score with the given hour var
        self._save()
        return self.m*hour + self.b
    
    def train(self, epochs, sample_size=10):
        #trains the parameters m and b, uses minibatch to make training faster
        for epoch in range(1, epochs+1):
            batch = r.sample(list(zip(self.hours, self.scores)), sample_size)
            total_error = 0
            for hour, score in batch:
                error = self.predict(hour)-score
                total_error += abs(error)
                self.m = self.m-self.learning_rate*(error*hour)
                self.b = self.b-self.learning_rate*error
                self.m_b["m"] = self.m
                self.m_b["b"] = self.b
            if epoch % (sample_size) == 0:
                print(total_error/sample_size)
                self._save()
    
    def _load(self):
        #loads the parameters and training data
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as file:
                data = js.load(file)
            for hour, score in data:
                self.hours.append(int(hour))
                self.scores.append(int(score))
        
        if os.path.exists(self.bmx_file):
            with open(self.bmx_file, "r") as file:
                self.m_b = js.load(file)
                self.m = self.m_b["m"]
                self.b = self.m_b["b"]
        
    def _save(self):
        #saves the parameters to a cache file for further use, no atomic save yet. Implementing it in v2
        with open(self.bmx_file, "w") as file:
            js.dump(self.m_b, file)
    
if __name__ == "__main__":
    mod = model_m91()
    print(mod.predict(10))
    #mod.train(100)
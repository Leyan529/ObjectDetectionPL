import os
from torchinfo import summary
import torch
import numpy as np
import matplotlib.pyplot as plt

def saveDetail(self):
    model_stats = summary(self.cuda(), self.sample)
    # model_stats = self.summarize(mode='full')
    os.makedirs(self.dir, exist_ok=True)
    summary_str = str(model_stats)
    summaryfile = os.path.join(self.dir, 'summary.txt')
    summary_file = open(summaryfile, 'w', encoding="utf-8")
    summary_file.write(summary_str + '\n')
    summary_file.close()

def write_Best_model_path(self):  
    best_model_path = self.checkpoint_callback.best_model_path
    if (best_model_path!=""):
        best_model_file = os.path.join(self.dir, 'best_model_path.txt')
        best_model_file = open(best_model_file, 'w', encoding="utf-8")
        print("\nWrite best_model_path : %s \n"% best_model_path)
        best_model_file.write(best_model_path + '\n')
        best_model_file.close()

def read_Best_model_path(self):  
    best_model_file = os.path.join(self.dir, 'best_model_path.txt')
    if os.path.exists(best_model_file):
        best_model_file = open(best_model_file, 'r', encoding="utf-8")
        best_model_path = best_model_file.readline().strip()
        print("\nLoad best_model_path : %s \n"% best_model_path)
        self.load_from_checkpoint(checkpoint_path=best_model_path, classes=self.classes, data_name=self.data_name)
        best_model_file.close()    
    else:
        print("No model can load \n")
        if not os.path.exists(best_model_file):
            self.saveDetail()



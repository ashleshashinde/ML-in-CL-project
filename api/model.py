#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:19:51 2019

@author: ashleshashinde
"""
import csv
import pandas as pd


feedback_dict={}
new_dict={}
column_list = ['title', 'artist', 'lyrics','genre', 'mood']

def max_voting(feedback_dict):
    for key in feedback_dict:
        value=feedback_dict[key]
        happy=0
        sad=0
        for feedback in value:
                mood=feedback[3] 
                genre=feedback[4]
                score=feedback[5]
                if mood=="happy" and score == '0':
                    happy+=1
                elif mood=="happy" and score == '1':
                    sad+=1
                elif mood=="sad" and score == '1':
                    happy+=1
                elif mood=="sad" and score == '0':
                    sad+=1
        print( key ,sad,happy)
        if happy > sad:
             new_dict[key]=(feedback[0],feedback[1],feedback[2],feedback[4],"happy")
        else:
             new_dict[key]=(feedback[0],feedback[1],feedback[2],feedback[4],"sad")
    return(new_dict)

 

with open('feedback.csv','r') as file:
    reader = csv.reader(file)
    for rows in reader:
        key= rows[1]+" "+rows[2]
        if key in feedback_dict:
            feedback_dict[key].append((rows[1],rows[2],rows[3],rows[4],rows[5],rows[7]))
#            print(rows[7])
        else:
            feedback_dict[key] = [(rows[1],rows[2],rows[3],rows[4],rows[5],rows[7])]

new_dict=max_voting(feedback_dict)
new_df = pd.DataFrame.from_dict(new_dict, orient = 'index',columns=column_list)

            
#    mydict = {rows[1]+" "+rows[2]:(rows[4],rows[6]) for rows in reader}
    
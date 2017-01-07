# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 21:55:58 2016

@author: Sachi Angle
"""

import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\Sachi Angle\\Desktop\diab.csv')
data = data[data["Percentage Management of diabetes"] != "*"]

data = data.drop(192, 0)
data = data.drop(193,0)
data = data.drop(194, 0)

conv = []
for i in data["Country"]:
    if i == 'England':
        conv.append(1)
    else:
        conv.append(2)
data["Country-bin"] = conv

conv = []
for i in data["Hospital/Site"]:
    if i == 'Hospital':
        conv.append(1)
    else:
        conv.append(2)
data["Hospital/Site-bin"] = conv
    

strs = ["Hospital Beds (HC)", "Diabetic Patients (BA)"]
for st in strs:
    con = []
    for i in range(len(data) + 4):
        if i in [26, 192, 193, 194]:
            continue
        x = 0.0
        if len(data.loc[i][st]) >= 4 and data.loc[i][st][1] == ',':
            s = ""
            s = data.loc[i][st][0] + data.loc[i][st][2:]
        else:
            s = data.loc[i][st]
        x = float(s)
        con.append(x)
    
    data = data.drop(st, 1)
    data[st] =con

input_data = pd.concat([data['Country-bin'], data['Hospital/Site-bin'], data['Hospital Beds (HC)'], data['Diabetic Patients (BA)'], data['Prevalence of diabetes'], data['Percentage emergency'], data['Percentage other medical'], data['Percentage Non Medical'], data['Nursing hours'], data['Consultant hours'], data['Dietician hours'], data['Podiatrist hours'], data['Pharmacist hours'], data[' Blood Glucose monitoring Days '], data[' Appropriate Days '], data[' Good Diabetes Days '], data['Visit by specialist diabetes team'], data['Medication errors'], data['prescription errors'], data['management errors'], data['Insulin Errors'], data['Admitted with foot disease'], data['Seen by the MDT within 24 hours'], data['Foot Risk Assessment completed within 24 hours'], data['Foot risk assessment after 24 hours only'], data['Foot Risk Assessment completed during the hospital stay'], data['% Severe Hypo'], data['% Minor Hypo'], data['Meals Suitable'], data['Meals Timing'], data['Meals Choice'], data['Staff Knowledge - Answer Qs'], data['Staff Knowledge - Emotional support'], data['Staff Knowledge - Work together'], data['Staff aware of diabetes'], data['All or most staff know enough about diabetes']], axis=1, keys=['Country-bin', 'Hospital/Site-bin', 'Hospital Beds (HC)', 'Diabetic Patients (BA)', 'Prevalence of diabetes', 'Percentage emergency', 'Percentage other medical','Percentage Non Medical', 'Nursing hours','Consultant hours', 'Dietician hours', 'Podiatrist hours', 'Pharmacist hours', ' Blood Glucose monitoring Days ',' Appropriate Days ', ' Good Diabetes Days ','Visit by specialist diabetes team', 'Medication errors','prescription errors', 'management errors', 'Insulin Errors','Admitted with foot disease', 'Seen by the MDT within 24 hours','Foot Risk Assessment completed within 24 hours','Foot risk assessment after 24 hours only','Foot Risk Assessment completed during the hospital stay', '% Severe Hypo', '% Minor Hypo', 'Meals Suitable', 'Meals Timing','Meals Choice', 'Staff Knowledge - Answer Qs','Staff Knowledge - Emotional support','Staff Knowledge - Work together', 'Staff aware of diabetes','All or most staff know enough about diabetes'])

percentages = ["Prevalence of diabetes", "Percentage emergency", "Percentage other medical", "Percentage Non Medical", "% Severe Hypo", "% Minor Hypo", "Meals Suitable", "Meals Timing", "Meals Choice", "Staff Knowledge - Answer Qs", "Staff Knowledge - Emotional support", "Staff Knowledge - Work together", "Staff aware of diabetes", "All or most staff know enough about diabetes", "Visit by specialist diabetes team", "Medication errors", "prescription errors", "management errors", "Insulin Errors", "Admitted with foot disease", "Seen by the MDT within 24 hours", "Foot Risk Assessment completed within 24 hours", "Foot risk assessment after 24 hours only", "Foot Risk Assessment completed during the hospital stay"]
for s in percentages:
    per = []
    for i in input_data[s]:
        x = 0.0
        if str(i) != 'nan':
            x = float(i[:-1])
        per.append(x)
    #print per
    input_data = input_data.drop(s, 1)
    input_data[s] = per



md = []
for i in data["Percentage Management of diabetes"]:
    x = 0.0
    x = float(i[:-1])
    md.append(x)
    
md = np.array(md)
md_class = np.zeros(len(data))
md_class[md > 15.0] = 1.0
md_data = input_data       

from sklearn.cross_validation import train_test_split
(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(md_data, md_class, train_size=0.75, random_state=1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(training_inputs, training_classes)
MD_sc = dtc.score(testing_inputs, testing_classes)
MD_features = dtc.feature_importances_

ab = []
for i in data["Able to take control of diabetes care"]:
    x = 0.0
    x = float(i[:-1])
    ab.append(x)

ab_class = np.zeros(len(data))
ab_class[ab > 75.0] = 1.0
ab_data = input_data   

(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(ab_data, ab_class, train_size=0.75, random_state=1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(training_inputs, training_classes)
AB_sc = dtc.score(testing_inputs, testing_classes)
AB_features = dtc.feature_importances_

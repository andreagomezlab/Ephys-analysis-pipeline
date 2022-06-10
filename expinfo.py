#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#functions used to get experiment info on cells and mice, such as genotype, conditions, etc

import pandas as pnd
from IPython.core.display import HTML

def get_exp_table(excel):
    '''Obtains table with experiment type for each abf file of a cell from I-clamp. Returns dictionary where keys are
    abf filenames and items are strings with experiment types and drug labels. Assumes excel file is in current
    directory.
    
    arguments:
    excel = str, name of excel file with metadata to turn into dictionary. 
    '''
    exp_table = pnd.read_excel(excel)
    exp_table.columns = exp_table.columns.str.replace(' ','_',regex=True)
    exp_table['Experiment'] = exp_table['Experiment'].str.rstrip('_')
    exp_table = exp_table[['File','Experiment','notes']]
    return exp_table

def get_mouse_table(excel):
    '''Obtains table with experiment type for each abf file of a cell from I-clamp. Returns dictionary where keys are
    abf filenames and items are strings with experiment types and drug labels. Assumes excel file is in current
    directory.
    
    arguments:
    excel = str, name of excel file with information of slices, stuff like the slice
    '''
    mouse_table = pnd.read_excel(excel,header=None)
    mouse_table.columns =['cell_name','drug_condition','hippocampal_region']
    mouse_table.columns = mouse_table.columns.str.replace(' ','_',regex=True)
    return mouse_table
    
def get_I_step_interval(abf):
    '''Parses through raw text to obtain mV step interval in membrane test.Assumes that there is only a single 
    voltage step in voltage clamp experiments.
    
    arguments:
    raw_str = str extracted from csv with get_metadata_values
    '''
    step_start = []
    step_end = []
    abf.setSweep(0,channel=0)
    raw_str = str(abf.sweepEpochs)
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    for i in range(parsed_total):
        if 'Step 0.00' not in parsed_str[i]:
            values = parsed_str[i][parsed_str[i].find("[")+1:parsed_str[i].find("]")]
            step_points = (values.split(':'))
            step_start, step_end = int(step_points[0]), int(step_points[1])    
    return step_start, step_end

def get_sag_step_intervals(abf):
    '''Parses through raw text to obtain current step interval in sag test.
     Assumes that there is only a single current amplitude for all steps.
     '''
    abf.setSweep(0,channel=0)
    raw_str = str(abf.sweepEpochs)
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    for i in range(parsed_total):
        if 'Step 0.00' not in parsed_str[i]:
            # also stolen: this extracts string inside []
            values = parsed_str[i][parsed_str[i].find("[")+1:parsed_str[i].find("]")]
            j = i - 1
            values2 = parsed_str[j][parsed_str[j].find("[")+1:parsed_str[j].find("]")]
            
    step_points = values.split(':')
    bl_points = values2.split(':')
    step_start, step_end = int(step_points[0]), int(step_points[1])
    bl_start, bl_end = int(bl_points[0]), int(bl_points[1])
    return step_start, step_end, bl_start, bl_end

def get_chirp_step_intervals(raw_str):
    '''Parses through raw text to obtain current step interval in sag test.Assumes that there is only a single 
    current amplitude for all steps.
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No'
    '''
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    values = [None] * parsed_total
    for i in range(parsed_total):
        values[i] = parsed_str[i][parsed_str[i].find("[")+1:parsed_str[i].find("]")]        
    step_points = [value.split(':') for value in values]
    start_chunk = [int(i) for i in step_points[0]]
    end_chunk = [int(i)for i in step_points[-1]]
    end_chunk[0] = end_chunk[1] - start_chunk[1]
    step_start, step_end = start_chunk[1], end_chunk[0]
    return step_start, step_end

def get_rheobase_step_intervals(raw_str):
    '''Parses through raw string to obtain current step interval in rheobase test.
    Assumes that the third epoch is the only one of interest.'''
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    values = [None] * parsed_total
    for i in range(parsed_total):
        values[i] = parsed_str[i][parsed_str[i].find("[")+1:parsed_str[i].find("]")]        
    step_points = [value.split(':') for value in values]
    current_step = [int(i) for i in step_points[2]] # just the third epoch is fine
    step_start, step_end = current_step[0], current_step[1]
    return step_start, step_end

def get_V_step_interval(raw_str):
    '''Parses through raw text to obtain mV step interval in membrane test. Assumes that there is only a single 
    voltage step in voltage clamp experiments.
    
    arguments:
    raw_str = str extracted from csv with get_metadata_values
    '''
    step_start=[] 
    step_end=[]
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    for i in range(parsed_total):
        if 'Step 0.00' not in parsed_str[i]:
            values = parsed_str[i][parsed_str[i].find("[")+1:parsed_str[i].find("]")]
            step_points = (values.split(':'))
            step_start, step_end = int(step_points[0]), int(step_points[1])
    return step_start, step_end

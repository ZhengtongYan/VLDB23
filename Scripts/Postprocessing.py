#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decimal import *
import numpy as np
from math import prod
import itertools

import os
import pathlib
import pickle
import json


# In[2]:


def get_rounded_val(val, decimal_pos = 4):
    return Decimal(val).quantize(Decimal(10) ** -decimal_pos, rounding=ROUND_HALF_EVEN)

def get_delta_thres(thres):
    delta_thres = np.diff(thres)
    delta_thres = np.insert(delta_thres, 0, thres[0])
    return delta_thres

## Prints the assignment of all activated tioT0 and tiiTJ variables which corresponds to the resulting join order
def get_join_tree_leaves(sample, num_tables):
    
    num_tiio_variables = num_tables * (num_tables-1)
    
    ## Extract variable assignments from the response
    
    inner_operand_for_join = {}
    # Fetch inner operands
    for i in range(num_tiio_variables, num_tiio_variables*2):
        #key = 'x[' + str(i) + ']'
        key = i
        if sample[key] == 1:
            table_index = (i // (num_tables-1)) - num_tables
            join_index = i % (num_tables-1)
            if join_index in inner_operand_for_join:
                # Erroneous solution, since another table has already been selected for this join
                return None
            inner_operand_for_join[join_index] = table_index
      
    outer_table_index = None
    for i in range(num_tables):
        if i not in inner_operand_for_join.values():
            outer_table_index = i
            break # Outer operand found -> abort search
    
    join_order = []
    join_order.append(outer_table_index)
    for i in range(num_tables-1):
        if i in inner_operand_for_join:
            join_order.append(inner_operand_for_join[i])
            
    if len(join_order) != len(set(join_order)):
        # Invalid solution: At least one relation has been selected for multiple leaves of the join tree
        return None
    
    if len(join_order) != num_tables:
        # Invalid solution: For at least one leaf of the join tree, no relation has been selected
        return None
    
    return join_order

def get_combined_pred_sel(join, pred, pred_sel, join_order):
    relations = np.array(join_order[0:join+1])
    pred_indices = []
    for comb in itertools.combinations(relations, 2):
        comb = tuple(sorted(comb))
        if comb in pred:
            ind = pred.index(comb)
            pred_indices.append(ind)
    combined_sel = prod(pred_sel[i] for i in pred_indices)
    return combined_sel

def calculate_intermediate_cardinality_for_join(join, card, pred, pred_sel, join_order):
    raw_cardinality = prod(card[join_order[i]] for i in range(join+1))
    combined_sel = get_combined_pred_sel(join, pred, pred_sel, join_order)
    return raw_cardinality * combined_sel

def get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres):
    delta_thres = get_delta_thres(thres)
    total_costs = np.int32(0)
    for j in range(1, len(card)-1): # Ignore first join
        int_card = calculate_intermediate_cardinality_for_join(j, card, pred, pred_sel, join_order)
        dec_int_card = get_rounded_val(int_card)
        for i in range(len(thres)):
            if dec_int_card > thres[i]:
                total_costs = total_costs + delta_thres[i]
    return total_costs.item()

def postprocess_IBMQ_response(response, card, pred, pred_sel, thres, optimal_join_order_costs = 0):
        
    best_join_order = None
    best_join_order_costs = 0
    valid_ratio = 0
    optimal_ratio = 0
            
    for i in range(len(response.samples)):
        sample = response.samples[i].x
        energy = response.samples[i].fval
        prob = response.samples[i].probability

        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            valid_ratio = valid_ratio + prob
            
            if best_join_order is None or best_join_order_costs > costs:
                best_join_order = join_order
                best_join_order_costs = costs
            
            if costs == optimal_join_order_costs:
                optimal_ratio += prob
    
    return best_join_order, best_join_order_costs, valid_ratio, optimal_ratio

def postprocess_DWave_response(response, card, pred, pred_sel, thres, optimal_join_order_costs = 0):
   
    val_solution_counter = 0
    opt_solution_counter = 0
    best_join_order = None
    best_join_order_costs = 0
        
    #for i in range(len(response.record)):
    #    (sample, energy, occ, chain) = response.record[i]
    for (sample, energy, occ) in response:
        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            
            val_solution_counter += 1
            
            # If the join order is valid (not None), fetch the corresponding actual costs
            costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            
            if best_join_order is None or best_join_order_costs > costs:
                best_join_order = join_order
                best_join_order_costs = costs
            
            if costs == optimal_join_order_costs:
                opt_solution_counter += 1
        
    valid_ratio = val_solution_counter / len(response)
    optimal_ratio = opt_solution_counter / len(response)
        
    return best_join_order, best_join_order_costs, valid_ratio, optimal_ratio


# In[3]:


def load_from_path_pickle(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
            return data

def load_from_path(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file) as file:
            data = json.load(file)
            return data
        
def format_loaded_pred(pred):
    form_pred = []
    for p in pred:
        form_pred.append(tuple(p))
    return form_pred
        
def load_problem_from_disc(problem_path):

    card = load_from_path(problem_path + "/card.txt")      
    pred = format_loaded_pred(load_from_path(problem_path + "/pred.txt"))    
    pred_sel = load_from_path(problem_path + "/pred_sel.txt")
    thres = load_from_path(problem_path + "/thres.txt")    
    
    return card, pred, pred_sel, thres

def test_parsing():
    annealing_times = [20, 60, 100]
    graph_types = ['CHAIN', 'STAR', 'CYCLE']
    for graph_type in graph_types:
        for at in annealing_times:
            for i in range(3, 6):
                if graph_type == 'STAR' and i == 3:
                    continue
                valid_ratios = []
                opt_ratios = []
                for j in range(20):
                    problem_path = 'Problems/' + graph_type + '_query/' + str(i) + 'relations'
                    card, pred, pred_sel, thres = load_problem_from_disc(problem_path)
                    response_path = 'Results/Data/' + str(at) + '_AT/' + graph_type + '_query/' + str(i) + 'relations/' + str(j) + "/best_embedding_response.txt"
                    response = load_from_path(response_path)
                    best_join_order, best_join_order_costs, valid_ratio, optimal_ratio = postprocess_DWave_response(response, card, pred, pred_sel, thres, optimal_join_order_costs = 0)
                    valid_ratios.append(valid_ratio)
                    opt_ratios.append(optimal_ratio)
                mean_valid_occ = get_rounded_val(np.mean(valid_ratios)*1000, 2)
                mean_opt_occ = get_rounded_val(np.mean(opt_ratios)*1000, 2)
                print((at, graph_type, i, mean_valid_occ, mean_opt_occ))


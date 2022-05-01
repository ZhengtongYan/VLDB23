#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decimal import *
import numpy as np
from math import prod
import itertools


# In[2]:


def get_rounded_val(val):
    return Decimal(val).quantize(Decimal(10) ** -4, rounding=ROUND_HALF_EVEN)

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

def print_IBMQ_result(valid_solutions, lowest_cost, best_sample, join_order, valid_ratio, opt_ratio):
    print("Number of valid solutions: " + str(len(valid_solutions)))
    print("Globally lowest cost: " + str(lowest_cost))
    print("Join Order:")
    print(join_order)
    print("Valid ratio:")
    print(valid_ratio)
    print("Optimal ratio:")
    print(opt_ratio)
    print("")

def evaluate_IBMQ_solutions(valid_solutions, valid_indices, optimal_solution, num_card, response):
    lowest_cost = np.min(valid_solutions)
    best_sample = valid_indices[valid_solutions.index(lowest_cost)]
    join_order = get_join_tree_leaves(response.samples[best_sample].x, num_card)
    return lowest_cost, best_sample, join_order

def postprocess_IBMQ_response(response, card, pred, pred_sel, thres, optimal_solution, weight_a, include_remedied_solutions=True):
   
    if include_remedied_solutions:
        print("Consider valid and remedied solutions: ")
        print("")
    else:
        print("Consider only valid solutions: ")
        print("")
        
    valid_solutions = []
    valid_indices = []
    valid_ratio = 0
    opt_ratio = 0
    
    cost_dict = {}
    
    rounded_weight_a= get_rounded_val(weight_a)
    
    for i in range(len(response.samples)):
        #(sample, energy, occ, chain) = response.record[i]
        sample = response.samples[i].x
        energy = response.samples[i].fval
        prob = response.samples[i].probability
        if not include_remedied_solutions and get_rounded_val(energy) >= rounded_weight_a:
            continue
        # If remedied solutions are to be included, the join order is fetched despite constraint violations
        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            # If the join order is valid (not None), fetch the corresponding actual costs
            costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            valid_solutions.append(costs)
            valid_indices.append(i)
            valid_ratio = valid_ratio + prob
            if costs == optimal_solution:
                opt_ratio += prob
            if not cost_dict.values():
                cost_dict[i] = costs
            if get_rounded_val(costs) < get_rounded_val(np.min(list(cost_dict.values())).item()):
                cost_dict[i] = costs
        
    opt_sample_index = None
    if get_rounded_val(valid_ratio) == 0:
        print("No solution was valid.")
        print("")
    else:
        lowest_cost, best_sample, join_order = evaluate_IBMQ_solutions(valid_solutions, valid_indices, optimal_solution, len(card), response)        
        ## TODO better solution: Check for a known lowest / optimal cost
        if lowest_cost == optimal_solution:
            opt_sample_index = best_sample
        print_IBMQ_result(valid_solutions, lowest_cost, best_sample, join_order, valid_ratio, opt_ratio)
    
    return valid_ratio, cost_dict, opt_sample_index

def print_DWave_result(valid_solutions, lowest_cost, best_sample, join_order, valid_ratio):
    print("Number of valid solutions: " + str(len(valid_solutions)))
    print("Globally lowest cost: " + str(lowest_cost))
    print("Found at sample " + str(best_sample))
    print("Join Order:")
    print(join_order)
    print("Valid ratio:")
    print(valid_ratio)
    print("")
    
def evaluate_DWave_solutions(valid_solutions, valid_indices, num_card, response):
    lowest_cost = np.min(valid_solutions)
    best_sample = valid_indices[valid_solutions.index(lowest_cost)]
    join_order = get_join_tree_leaves(response.record[best_sample][0], num_card)
    valid_ratio = round(len(valid_solutions) / len(response.record), 4)
    return lowest_cost, best_sample, join_order, valid_ratio

def postprocess_DWave_response(response, card, pred, pred_sel, thres, weight_a, include_remedied_solutions=True):
   
    if include_remedied_solutions:
        print("Consider valid and remedied solutions: ")
        print("")
    else:
        print("Consider only valid solutions: ")
        print("")
        
    valid_solutions = []
    valid_indices = []
    cost_dict = {}
    opt_solution_counter = 0
    
    rounded_weight_a= get_rounded_val(weight_a)
    
    for i in range(len(response.record)):
        (sample, energy, occ, chain) = response.record[i]
        if not include_remedied_solutions and get_rounded_val(energy) >= rounded_weight_a:
            continue
        # If remedied solutions are to be included, the join order is fetched despite constraint violations
        join_order = get_join_tree_leaves(sample, len(card))
        if join_order is not None:
            # If the join order is valid (not None), fetch the corresponding actual costs
            costs = get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
            valid_solutions.append(costs)
            if costs == 0:
                opt_solution_counter += 1
            valid_indices.append(i)
            if not cost_dict.values():
                cost_dict[i] = costs
            if get_rounded_val(costs) < get_rounded_val(np.min(list(cost_dict.values())).item()):
                cost_dict[i] = costs
        
    valid_ratio = None
    opt_ratio = opt_solution_counter / len(response.record)
    opt_sample_index = None
    if not valid_solutions:
        print("No solution was valid.")
        print("")
        valid_ratio = 0
    else:
        lowest_cost, best_sample, join_order, ret_valid_ratio = evaluate_DWave_solutions(valid_solutions, valid_indices, len(card), response)
        ## TODO better solution: Check for a known lowest / optimal cost
        if lowest_cost == 0.0:
            opt_sample_index = best_sample
        valid_ratio = ret_valid_ratio
        print_DWave_result(valid_solutions, lowest_cost, best_sample, join_order, valid_ratio)
    
    return valid_ratio, opt_ratio, cost_dict, opt_sample_index


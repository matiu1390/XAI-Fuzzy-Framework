import re

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from modules.fuzzy_load import *

# Fork note: standalone validation script combining parsing, inference, custom rules, and verbose defuzzification plots (centroid/bisector).


### LOADS ####

def read_variables(file):
    '''
    Parses the input fuzzy variables from the knowledge base into a dictionary. 
    
        Args:
            file(str): the input filename
            
        Returns:
            fuzzy_vars(dict): the dictionary of variables along with their 4-tuple representations.
            
    '''


    with open(file) as fp:
        line = fp.readline()
        fuzzy_vars = {}
        while line:
            text = line.strip()
            text = str.split(text, " ")
            if len(text) == 1 and 'Rule' not in text[0] and text[0] != '':
                fuzzy_categories = {}
                var_name = text[0]
                fp.readline()
                line_cat = fp.readline()
                while len(line_cat.strip()) > 0:
                    category = line_cat.strip()
                    category_values = str.split(category, ' ')
                    cat_name = category_values[0].strip()
                    print(category_values)
                    fuzzy_set = [float(category_values[1]), float(category_values[2]), float(category_values[3]),
                                 float(category_values[4])]
                    fuzzy_categories[str(cat_name).strip()] = eval(str(fuzzy_set))
                    line_cat = fp.readline()

                if len(fuzzy_categories) > 0:
                    fuzzy_vars[str(var_name)] = fuzzy_categories
            line = fp.readline()
        return fuzzy_vars



### DEFUZZIFIER ####3

def defuzzify_centroid(activation_dict, vmfx_list):
    '''
    Estimates the defuzzified value using the centroid method.
    The membership activations are aggregated based on their maximum value within the range.
    This area/centroid estimator is largely based on scikit-fuzzy's centroid method:https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/defuzzify/defuzz.py

        Args:
            activation_dict(dict): membership activation values throughout the range
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions

        Returns:
            result(np.float32): estimated defuzzified value
            conseq_range(np.array): the range of possible values for the consequent variable aggregated_mfx(np.array): the aggregated activation functions for each rule
            
    '''

    aggregated_mfx = np.zeros_like(list(activation_dict.values())[0])

    for i in range(len(aggregated_mfx)):
        max_val = 0.0
        for k, v in activation_dict.items():
            if v[i] > max_val:
                max_val = v[i]
        aggregated_mfx[i] = max_val

    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_range = vmfx['range']
            conseq_name = vmfx['name']

    sum_centroid_area = 0.0
    sum_area = 0.0
    result = 0.0

    if conseq_range is not None:

        # If the membership function is a singleton fuzzy set:
        if len(conseq_range) == 1:
            result = conseq_range[0] * aggregated_mfx[0] / np.float32(aggregated_mfx[0])

        else:
            # else return the sum of centroid*area/sum of area
            for i in range(1, len(conseq_range)):
                x1 = conseq_range[i - 1]
                x2 = conseq_range[i]
                y1 = aggregated_mfx[i - 1]
                y2 = aggregated_mfx[i]

                # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
                if not (y1 == y2 == 0.0 or x1 == x2):
                    if y1 == y2:  # rectangle
                        centroid = 0.5 * (x1 + x2)
                        area = (x2 - x1) * y1
                    elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                        centroid = 2.0 / 3.0 * (x2 - x1) + x1
                        area = 0.5 * (x2 - x1) * y2
                    elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                        centroid = 1.0 / 3.0 * (x2 - x1) + x1
                        area = 0.5 * (x2 - x1) * y1
                    else:
                        centroid = (2.0 / 3.0 * (x2 - x1) * (y2 + 0.5 * y1)) / (y1 + y2) + x1
                        area = 0.5 * (x2 - x1) * (y1 + y2)

                    sum_centroid_area += centroid * area
                    sum_area += area

            result = np.round(sum_centroid_area / np.float32(sum_area), 2)

    print('Centroid defuzzified value for {}:{}'.format(conseq_name, result))
    return result, conseq_range, aggregated_mfx


def defuzzify_bisector(activation_dict, vmfx_list):
    '''
    Estimates the defuzzified value using the bisector method.
    The membership activations are aggregated based on their maximum value within the range.
    This area/subarea estimator is largely based on scikit-fuzzy's bisector method:https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/defuzzify/defuzz.py

        Args:
            activation_dict(dict): membership activation values throughout the range
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions

        Returns:
            result(np.float32): estimated defuzzified value
            conseq_range(np.array): the range of possible values for the consequent variable aggregated_mfx(np.array): the aggregated activation functions for each rule
            
    '''


    aggregated_mfx = np.zeros_like(list(activation_dict.values())[0])

    for i in range(len(aggregated_mfx)):
        max_val = 0.0
        for k, v in activation_dict.items():
            if v[i] > max_val:
                max_val = v[i]
        aggregated_mfx[i] = max_val

    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_range = vmfx['range']
            conseq_name = vmfx['name']

    sum_area = 0.0
    acc_area = [0.0] * (len(conseq_range) - 1)
    result = 0.0

    if conseq_range is not None:

        # If the membership function is a singleton fuzzy set:
        if len(conseq_range) == 1:
            result = conseq_range[0]

        else:
            # else return the sum of centroid*area/sum of area
            for i in range(1, len(conseq_range)):
                x1 = conseq_range[i - 1]
                x2 = conseq_range[i]
                y1 = aggregated_mfx[i - 1]
                y2 = aggregated_mfx[i]

                # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
                if not (y1 == y2 == 0.0 or x1 == x2):
                    if y1 == y2:  # rectangle
                        area = (x2 - x1) * y1
                    elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                        area = 0.5 * (x2 - x1) * y2
                    elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                        area = 0.5 * (x2 - x1) * y1
                    else:
                        area = 0.5 * (x2 - x1) * (y1 + y2)

                    sum_area += area
                    acc_area[i - 1] = sum_area

            index = np.nonzero(np.array(acc_area) >= sum_area / 2.0)[0][0]

            if index == 0:
                subarea = 0
            else:
                subarea = acc_area[index - 1]

            x1 = conseq_range[index]
            x2 = conseq_range[index + 1]
            y1 = aggregated_mfx[index]
            y2 = aggregated_mfx[index + 1]

            subarea = sum_area / 2.0 - subarea

            diff = x2 - x1
            if y1 == y2:  # rectangle
                result = np.round(subarea / y1 + x1, 2)
            elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                root = np.sqrt(2.0 * subarea * diff / y2)
                result = np.round((x1 + root), 2)
            elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                root = np.sqrt(diff * diff - (2.0 * subarea * diff / y1))
                result = np.round((x2 - root), 2)
            else:                          # trapezium
                m = (y2 - y1) / diff
                root = np.sqrt(y1 * y1 + 2.0 * m * subarea)
                result = np.round((x1 - (y1 - root) / m), 2)

    print('Bisector defuzzified value for {}:{}:{}'.format(conseq_name, result, conseq_range))
    return result, conseq_range, aggregated_mfx


def plot_defuzz(vmfx_list, fuzzy_dict, c_res, c_x, c_mfx, b_res, b_x, b_mfx):
    '''
    Plots the defuzzification results of the centroid and bisector methods.
    
        Args:
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions 
            fuzzy_dict(dict): the original parsed fuzzy variable dictionary
            c_res(np.float32): estimated defuzzified value using the centroid method
            c_x(np.array): the range of possible values for the consequent variable (returned from the centroid method)
            c_mfx(np.array): the aggregated activation functions for each rule (returned from the centroid method)
            b_res(np.float32): estimated defuzzified value using the bisector method
            b_x(np.array): the range of possible values for the consequent variable (returned from the bisector method)
            b_mfx(np.array): the aggregated activation functions for each rule (returned from the bisector method)

            
    '''

    print('c_x:',c_x)
    print('b_x:',b_x)
    
    print('c_res:',c_res)
    print('b_res:',b_res)


    tip0 = np.zeros_like(c_x)
    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_name = vmfx['name']

    print('conseq_name :',vmfx)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['r', 'b', 'g', 'c', 'v']
    c_idx = 0

    for k, v in fuzzy_dict.items():
        if str(k) == conseq_name:
            for k_j, v_j in v.items():
                c_idx += 1
                if c_idx == len(colors) - 1:
                    c_idx = 0
                # c_x=c_x
                print('c',c_idx)
                if c_idx==1:
                    y_moved = np.concatenate((v_j[-5:], v_j[:-5]))
                    cx_int = np.concatenate((-np.flip(c_x[1:-5]), c_x[:-5]))
                    cx_int_0 = cx_int.copy()
                elif c_idx==2:
                    y_moved = np.concatenate((v_j[-5:], v_j[:-5]))
                    print(c_x[-5:])
                    print(1+c_x[1:-4])
                    cx_int = np.concatenate((c_x[-5:] , 1+c_x[1:-4]))
                ax.plot(cx_int, y_moved, c_idx, linewidth=2, linestyle='--')
                ax.set_ylim(-0.01, 1.1)
                ax.set_xlim(-0.5, 1.5)
                print(cx_int, y_moved)
                print(c_x)
                

    print('c_fmx:',c_mfx)
    print('b_fmx:',b_mfx)
    c_mfx_moved = np.concatenate((c_mfx[-5:], c_mfx[:-5]))
    print('c_mfx_moved:',c_mfx_moved)
    # c_mfx_moved = np.concatenate((c_mfx[-5:], c_mfx[:-5]))
    # print('elemento no zero', np.nonzero(c_mfx)[0])
    non_zero = np.nonzero(c_mfx)[0] # the aggregated activation functions for each rule
    Agg_value=c_mfx[non_zero][0]
    if non_zero != 0:
        print('Agg_value:', c_mfx[non_zero])
        ax.fill_between(cx_int, tip0, c_mfx_moved, facecolor='Cyan', alpha=0.3, label=f" Area Membership (Agg = {Agg_value:.2f})")
    else:
        print('Agg_value:', c_mfx[non_zero])
        ax.fill_between(cx_int_0, tip0, c_mfx_moved, facecolor='Cyan', alpha=0.3, label=f" Area Membership (Agg = {Agg_value:.2f})") 
    c_activation = np.interp(c_res, c_x, c_mfx, left=0, right=0)
    b_activation = np.interp(b_res, b_x, b_mfx, left=0, right=0)
    ax.plot([c_res, c_res], [0, c_activation], 'k', linewidth=1.5, alpha=0.9, color='darkgreen', label=f" Centroid DV (deff = {c_res:.2f})")
    ax.plot([b_res, b_res], [0, b_activation], 'k', linewidth=1.5, alpha=0.9, color='darkred', label=f" Bisector DV (deff = {b_res:.2f})")
    ax.set_title('Defuzzification results for Centroid and Bisector methods')
    plt.legend(loc='upper center')
    plt.ylabel('Fuzzy membership value')
    plt.xlabel('Variables')
    plt.savefig("rulesPE2.svg", format="svg")
    plt.show()


##### FUZZY INFERENCE #####







### MEMBERSHIP #####

def create_membership_functions(file=None, from_file=True, default_dict=None):
    '''
    Creates trapezoidal membership functions using the parsed fuzzy variables.
    Generates discrete values of membership based on the estimated range of the variables.

        Args:
            file(str): the input filename
            from_file: feed the file as input or feed an already parsed dictionary (def. True)
            default_dict(dict): the default dictionary to use if from_file is False

        Returns:
            fuzzy_dict(dict): the processed fuzzy variable dictionary with assigned memberships
            x_ranges(dict): the generated range of values for each membership
            var_names(list): list of variable names for lookup
            fuzzy_variables(dict): the original parsed fuzzy variable dictionary
            
    '''


    if from_file:
        if file is None:
            raise ValueError("El archivo no puede ser None si from_file es True.")
        fuzzy_variables = read_variables(file)
    else:
        if default_dict is None:
            raise ValueError("El diccionario por defecto no puede ser None si from_file es False.")
        fuzzy_variables = default_dict

    fuzzy_dict = {}
    x_ranges = {}
    var_names = []
    for k, v in fuzzy_variables.items():
        var_name = k
        max_range = 0
        fuzzy_val_dict = {}
        for k_j, v_j in v.items():
            max_range_j = v_j[1] + v_j[3]
            if max_range_j > max_range:
                max_range = max_range_j

        x_range = np.arange(0, max_range + 0.1, 0.1)

        for k_j, v_j in v.items():
            cat_name = k_j
            a, b, c, d = np.r_[np.float32([v_j[0] - v_j[2], v_j[0], v_j[1], v_j[3] + v_j[1]])]
            y = np.ones(len(x_range))

            ### triangle membership 1
            idx = np.nonzero(x_range <= b)[0]

            a1, b1, c1 = np.r_[np.r_[a, b, b]]
            y1 = np.zeros(len(x_range[idx]))

            # Left side
            if a1 != b1:
                idx1 = np.nonzero(np.logical_and(a1 < x_range[idx], x_range[idx] < b1))[0]
                y1[idx1] = (x_range[idx][idx1] - a1) / float(b1 - a1)

            # Right side
            if b1 != c1:
                idx1 = np.nonzero(np.logical_and(b1 < x_range[idx], x_range[idx] < c1))[0]
                y1[idx1] = (c1 - x_range[idx][idx1]) / float(c1 - b1)

            idx1 = np.nonzero(x_range[idx] == b1)
            y1[idx1] = 1
            y[idx] = y1

            ### Triangle membership 2
            idx = np.nonzero(x_range >= c)[0]

            a2, b2, c2 = np.r_[np.r_[c, c, d]]
            y2 = np.zeros(len(x_range[idx]))

            # Left side
            if a2 != b2:
                idx2 = np.nonzero(np.logical_and(a2 < x_range[idx], x_range[idx] < b2))[0]
                y2[idx2] = (x_range[idx][idx2] - a2) / float(b2 - a2)

            # Right side
            if b2 != c2:
                idx2 = np.nonzero(np.logical_and(b2 < x_range[idx], x_range[idx] < c2))[0]
                y2[idx2] = (c2 - x_range[idx][idx2]) / float(c2 - b2)

            idx2 = np.nonzero(x_range[idx] == b2)
            y2[idx2] = 1
            y[idx] = y2

            idx = np.nonzero(x_range < a)[0]
            y[idx] = np.zeros(len(idx))

            idx = np.nonzero(x_range > d)[0]
            y[idx] = np.zeros(len(idx))

            fuzzy_val_dict[str(cat_name)] = y

        x_ranges[str(var_name)] = x_range
        fuzzy_dict[str(var_name)] = fuzzy_val_dict
        var_names.append(var_name)

    return fuzzy_dict, x_ranges, var_names, fuzzy_variables


def plot_fuzzy_sets(fuzzy_dict, x_ranges):
    '''
    Creates one plot for each fuzzy variable and displays the resulting sets.
    
        Args:
            fuzzy_dict(dict): the processed fuzzy variable dictionary with assigned memberships
            x_ranges(dict): the generated range of values for each membership
            
    '''


    for k, v in fuzzy_dict.items():
        var_name = k
        plt.figure(figsize=(8, 6))
        plt.title(str(var_name))
        for k_j, v_j in v.items():
            sns.lineplot(x=x_ranges[var_name], y=v_j, label=str(k_j), linewidth=3)
            plt.ylabel('Fuzzy membership value')
            plt.xlabel('Variables')
            plt.ylim(-0.01, 1.1)
            plt.legend()
        plt.show()


def read_measurements(file, fuzzy_vars):
    '''
    Parses the input fuzzy measurements from the knowledge base into a dictionary. 
    
        Args:
            file(str): the input filename
            fuzzy_vars(dict): the fuzzy variable dictionary
            
        Returns:
            fuzzy_measurement_dict(dict): the output measurements dictionary
            
    '''


    with open(file) as fp:
        line = fp.readline()
        fuzzy_measurement_dict = {}
        while line:
            if '=' in line:
                variable, result = str.split(line, '=')
                if variable.strip() in fuzzy_vars:
                    fuzzy_measurement_dict[str(variable).strip()] = np.float32(result.strip())
            line = fp.readline()
            print(line)
        return fuzzy_measurement_dict

'''
fuzzy_variables = read_variables('dv.fuzzy')
print(fuzzy_variables)
fuzzy_rules = read_rulebase('dv.fuzzy', fuzzy_variables)
print(fuzzy_rules)
fuzzy_measurements = read_measurements('dv.fuzzy', fuzzy_variables)
print(fuzzy_measurements)
'''


def map_variable_types(measurement_file, fuzzy_variables, var_names, x_ranges, fuzzy_dict, default_dict=None):
    '''
    Creates a lookup mapping of the current fuzzy variables, which includes their range and type (anticedent or consequent).
    Optionally, it can use a default dictionary for the fuzzy variables if provided.
    Additionally, it can use a measurements dictionary to determine the type of variable.
    
        Args:
            measurement_file(str): the input knowledge base file name
            fuzzy_variables(dict): the processed fuzzy dictionary with memberships
            var_names(list): lookup list with variable names
            x_ranges(dict): membership ranges for each fuzzy variable
            fuzzy_dict(dict): the original parsed fuzzy variable dictionary
            default_dict(dict): the default dictionary to use if provided (default is None)
            measurements(dict): the measurements dictionary to use if provided (default is None)
            
        Returns:
            var_type_list(list): list of dictionaries containing the variable name, type and range
            fuzzy_measurements(dict): parsed dictionary of measurements
            
    '''


    var_type_list = []
    if default_dict is None:
        fuzzy_measurements = read_measurements(measurement_file, fuzzy_variables if default_dict is None else default_dict)
    else:
        fuzzy_measurements = default_dict

    for var_name in var_names:
        var_type_dict = {}
        if var_name in fuzzy_measurements.keys():
            var_type_dict['name'] = var_name
            var_type_dict['type'] = 'Antecedent'
            var_type_dict['range'] = x_ranges[var_name]

            var_type_list.append(var_type_dict)
        else:
            var_type_dict['name'] = var_name
            var_type_dict['type'] = 'Consequent'
            var_type_dict['range'] = x_ranges[var_name]

            var_type_list.append(var_type_dict)

    for k, v in fuzzy_dict.items():
        if k in var_names:
            for k_j, v_j in v.items():
                for vmfx in var_type_list:
                    if vmfx['name'] == k:
                        vmfx['vmfx'] = v_j

    return var_type_list, fuzzy_measurements


def read_rulebase(file, fuzzy_vars):
    '''
    Parses the input fuzzy rulebase from the knowledge base into a list of dictionaries. 
    
        Args:
            file(str): the input filename
            fuzzy_vars(dict): the fuzzy variable dictionary
            
        Returns:
            fuzzy_rules(list): the output list of rule objects
            
    '''


    with open(file) as fp:
        line = fp.readline()
        fuzzy_rules = []
        while line != "":
            if ':' in line:
                rule_text = str.split(line, ':')[1]
                if rule_text != "" and 'then' in rule_text:
                    precedent, result = str.split(rule_text, 'then')
                    
                    fuzzy_rules_dict = {}
                    res_dict = {}
                    req_dict = {}
                    connector = 'SIMPLE'
                    if ' and ' in precedent:
                        connector = 'and'
                        precedents = str.split(precedent, ' and ')
                    elif ' or ' in precedent:
                        connector = 'or'
                        precedents = str.split(precedent, ' or ')
                    else:
                        precedents = precedent

                    for i in range(len(precedents)):
                        if i == 0:
                            if not isinstance(precedents, list):
                                precedents = str.split(precedents, 'If')[1]
                                match = re.search(r'(.*) is (.*)', precedents)
                                if match:
                                    if (match.groups()[0] and match.groups()[1]) is not None and match.groups()[
                                        0].strip() in fuzzy_vars:
                                        req = match.groups()[0].strip()
                                        outcome = match.groups()[1].strip()
                                        req_dict[str(req)] = outcome
                                break

                        match = re.search(r'(.*) is (.*)', precedents[i])
                        if match:
                            if (match.groups()[0] and match.groups()[1]) is not None:
                                if 'If' in match.groups()[0].strip():
                                    req = str.split(match.groups()[0].strip(), ' ')[1]
                                else:
                                    req = match.groups()[0].strip()
                                outcome = match.groups()[1].strip()
                                req_dict[str(req)] = outcome

                    result_lhs, result_rhs = str.split(result, 'is')
                    if (result_lhs and result_rhs) is not None and result_lhs.strip() in fuzzy_vars:
                        res_dict[str(result_lhs).strip()] = result_rhs.strip()

                    fuzzy_rules_dict['precedents'] = req_dict
                    fuzzy_rules_dict['connector'] = connector
                    fuzzy_rules_dict['result'] = res_dict
                    fuzzy_rules.append(fuzzy_rules_dict)

            line = fp.readline()

        return fuzzy_rules


def infer_rules(file, fuzzy_vars, fuzzy_dict, fuzzy_measurements, x_ranges, default_rules=None):
    '''
    Creates activations for each fuzzy rule, based on the Mamdani inference principles.
    The areas of activation are then aggregated using max-min composition.
    Can handle simple rules with 1 (SIMPLE) or 2 (AND, OR) conditions.
    Optionally, it can use a default dictionary for the fuzzy variables if provided.
    Additionally, it can use a measurements dictionary to determine the type of variable.
    
        Args:
            file(str): the input knowledge base file name
            fuzzy_vars(dict): the processed fuzzy dictionary with memberships
            fuzzy_dict(dict): the original parsed fuzzy variable dictionary
            fuzzy_measurements(dict): the original parsed measurements dictionary
            x_ranges(dict): membership ranges for each fuzzy variable
            use_default_rules(bool): flag to use default rules instead of reading from file
            
        Returns:
            activation_dict(dict): resulting membership values throughout the range
            
    '''


    anticedent_keys = list(fuzzy_measurements.keys())
    # print(anticedent_keys)
    for k, v in fuzzy_dict.items():
        if str(k) in anticedent_keys:
            for k_j, v_j in v.items():
                cur_interp = np.interp(fuzzy_measurements[k], x_ranges[k], v[k_j], left=0, right=0)
                v[k_j] = cur_interp

    if default_rules:
        fuzzy_rules = default_rules
    else:
        fuzzy_rules = read_rulebase(file, fuzzy_vars)
    activation_dict = {}
    idx = 1
    for rule in fuzzy_rules:

        cur_condition = rule['precedents']
        cur_result = rule['result']
        print('precedente',cur_condition)
        print('resultado',cur_result)
        print('conector',rule['connector'])
        result_membership = fuzzy_dict[list(cur_result.keys())[0]][list(cur_result.values())[0]]
        # print(result_membership)
        # print(rule)
        if rule['connector'] == 'SIMPLE':
            precedent_membership = fuzzy_dict[list(cur_condition.keys())[0]][list(cur_condition.values())[0]]
            # print(fuzzy_dict)
            activation = np.fmin(precedent_membership, result_membership)
            activation_dict['R' + str(idx)] = activation
            idx += 1

        else:
            # can handle rules with 2 conditions
            if len(cur_condition) == 2:
                precedent_membership_i = fuzzy_dict[list(cur_condition.keys())[0]][list(cur_condition.values())[0]]
                precedent_membership_j = fuzzy_dict[list(cur_condition.keys())[1]][list(cur_condition.values())[1]]
                if rule['connector'] == 'and':
                    rule_activation = np.fmin(precedent_membership_i, precedent_membership_j)
                elif rule['connector'] == 'or':
                    rule_activation = np.fmax(precedent_membership_i, precedent_membership_j)

                activation = np.fmin(rule_activation, result_membership)
                activation_dict['R' + str(idx)] = activation
                idx += 1
            elif len(cur_condition) == 3:
                precedent_memberships = [fuzzy_dict[k][v] for k, v in cur_condition.items()]
                if rule['connector'] == 'and':
                    rule_activation = np.fmin.reduce(precedent_memberships)  # Reducción de la función fmin para 'and'
                elif rule['connector'] == 'or':
                    rule_activation = np.fmax.reduce(precedent_memberships)  # Reducción de la función fmax para 'or'
                
                activation = np.fmin(rule_activation, result_membership)
                activation_dict['R' + str(idx)] = activation
                idx += 1
            else:
                print('Invalid condition count')
                return

    return activation_dict


#### ESTO ES LO ITERABLE #######

archivo='reglas_PE2.txt'

fuzzy_variables = {
    'MAP': {
        'VL': [61.66667, 61.66667, 0.0, 26.8444060576848],
        'L': [88.5110760576848, 88.5110760576848, 26.8444060576848, 12.582808131227992],
        'M': [101.0938841889128, 101.0938841889128, 12.582808131227992, 32.23941597810989],
        'H': [133.33330016702268, 133.33330016702268, 32.23941597810989, 0.0]
    },
    'BMI': {
        'VL': [17.310000047419003, 17.310000047419003, 0.0, 3.915658682627999],
        'L': [21.225658730047, 21.225658730047, 3.915658682627999, 9.048609730488998],
        'M': [30.274268460536, 30.274268460536, 9.048609730488998, 13.121631539463998],
        'H': [43.3959, 43.3959, 13.121631539463998, 0.0]
    },
    'AGE': {
        'VL': [18.0, 18.0, 0.0, 8.396778448],
        'L': [26.396778448, 26.396778448, 8.396778448, 5.0183836799999995],
        'M': [31.415162128, 31.415162128, 5.0183836799999995, 13.784837872000004],
        'H': [45.2, 45.2, 13.784837872000004, 0.0]
    },
    'p': {
        'Sana': [0, 0, 0, 0],
        'PE': [1, 1, 0, 0]
    }
}

fuzzy_dict, x_ranges, var_names, fuzzy_variables = create_membership_functions(file=archivo, from_file=False, default_dict=fuzzy_variables)
#plot_fuzzy_sets(fuzzy_dict, x_ranges)

# measurements = {
#     'MAP': 82.33,
#     'BMI': 23.22,
#     'AGE': 36
# }

# Inicializamos el diccionario
measurements = {}

# Abrimos y leemos el archivo
with open(archivo, 'r') as f:
    for linea in f:
        linea = linea.strip()
        if linea.startswith('MAP ='):
            measurements['MAP'] = float(linea.split('=')[1].strip())
        elif linea.startswith('BMI ='):
            measurements['BMI'] = float(linea.split('=')[1].strip())
        elif linea.startswith('AGE ='):
            measurements['AGE'] = float(linea.split('=')[1].strip())


vmfx_list, fuzzy_measurements = map_variable_types(archivo, fuzzy_variables, var_names, x_ranges, fuzzy_dict, default_dict=measurements)

fuzzy_rules = [
    {
        'precedents': {'MAP': 'VL', 'BMI': 'H'},
        'connector': 'and',
        'result': {'p': 'Sana'}
    },
    {
        'precedents': {'MAP': 'L', 'BMI': 'L'},
        'connector': 'and',
        'result': {'p': 'Sana'}
    },
    {
        'precedents': {'MAP': 'L'},
        'connector': 'SIMPLE',
        'result': {'p': 'Sana'}
    },
    {
        'precedents': {'MAP': 'H'},
        'connector': 'SIMPLE',
        'result': {'p': 'PE'}
    },
    {
        'precedents': {'MAP': 'H', 'BMI': 'VL', 'AGE': 'L'},
        'connector': 'and',
        'result': {'p': 'Sana'}
    },
    {
        'precedents': {'MAP': 'H', 'BMI': 'M', 'AGE': 'L'},
        'connector': 'and',
        'result': {'p': 'PE'}
    }
]


activation_dict = infer_rules(archivo, fuzzy_variables, fuzzy_dict, fuzzy_measurements, x_ranges, default_rules=fuzzy_rules)


c_res, c_x, c_mfx = defuzzify_centroid(activation_dict, vmfx_list)
b_res, b_x, b_mfx = defuzzify_bisector(activation_dict, vmfx_list)
plot_defuzz(vmfx_list, fuzzy_dict, c_res, c_x, c_mfx, b_res, b_x, b_mfx)

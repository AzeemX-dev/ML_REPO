import math
import numpy
import pprint
import sys
data = numpy.loadtxt(sys.argv[1], dtype=int, delimiter=",")

data_shape = data.shape
encodings = numpy.loadtxt(sys.argv[2], dtype=str)
data_set = numpy.empty([data_shape[0] + 1, data_shape[1]],
                       dtype=numpy.dtype("U100"))


def helper_with_encodings(data_set, data, encodings):
    encodings_split = encodings[0].split(",")
    data_set[0] = encodings_split
    data_list = []
    for row in range(len(data)):
        temp_list = []
        heading_index = 1
        for items in data[row]:
            temp_list.append(encodings[heading_index].split(",")[items])
            heading_index += 1
        data_list.append(temp_list)
    index = 1
    for items in data_list:
        data_set[index] = items
        index += 1


def find_entropy_given_attribute(dataset, attrbute_index):
    entropy_to_return = 0
    ### YOUR CODE HERE ###
    VALS, cTnS = numpy.unique(data_set[1:, -1], return_counts=True)
    entropy_to_return = (attrbute_index/sum(cTnS)) * \
        math.log(attrbute_index/sum(cTnS), 2)

    return entropy_to_return


def find_entropy_of_system(data_set):

    VALS, cTnS = numpy.unique(data_set[1:, -1], return_counts=True)
    valLen = len(VALS)
    if valLen == 1:
        return 0
    elif cTnS[0] == cTnS[1]:
        return 1
    
    entropy = 0
    for attr_index in cTnS:
        entropy -= find_entropy_given_attribute(data_set, attr_index)
    return entropy


def getInd(data_set, attribute):
    dat_shaped = data_set.shape
    dat_set = data_set
    dat_shp_1 = dat_shaped[1]
    for COLS in range(dat_shp_1):
        zer = 0
        if dat_set[zer][COLS] == attribute:
            return COLS


def sub_table_attrb(data_set, attribute, sub_attribute, encodings):

    shape = data_set.shape
    VALS, cTnS = numpy.unique(
        data_set[1:, getInd(data_set, attribute)], return_counts=True)
    row = 0
    column = shape[1]
    for i in range(len(VALS)):
        if VALS[i] == sub_attribute:
            row = cTnS[i]
            break
    data_set_temp = numpy.empty([row + 1, column], dtype=numpy.dtype("U100"))
    N_hed = []
    splt_enc = encodings[0].split(",")
    for attr in splt_enc:
        N_hed.append(attr)
    data_set_temp[0] = N_hed
    i = 1
    column = getInd(data_set, attribute)
    for row in range(len(data_set)):
        if data_set[row][column] == sub_attribute:
            data_set_temp[i] = data_set[row]
            i += 1

    return data_set_temp


def find_information_gain_of_attribute(data_set, attrbute_index):
    overall_entropy = find_entropy_of_system(data_set)
    total_length = len(data_set)
    gain = {}
    attributes = data_set[0][:-1]
    for attribute in attributes:
        sub_attributes = numpy.unique(
            data_set[1:, getInd(data_set, attribute)])
        gain[attribute] = overall_entropy
        for sub_attribute in sub_attributes:
            sub_entropy = find_entropy_of_system(sub_table_attrb(
                data_set, attribute, sub_attribute, attrbute_index))
            sub_length = len(sub_table_attrb(
                data_set, attribute, sub_attribute, attrbute_index))

            multiPl = (sub_length/total_length) * sub_entropy
            gain[attribute] =gain[attribute] - multiPl

    return gain


def max_gain_node(data_set, tablee):
    information_gain = find_information_gain_of_attribute(data_set, tablee)
    # max info gain out of all keys
    return max(information_gain, key=information_gain.get)


def decision_tree_maker(data_set, tablee=None):
    decision_tree = {}  # Our Decision Tree
    gain_at_max_node = max_gain_node(data_set, tablee)  # Calc max gain
    gain_at_max_node_attributes = numpy.unique(data_set[1:, getInd(
        data_set, gain_at_max_node)])  # Get the unique elems (num of elems)
    # Make a sub-dict at the specific index of max_gain
    decision_tree[gain_at_max_node] = {}
    for attribute in gain_at_max_node_attributes:
        sub_table = sub_table_attrb(
            data_set, gain_at_max_node, attribute, tablee)
        VALS = numpy.unique(sub_table[1:, -1])
        if len(VALS) == 1:
            decision_tree[gain_at_max_node][attribute] = VALS[0]
        else:
            decision_tree[gain_at_max_node][attribute] = decision_tree_maker(
                sub_table, tablee)
    return decision_tree


helper_with_encodings(data_set, data, encodings)
decision_tree = decision_tree_maker(data_set, encodings)
pprint.pprint(decision_tree)

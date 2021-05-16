import pandas as pd
import operator
import numpy as np


class Node:

    def __init__(self, attribute: str, previous_attribute_value: str, decision: str, children: list):
        self.attribute = attribute
        self.decision = decision
        self.children = []
        self.previous_attribute_value = previous_attribute_value


root = Node("Hello", None, None, None)


def entropy(probs):
    import math
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_of_list(a_list):
    print(a_list)
    from collections import Counter
    # print(x for x in a_list)
    cnt = Counter(a_list)  # Counter calculates the propotion of class
    # print("entropy_of_list function")
    print(cnt)

    prob = []
    for i, j in cnt.items():
        prob.append(j / len(a_list))

    print(prob)
    return entropy(prob)


# total_entropy = 0
# print(total_entropy)
isnode = 0


def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    # print("DF")
    # print(df)

    total_entropy = entropy_of_list(df[target_attribute_name[0]])
    d = []
    df_split = df.groupby(split_attribute_name)
    # print(df_split)
    for i, j in df_split:
        # print(i)
        # print(j)
        # print(len(j))
        d.append(j)
    """
    for i in range(0, len(d)):
        print(d[i])
    """
    entropy_d = {}
    for i, j in df_split:
        entropy_d[i] = entropy_of_list(j[target_attribute_name[0]])

    remaining_entropy = 0
    for i, j in df_split:
        remaining_entropy = remaining_entropy + len(j) / len(df) * entropy_d[i]

    return total_entropy - remaining_entropy


"""



"""


def C4_5(isnode, d, attributes, target_attribute, current_node):
    print(d)

    if isnode != 0:
        attribute_value = d[current_node.attribute]
        attribute_value = list(attribute_value)
        attribute_value = attribute_value[0]
        d.drop(current_node.attribute, inplace=True, axis=1)
        attributes.remove(current_node.attribute)

    # current_node = None
    dic = {}
    for i in range(0, len(attributes)):
        t = information_gain(d, attributes[i], target_attribute)
        en = entropy_of_list(d[attributes[i]])
        dic[attributes[i]] = t / en
    i = 0
    for i, j in dic.items():
        print(i, " ", j)
    dic2 = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    print(dic)
    print(dic2)

    if isnode == 0:
        global root
        root = Node(dic2[0][0], None, None, None)
        print(root.attribute, " ", root.previous_attribute_value, " ", root.decision, " ", root.children)
        isnode = isnode + 1
        current_node = root
        # st = ""+current_node.attribute
        # print(st)

        df_split = d.groupby(current_node.attribute)
        for i, j in df_split:
            # print("Hello world.")
            print(i, " \n", j)
            attribute_value = j[current_node.attribute]
            attribute_value = list(attribute_value)
            attribute_value = attribute_value[0]
            print(attribute_value)
            print(current_node.attribute)
            # j.drop(current_node.attribute , inplace=True, axis=1)
            # j.drop( columns= ""+current_node.attribute)
            print(j)
            attributes = j.columns
            attributes = list(attributes)
            print(attributes)
            print(target_attribute[0])
            attributes.remove(target_attribute[0])
            print(attributes)
            decision = j[target_attribute[0]]
            decision = np.array(decision)
            if len(np.unique(decision)) == 1:
                # current_node.decision = decision
                current_node.children.append(Node(None, attribute_value, decision[0], None))
                continue
            C4_5(isnode, j, attributes, target_attribute, current_node)





    else:
        array = []
        df_split = d.groupby(dic2[0][0])
        n = Node(dic2[0][0], attribute_value, None, None)
        current_node.children.append(n)
        current_node = n
        print(current_node.attribute)
        df_split = d.groupby(current_node.attribute)
        for i, j in df_split:
            # print("Hello world.")
            print(i, " \n", j)
            print(current_node.attribute)
            # j.drop(current_node.attribute , inplace=True, axis=1)
            # j.drop( columns= ""+current_node.attribute)
            print(j)
            attributes = j.columns
            attributes = list(attributes)
            print(attributes)
            print(target_attribute[0])
            attributes.remove(target_attribute[0])
            print(attributes)
            decision = j[target_attribute[0]]
            decision = np.array(decision)
            if len(np.unique(decision)) == 1:
                current_node.children.append(Node(None, i, decision[0], None))
                continue
            C4_5(isnode, j, attributes, target_attribute, current_node)

            return  root


def prediction(input, root):


    # temp = root.children[0]
    # print(temp.previous_attribute_value)
    print(input)
    current = root
    while (1):
        value = input[current.attribute][0]
        print(value)
        for i in range(0, len(current.children)):

            temp = current.children[i].previous_attribute_value
            print(temp)
            if value == current.children[i].previous_attribute_value:
                current = current.children[i]
                break
        if current.decision != None:
            print(current.decision)
            break


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("PlayTennis.csv")

    X = d[["Outlook", "Temperature", "Humidity", "Wind"]]
    Y = d["PlayTennis"]
    # print(X)
    # print("\n")
    # print(Y)


    target_attribute = ['PlayTennis']
    attribute = ["Outlook", "Temperature", "Humidity", "Wind"]

    i = C4_5(0, d, attribute, target_attribute, None)

    tree = root

    

    input = [['Rain', 'Cool', 'Normal', 'Strong']]

    input = pd.DataFrame(input,
                         columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
    prediction(input, tree)




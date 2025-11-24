from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Fork note: rules now carry ponderaciones and prediction helpers expose matching-degree/debug info and extra visualizations.

class ClassificationRule():

    def __init__(self, antecedent, fuzzyset, consequent, ponderaciones, weight=1.0, label=None):
        """

        @param antecedent: antecedents in the rule. For each one [a, b, c] that represents the triangular fuzzyset
        @param fuzzyset:
        @param consequent:
        @param weight:
        @param label:
        """
        self.antecedent = antecedent
        self.fuzzyset = fuzzyset
        self.consequent = consequent
        self.weight = weight
        self.ponderaciones= ponderaciones
        # self.md_app = []

        if label is not None:
            self.label = label
        else:
            self.label = id(self)

@jit(nopython=True)
def membership_value(mf, value, type='triangular'):
    if type=='triangular' and len(mf) == 3:
        if mf[0] == mf[1]: # left triangular
            if value < mf[0]:
                return 1.0
            elif value > mf[2]:
                return 0.0
            else:
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
        elif mf[1] == mf[2]: # right triangular
            if value < mf[0]:
                return 0.0
            elif value > mf[2]:
                return 1.0
            else:
                return (value - mf[0]) / (mf[1] - mf[0])
        else: # triangular
            if value < mf[0] or value > mf[2]:
                return 0.0
            elif value <= mf[1]:
                return (value - mf[0]) / (mf[1] - mf[0])
            elif value <= mf[2]:
                return 1.0 - ((value - mf[1]) / (mf[2] - mf[1]))
    # Not implemented
    return 0.0

@jit(nopython=True)
def predict_fast(x, ant_matrix, cons_vect, weights, part_matrix):
    """

    :param x: input matrix NxM where N is the number of samples and M is the number of features
    :param ant_matrix: antecedents of every rule in the RB
    :param cons_vect: consequents of every rule in the RB
    :param weights:
    :param part_matrix: partitions of fuzzysets
    :return:
    """
    sample_size = x.shape[0]
    y = np.zeros(sample_size)
    # For each sample
    # z=[]
    md=[]
    bm_app=[]
    bm_app_real=[]
    for i in range(sample_size):
        best_match_index = 0
        j_ant=0
        best_match = 0.0
        # For each rule
        z=np.zeros((ant_matrix.shape[0],ant_matrix.shape[1]+1))
        bm=np.zeros_like(weights)
        bm_real=np.zeros_like(weights)
        for j in range(ant_matrix.shape[0]):
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if not np.isnan(ant_matrix[j][k]):
                    base = int(ant_matrix[j][k])
                    ant = part_matrix[k][base:base+3]
                    m_degree = membership_value(ant, x[i][k])
                    matching_degree *= m_degree
                    z[j][k]= m_degree
                    z[j][-1]= matching_degree
            if (weights[j] * matching_degree) > best_match:
                bm_real[j_ant]=0
                j_ant=j
                best_match_index = j
                best_match = weights[j] * matching_degree
                bm[j]=best_match
                # bm_real=np.zeros_like(weights)
                bm_real[j]=best_match
                
                
        bm_app.append(bm)
        bm_app_real.append(bm_real)
        md.append(z)
        # z.append(weights)
        y[i] = cons_vect[best_match_index]
        # md_app.append(md)
        # if i==sample_size:
        #     print(weights)
        #     print('Pesos:',len(weights))
    return y, md, bm_app, bm_app_real

@jit(nopython=True)
def predict_fast_bm(x, ant_matrix, cons_vect, weights, part_matrix):
    """

    """
    sample_size = x.shape[0]
    y = np.zeros(sample_size)
    y_bm = np.zeros(sample_size)
    # y_md_app = []
    y_md = np.zeros(sample_size)
    bm_ind = np.zeros(sample_size)
    print(y_md)
    # For each sample
    # z=[]
    # print(len(weights))
    for i in range(sample_size):
        best_match_index = 0
        best_match = 0.0
        # For each rule
        for j in range(ant_matrix.shape[0]):
            # md = []
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if not np.isnan(ant_matrix[j][k]):
                    base = int(ant_matrix[j][k])
                    ant = part_matrix[k][base:base+3]
                    m_degree = membership_value(ant, x[i][k]) #valor de x en el triangulo
                    matching_degree *= m_degree
            if (weights[j] * matching_degree) > best_match:
                best_match_index = j
                best_match = weights[j] * matching_degree
                # z.append(weights) 
        # z.append(weights)
        y[i] = cons_vect[best_match_index]
        y_bm[i]= best_match           
        bm_ind[i]= best_match_index
        y_md[i]= matching_degree
        
        # if i==sample_size:
        #     print(weights)
        #     print('Pesos:',len(weights))
    return y, y_bm, y_md, bm_ind

@jit(nopython=True)
def compute_weights_fast(train_x, train_y, ant_matrix, cons_vect, part_matrix):
    """

    :param train_x: Training input
    :param train_y: Training output
    :param ant_matrix: antecedents
    :param cons_vect: consequents
    :param part_matrix: partitions of fuzzysets
    :return: for each rule in the RB, compute the weight from the provided
            training set
    """
    weights = np.ones(ant_matrix.shape[0])
    # z=[]
    # md=[]
    # md_app=[]   
    # For each rule
    for i in range(ant_matrix.shape[0]):
        matching = 0.0
        total = 0.0
        for j in range(train_y.shape[0]):
            matching_degree = 1.0
            for k in range(ant_matrix.shape[1]):
                if not np.isnan(ant_matrix[i][k]):
                    index = int(ant_matrix[i][k])
                    ant = part_matrix[k][index:index+3]
                    m_degree = membership_value(ant, train_x[j][k])
                    # z.append(m_degree)
                    matching_degree *= m_degree
            if train_y[j] == cons_vect[i]:
                matching += matching_degree
            total += matching_degree
        weights[i] = total if total == 0 else matching / total
    return weights

class FuzzyRuleBasedClassifier():
    """
    Fuzzy Rule-Based Classifier class
    """

    def __init__(self, rules, partitions):

        """

        :param rules: a list of ClassificationRule objects
        :param partitions: fuzzyset partitions for each fuzzy input
        """

        # RB info
        self.rules = rules
        # print('Rules:',len(rules))
        # DB info
        self.partitions = partitions

        # RB and DB information are converted into NumPy matrices
        self.ant_matrix = np.full((len(rules), len(partitions)), np.nan)
        self.ant_dic = []
        self.fuzzydatos=[]
        self.cons_vect = np.empty((len(rules)))
        self.weights = np.ones((len(rules)))
        self.granularities = np.array([len(partition) for partition in partitions], dtype=np.int32)
        self.part_matrix = np.zeros([len(self.granularities), 2 + max(self.granularities)])
        # self.z=[]
        self.ponderacion=[]
        self.pesos=[]
        
        self.md_app=[]
        self.bm_app=[]
        self.bm_app_real=[]
        self.pred_y=[]
        
        for i in range(len(self.granularities)):
            # print(partitions)
            if partitions[i].size > 0:
                self.part_matrix[i][0] = partitions[i][0]
                for j in range(self.granularities[i] ):
                    self.part_matrix[i][j+1] = partitions[i][j]
                self.part_matrix[i][self.granularities[i]+1] = partitions[i][-1]
            # self.z.append(partitions)
            # if i ==len(self.granularities)-1:
                # print(partitions)
            
        for i, rule in enumerate(self.rules):
            for key in rule.antecedent:
                self.ant_matrix[i, key] = rule.fuzzyset[key] - 1 #las posiciones de las caracteristicas activadas
                # print(self.ant_matrix)
            self.fuzzydatos.append(rule.fuzzyset)
            self.ant_dic.append(rule.antecedent) # es la particion
            # self.z.append(partitions) # es la particion
            self.cons_vect[i] = rule.consequent
            self.ponderacion.append(rule.ponderaciones)
            self.pesos.append(rule.weight)
            # print(len(self.z))
            

    def addrule(self, new_rule):
        self.rules.append(new_rule)
        
    # def del_rule(self):
    #     # print(self.ant_matrix)
    #     if len(self.rules)>1:
    #         for i,r in enumerate(self.rules):
    #             if self.weights[i]<0.1:
    #                 self.rules.pop(i)
    #                 self.fuzzydatos.pop(i)
    #                 self.weights=np.delete(self.weights, i)
    #                 self.ant_matrix=np.delete(self.ant_matrix, i, axis=0)
    #                 self.cons_vect=np.delete(self.cons_vect, i)
    #                 self.ant_dic.pop(i)
    #                 self.pond.pop(i)
    #     return print('OK')
        # print(self.ant_matrix)
                
        
    def fuzzy_set(self):
        return self.fuzzydatos
        
    def matriz_particion(self):
        return self.part_matrix
    
    def particiones(self):
        return self.partitions
        
    def antecedentes_matrix(self):
        return self.ant_matrix
        
    def granularidad(self):
        return self.granularities
    
    def antecedentes_dict(self):
        return self.ant_dic
    
    def ponderaciones_(self):
        return self.ponderacion
    
    def result_class(self):
        return self.cons_vect

    def num_rules(self):
        return len(self.rules)

    def predict(self, x):
        self.pred_y, self.md_app, self.bm_app, self.bm_app_real = predict_fast(x, self.ant_matrix, self.cons_vect, self.weights, self.part_matrix)
        return self.pred_y
    
    def predict_bm(self, x):
        return predict_fast_bm(x, self.ant_matrix, self.cons_vect, self.weights, self.part_matrix)
    

    def compute_weights(self, train_x, train_y):
        self.weights = compute_weights_fast(train_x, train_y, self.ant_matrix, self.cons_vect, self.part_matrix)

    def pesos_reglas(self):
        return self.weights

    def trl(self):
        n_antecedents = [len(rule.antecedent) for rule in self.rules]
        return np.sum(n_antecedents)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return sum(y_pred == y) / len(y)

    def auc(self, x, y):
        y_pred = self.predict(x)
        # print(y_pred)
        # Binarize labels (One-vs-All)
        lb = LabelBinarizer()
        lb.fit(y)
        # Transform labels
        y_bin = lb.transform(y)
        y_pred_bin = lb.transform(y_pred)
        # print(y_pred_bin)
        return roc_auc_score(y_bin, y_pred_bin, average='macro')

    def _get_labels(self, size):
        if size == 3:
            return ['L', 'M', 'H']
        if size == 4:
            return ['VL','L', 'M', 'H']
        if size == 5:
            return ['VL', 'L', 'M', 'H', 'VH']
        if size == 6:
            return ['VL','L', 'ML','M' , 'H','VH']
        if size == 7:
            return ['VL', 'L', 'ML', 'M', 'MH', 'H', 'VH']

    def show_RB(self, inputs, outputs, f=None):
        if f:
            f.write('RULE BASE\n')
            startBold = ''
            endBold = ''
        else:
            print('RULE BASE')
            startBold = '\033[1m'
            endBold = '\033[0m'
        if_keyword = startBold + 'If' + endBold
        then_keyword = startBold + 'then' + endBold
        is_keyword = startBold + 'is' + endBold

        for i, rule in enumerate(self.rules):
            if_part = if_keyword + ' '
            count = 0
            for key in rule.antecedent:
                size = len(self.partitions[key])
                labels = self._get_labels(size)
                if count > 0:
                    if_part += startBold + ' and ' + endBold
                count += 1
                if inputs is None:
                    feature = 'X_' + str(key + 1)
                else:
                    feature = inputs[key]
                if_part += feature
                if_part += ' ' + is_keyword + ' ' + labels[rule.fuzzyset[key]-1] + ' '
            if outputs is None:
                output = 'Class'
            else:
                output = outputs[0]
            then_part = then_keyword + ' ' + output + ' is ' + str(rule.consequent)+ str(rule.ponderaciones)
            if f:
                f.write('Rule '+ str(i+1) + ':\t' + if_part + then_part + '\n')
            else:
                print('Rule ' + str(i+1) + ':\t' + if_part + then_part)
    
    def NEW_show_RB(self, inputs, outputs, partitiones,consecuente, f=None):
        f= open("reglas.txt", 'w')
        if f:
            print('Existe F')
            f.write('RULE BASE\n')
            startBold = ''
            endBold = ''
        else:
            print('RULE BASE')
            startBold = '\033[1m'
            endBold = '\033[0m'
        if_keyword = startBold + 'If' + endBold
        then_keyword = startBold + 'then' + endBold
        is_keyword = startBold + 'is' + endBold

        for i, rule in enumerate(self.rules):
            if_part = if_keyword + ' '
            count = 0
            for key in rule.antecedent:
                size = len(self.partitions[key])
                labels = self._get_labels(size)
                if count > 0:
                    if_part += startBold + ' and ' + endBold
                count += 1
                if inputs is None:
                    feature = 'X_' + str(key + 1)
                else:
                    feature = inputs[key]
                if_part += feature
                if_part += ' ' + is_keyword + ' ' + labels[rule.fuzzyset[key]-1] + ' '
            if outputs is None:
                output = 'Class'
            else:
                output = outputs[0]
            then_part = then_keyword + ' ' + output + ' is ' + str(rule.consequent)
            # then_part = then_keyword + ' ' + output + ' is ' + str(rule.consequent) + str(rule.ponderaciones)
            if f:
                f.write('Rule '+ str(i+1) + ':\t' + if_part + then_part + '\n')
            else:
                print('Rule ' + str(i+1) + ':\t' + if_part + then_part)


        for k,key in enumerate(inputs):
           # if partitiones[k]!=[]:
            feature = key
            # print(feature)
            labels = self._get_labels(len(partitiones[k]))
            if len(partitiones[k])==0:
                anchors=([])
                fuzzyset_size= 0
            else:
                f.write('\n' + str(feature) + '\n'+ '\n')
                anchors = np.concatenate(([partitiones[k][0]], partitiones[k], [partitiones[k][-1]]))
                fuzzyset_size= len(anchors) - 2
            # print('FUZZYSET:',fuzzyset_size)
            for j in range(fuzzyset_size):
                # print('LABELS:',len(labels))
                eje_triangulos=anchors[j:j+3]
                f.write(str(labels[j])+ ' '+ str(eje_triangulos[1])+' '+str(eje_triangulos[1])+' '+ 
                        str(eje_triangulos[1]-eje_triangulos[0])+' '+ str(eje_triangulos[2]-eje_triangulos[1])+ '\n')
        
        
        # X[87] 0.527778	0.375	0.559322	0.5
        
        f.write('\n')
        for k in outputs:
            f.write(k[-1])
            
        f.write('\n' + str(0)+ ' ' + str(0)+ ' ' +str(0)+ ' ' +str(0)+ ' ' +str(0)+ '\n')
        f.write(str(1)+ ' ' + str(1)+ ' ' +str(1)+ ' ' +str(0)+ ' ' +str(0)+ '\n')
        
            # for n,i in enumerate(consecuente):
                # f.write('\n' +'Class'+ str(n+1)+ ' ' + str(int(i))+ ' ' +str(int(i))+ ' ' +str(0)+ ' ' +str(0))
                # f.write('\n' + str(n)+ ' ' + str(int(i))+ ' ' +str(int(i))+ ' ' +str(0)+ ' ' +str(0))
                # f.write('\n' + str(n))

        
        f.write('\n'+'\n' +str(inputs[0])+ ' ' +'='+ ' ' + str(0.527778))
        f.write('\n' +str(inputs[2])+ ' ' +'='+ ' ' + str(0.559322))
        f.write('\n' +str(inputs[3])+ ' ' +'='+ ' ' + str(0.5))

               


    def show_DB(self, inputs):
        """

        :param inputs: names of the input variables
        :return: for each input, plot a graph representing the membership functions of
                fuzzyset partitions
        """

        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', titlesize=14)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=14)  # fontsize of the figure title
        for k, partition in enumerate(self.partitions):
            # print(partition.size)
            if partition.size == 0:
                plt.plot()
                if inputs is not None:
                    xLabel = inputs[k]
                else:
                    xLabel = 'X_' + str(k+1)
                plt.xlabel(xLabel)
                plt.ylim([0.0, 1.1])
                # plt.xlim([0.0, max(partition)])
                plt.xlim([0.0, 1.0])
                
            # if partition.size != 0:
            else:
                plt.figure()
                anchors = np.concatenate(([partition[0]], partition, [partition[-1]]))
                fuzzyset_size= len(anchors) - 2
                triangle = np.array([0.0, 1.0, 0.0])
                for i in range(fuzzyset_size):
                    plt.plot(anchors[i:i+3], triangle, linestyle='solid', linewidth=2, color='k')
                    # print('ANCHORS_ORIG=',anchors[i:i+3])

                # Add a legend
                if inputs is not None:
                    xLabel = inputs[k]
                else:
                    xLabel = 'X_' + str(k+1)
                plt.xlabel(xLabel)
                plt.ylim([0.0, 1.1])
                plt.xlim([0.0, max(anchors)])
                # plt.xlim([0.0, 1.0])
    
                # Show the plot
                plt.show()
                plt.close()
                

    def NEW_show_DB(self, inputs, attribute, partitiones):
        """

        :param inputs: names of the input variables
        :return: for each input, plot a graph representing the membership functions of
                fuzzyset partitions
        """

        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', titlesize=14)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        plt.rc('figure', titlesize=14)  # fontsize of the figure title
        for k, partition in enumerate(self.partitions):
            # print(partition.size)
            if partition.size == 0:
                plt.plot()
                if inputs is not None:
                    xLabel = inputs[k]
                else:
                    xLabel = 'X_' + str(k+1)
                plt.xlabel(xLabel)
                plt.ylim([0.0, 1.1])
                # plt.xlim([0.0, max(partition)])
                plt.xlim([0.0, 1.0])
                
            # if partition.size != 0:
            else:
                labels = self._get_labels(len(partitiones[k]))
                print('LABELS:',labels)
                plt.figure()
                anchors = np.concatenate(([partition[0]], partition, [partition[-1]]))
                fuzzyset_size= len(anchors) - 2
                triangle = np.array([0.0, 1.0, 0.0])
                for i in range(fuzzyset_size):
                    anchors_temp=anchors.copy()
                    anchors_temp[i:i+3]=anchors[i:i+3]*(attribute[k][1]-attribute[k][0])+attribute[k][0]
                    triangulo=anchors_temp[i:i+3]
                    plt.plot(anchors_temp[i:i+3], triangle, linestyle='solid', linewidth=2, color='k')
                    # if i==0:
                        # plt.text(triangulo[0], 0, labels[i], fontsize=15)
                    # else:
                    plt.text(triangulo[1], 0.5, labels[i], fontsize=15)
                    # print('ANCHORS_DESNORM=',anchors[i:i+3])

                # Add a legend
                if inputs is not None:
                    xLabel = inputs[k]
                else:
                    xLabel = 'X_' + str(k+1)
                plt.xlabel(xLabel)
                plt.ylim([0.0, 1.1])
                # plt.xlim([min(anchors_temp), max(anchors_temp)])
                # plt.xlim([0.0, 1.0])
    
                # Show the plot
                plt.show()
                plt.close()

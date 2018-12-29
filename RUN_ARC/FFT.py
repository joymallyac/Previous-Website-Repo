__author__ = 'Joy'

import sys
import collections
import numpy as np
import pandas as pd
from helpers import get_performance, get_score, get_auc

PRE, REC, SPEC, FPR, NPV, ACC, F1 = 7, 6, 5, 4, 3, 2, 1
class FFT(object):

    def __init__(self, max_level=5):
        self.max_depth = max_level - 1
        cnt = 2 ** self.max_depth
        self.tree_cnt = cnt
        self.tree_depths = [0] * cnt
        self.best = -1
        self.target = ""
        self.split_method = "median"
        self.criteria = "Dist2Heaven"
        self.data_name = ''
        self.train, self.test = None, None
        self.structures = None
        self.computed_cache = {}
        self.selected = [{} for _ in range(cnt)]
        self.tree_scores = [None] * cnt
        self.dist2heavens = [None] * cnt
        self.node_descriptions = [None] * cnt
        self.performance_on_train = [collections.defaultdict(dict) for _ in range(cnt)]
        self.performance_on_test = [None] * cnt
        self.predictions = [None] * cnt
        self.loc_aucs = [None] * cnt
        self.print_enabled = False
        self.store_cur_selected = [0,0,0,0]
        self.first_thresholds = {}



    def build_trees1(self):
        self.structures = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        data = self.train
        store_tree = {}
        for i in range(16):
            if i%2 == 0:
                self.growEven(data, i, 0, [0, 0, 0, 0])
            else:
                self.growOdd(self.store_cur_selected['undecided'], i, 3, self.store_cur_selected['metrics'])
            store_tree[i] = self.print_tree(i)
        # for j in range(16):            
        #     print(store_tree[j])


    def build_trees2(self):
        self.structures = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        data = self.train
        for i in range(4,8):
            if i%2 == 0:
                self.growEven(data, i, 0, [0, 0, 0, 0])
            else:
                self.growOdd(self.store_cur_selected['undecided'], i, 3, self.store_cur_selected['metrics'])




    def build_trees3(self):
        self.structures = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        data = self.train
        for i in range(8,12):
            if i%2 == 0:
                self.growEven(data, i, 0, [0, 0, 0, 0])
            else:
                self.growOdd(self.store_cur_selected['undecided'], i, 3, self.store_cur_selected['metrics'])



    def build_trees4(self):
        self.structures = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        data = self.train
        for i in range(12,16):
            if i%2 == 0:
                self.growEven(data, i, 0, [0, 0, 0, 0])
            else:
                self.growOdd(self.store_cur_selected['undecided'], i, 3, self.store_cur_selected['metrics'])



    "Evaluate all tress built on TEST data."

    def eval_trees(self):
        for i in range(self.tree_cnt):            
            self.eval_tree(i)

    "Evaluate the performance of the given tree on the TEST data."

    def eval_tree(self, t_id):        
        if self.performance_on_test[t_id]:
            return
        depth = self.tree_depths[t_id]
        self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        TP, FP, TN, FN = 0, 0, 0, 0
        data = self.test
        for level in range(depth + 1):
            cue, direction, threshold, decision = self.selected[t_id][level]
            undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
            tp, fp, tn, fn = self.update_metrics(level, depth, decision, metrics)
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
            if len(undecided) == 0:
                break
            data = undecided
        pre, rec, spec, fpr, npv, acc, f1 = get_performance([TP, FP, TN, FN])
        self.performance_on_test[t_id] = [TP, FP, TN, FN, pre, rec, spec, fpr, npv, acc, f1]

    "Find the best tree based on the score in TRAIN data."

    def find_best_tree(self,start,end):
        if self.tree_scores and self.tree_scores[0]:
            return
        if self.print_enabled:
            print ("\t----- PERFORMANCES FOR ALL FFTs on Training Data -----")
            print (PERFORMANCE + " \t" + self.criteria)
        best = [-1, float('inf')]
        for i in range(start,end):
            all_metrics = self.performance_on_train[i][self.tree_depths[i]]            
            if self.criteria == "LOC_AUC":
                score = self.loc_aucs[i]
            else:
                score = get_score(self.criteria, all_metrics[:4])
            self.tree_scores[i] = score
            self.dist2heavens[i] = get_score("Dist2Heaven", all_metrics[:4])
            if score < best[-1]:
                best = [i, score]
            if self.print_enabled:
                print ("\t" + "\t".join(
                    ["FFT(" + str(i) + ")"] + \
                    [str(x).ljust(5, "0") for x in all_metrics[4:] + \
                     [score if self.criteria == "Dist2Heaven" else -score]]))
        if self.print_enabled:
            print ("\tThe best tree found on training data is: FFT(" + str(best[0]) + ")")
        self.best = best[0]
        return best[0]

    "Given how the decision is made, get the description for the node."

    def describe_decision(self, t_id, level, metrics, reversed=False):
        cue, direction, threshold, decision = self.selected[t_id][level]
        tp, fp, tn, fn = metrics
        results = ["\'True\'", "\'False\'"]
        if type(threshold) != type(np.ndarray(1)):
            mappings = {">":"<=", "<":">="}
        else:
            mappings = {"inside":"outside", "outside":"inside"}        
        direction = mappings[direction] if reversed else direction+""
        description = ("\t| " * (level + 1) +
                       " ".join([str(cue), direction, str(threshold)]) +
                       "\t--> " + results[1 - decision if reversed else decision]).ljust(30, " ")
        pos = "\tFalse Alarm: " + str(fp) + ", Hit: " + str(tp)
        neg = "\tCorrect Rej: " + str(tn) + ", Miss: " + str(fn)
        if not reversed:
            description += pos if decision == 1 else neg
        else:
            description += neg if decision == 1 else pos
        return description

    "Given how the decision is made, get the performance for this decision."

    def eval_decision(self, data, cue, direction, threshold, decision):
        try:
            if type(threshold) != type(np.ndarray(1)):
                if direction == ">":
                    decided, undecided = data.loc[data[cue] > threshold], data.loc[data[cue] <= threshold]
                else:
                    decided, undecided = data.loc[data[cue] < threshold], data.loc[data[cue] >= threshold]
            else:
                decided = data.loc[(data[cue] >= threshold[0]) & (data[cue] < threshold[1])]
                undecided = data.loc[(data[cue] < threshold[0]) | (data[cue] >= threshold[1])]
        except:
            print ("Exception")
            return 1, 2, 3
        if decision == 1:
            pos, neg = decided, undecided
        else:
            pos, neg = undecided, decided        
        if "loc" in data:
            sorted_data = pd.concat([df.sort_values(by=["loc"], ascending=True) for df in [pos, neg]])
            loc_auc = get_auc(sorted_data)
        else:
            loc_auc = 0
        tp = pos.loc[pos[self.target] == 1]
        fp = pos.loc[pos[self.target] == 0]
        tn = neg.loc[neg[self.target] == 0]
        fn = neg.loc[neg[self.target] == 1]        
        return undecided, map(len, [tp, fp, tn, fn]), loc_auc
    

    def eval_point_split(self, level, cur_selected, cur_performance, data, cue, direction, threshold, decision):
        TP, FP, TN, FN = cur_performance
        undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
        tp, fp, tn, fn = self.update_metrics(level, self.max_depth, decision, metrics)
        # if the decision lead to no data, punish the score
        if sum([tp, fp, tn, fn]) == 0:
            score = float('inf')
        elif self.criteria == "LOC_AUC":
            score = loc_auc
        else:
            score = get_score(self.criteria, [TP + tp, FP + fp, TN + tn, FN + fn])
        if not cur_selected or score < cur_selected['score']:
            cur_selected = {'rule': (cue, direction, threshold, decision),\
                            'undecided': undecided,\
                            'metrics': [TP + tp, FP + fp, TN + tn, FN + fn], \
                            'score': score}
        return cur_selected    


    "Grow the t_id_th tree for the level with the given data. Also save its performance on the TRAIN data"

    def call_eval_point_split(self, data, t_id, level, cur_performance,cur_selected,decision):
        for cue in list(data):
                if cue == self.target:
                    continue
                if level == 0 and t_id == 0:
                    threshold = data[cue].median()
                    self.first_thresholds[cue] = threshold
                elif level == 0:
                    threshold = self.first_thresholds[cue]
                else:
                    threshold = data[cue].median()
                threshold = data[cue].median()
                for direction in "><":
                    cur_selected = self.eval_point_split(level, cur_selected, cur_performance, data, cue, direction, threshold, decision)
        return cur_selected


    def growEven(self, data, t_id, level, cur_performance):
        if level >= self.max_depth:
            return
        if len(data) == 0:
            print ("No data")
            return
        self.tree_depths[t_id] = level
        decision = self.structures[t_id][level]
        structure = tuple(self.structures[t_id][:level + 1])
        cur_selected = self.computed_cache.get(structure, None)        
        if not cur_selected:
            cur_selected = self.call_eval_point_split(data, t_id, level, cur_performance,cur_selected,decision)
            self.computed_cache[structure] = cur_selected
        self.selected[t_id][level] = cur_selected['rule']
        self.performance_on_train[t_id][level] = cur_selected['metrics'] + get_performance(cur_selected['metrics'])
        if level < 3 :
            self.selected[t_id+1][level] = self.selected[t_id][level]
            self.performance_on_train[t_id+1][level] = self.performance_on_train[t_id][level]
        if level == 2:
            #global store_cur_selected
            self.store_cur_selected = cur_selected
        self.growEven(cur_selected['undecided'], t_id, level + 1, cur_selected['metrics'])
    


    def growOdd(self, data, t_id, level, cur_performance):
        if level >= self.max_depth:
            return
        if len(data) == 0:
            print ("No data")
            return
        self.tree_depths[t_id] = level
        decision = self.structures[t_id][level]
        structure = tuple(self.structures[t_id][:level + 1])
        cur_selected = self.computed_cache.get(structure, None)        
        if not cur_selected:
            cur_selected = self.call_eval_point_split(data, t_id, level, cur_performance,cur_selected,decision)                        
            self.computed_cache[structure] = cur_selected
        self.selected[t_id][level] = cur_selected['rule']
        self.performance_on_train[t_id][level] = cur_selected['metrics'] + get_performance(cur_selected['metrics'])
        self.growOdd(cur_selected['undecided'], t_id, level + 1, cur_selected['metrics'])


    def print_tree(self, t_id):
        data = self.test
        depth = self.tree_depths[t_id]
        string=''
        if not self.node_descriptions[t_id]:
            self.node_descriptions[t_id] = [[] for _ in range(depth + 1)]
        for i in range(depth + 1):
            if self.node_descriptions[t_id][i]:                
                string+=self.node_descriptions[t_id][i][0]+'\n'
            else:
                cue, direction, threshold, decision = self.selected[t_id][i]
                undecided, metrics, loc_auc = self.eval_decision(data, cue, direction, threshold, decision)
                description = self.describe_decision(t_id, i, metrics)
                self.node_descriptions[t_id][i] += [description]                
                string+=description+'\n'
                if len(undecided) == 0:
                    break
                data = undecided
        return string


    "Update the metrics(TP, FP, TN, FN) based on the decision."

    def update_metrics(self, level, depth, decision, metrics):
        tp, fp, tn, fn = metrics
        if level < depth:  # Except the last level, only part of the data(pos or neg) is decided.
            if decision == 1:
                tn, fn = 0, 0
            else:
                tp, fp = 0, 0
        return tp, fp, tn, fn

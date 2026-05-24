import pandas as pd
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import math
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import torch
import numpy as np
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyClass:
    def __init__(self, dataset_result, train_indices):
        """"    Dataset Chinese Explanations:
        'Purpose(用途)', 'Category(类别)' are the column headers in the table.
        'OfficialReception(公务接待)', ' OfficialVehicles(公务用车)' '  ConventionExpense(会议培训)',
        ' Remuneration(人员薪酬)'' PropertyManagement(物业管理)'' Procurement(设备/服务采购)',' NULL(空)',
        are the class label in the dataset.
        """
        self.df_data = dataset_result
        self.df_train = self.df_data.iloc[train_indices]
        self.df_train_index = train_indices

        self.all_index = self.df_data.index
        self.df_test_index = self.all_index.difference(self.df_train_index)
        self.df_test = self.df_data.loc[self.df_test_index]



    def Max(self):
        data = self.df_data.loc[self.df_test_index]
        data['predict'] = "空"
        count_label = data['类别'].value_counts().to_dict()
        care_labels = []
        cols = []
        # print(count_label)
        for col in data.columns:
            if col in count_label:
                cols.append(col)

                sorted_data = data.sort_values(by=col, ascending=False)
                top_value = sorted_data.head(count_label[col])
                # 3. Extract corresponding 'index' column values
                top_indices = top_value['index'].tolist()
                data.loc[top_indices, 'predict'] = col
                care_labels.extend(top_indices)
            else:
                pass
        # print(data['predict'].value_counts().to_dict())

        counter = Counter(care_labels)
        # Find duplicate elements (count > 1)
        duplicates = [item for item, count in counter.items() if count > 1]

        # Process duplicate indices: find column with maximum value in cols
        if duplicates:
            # Filter rows with duplicate indices
            duplicate_rows = data.loc[duplicates, cols]
            # Find column with minimum value in each row
            max_cols = duplicate_rows.idxmin(axis=1)
            # Update predict column
            data.loc[duplicates, 'predict'] = max_cols
        # print(data['predict'].value_counts().to_dict())

        y_predict = data['predict'].values.tolist()
        y_test = data['类别'].values.tolist()

        return y_predict, y_test

    def Stacking(self):
        data = self.df_data
        cols = ['0','1','2','3','4','5']
        scaler = StandardScaler()

        y_train = data.loc[self.df_train_index, '类别'].values.tolist()
        x_train = scaler.fit_transform(data.loc[self.df_train_index, cols].values)
        y_test = data.loc[self.df_test_index, '类别'].values.tolist()
        x_test = scaler.fit_transform(data.loc[self.df_test_index, cols].values)

        # Train with optimal parameters
        clf = MLPClassifier(max_iter=1000, random_state=42, activation='tanh', alpha=0.001,
                            hidden_layer_sizes=(128, 64), learning_rate_init=0.01, solver='sgd')
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)

        p_proba = clf.predict_proba(x_test)
        p_proba = np.round(p_proba, 10)
        # Get class names
        class_names = clf.classes_

        # return y_predict,y_test,p_proba,class_names

        return y_predict, y_test


def run_ensemble_analysis(ensemble_method,dataset_result, train_indices):

    my_instance = MyClass(dataset_result, train_indices)
    if ensemble_method == "stacking":
        y_predict, y_test = my_instance.Stacking()
    elif ensemble_method == "max":
        y_predict, y_test = my_instance.Max()



    category_names = ['公务接待', '公务用车', '会议培训', '人员薪酬', '物业管理', '设备/服务采购', '空']
    a = Counter(y_predict)
    output_temp = classification_report(y_test, y_predict, output_dict=True, digits=3)  # output_dict=True
    #print(output_temp)
    new_output = report(output_temp, category_names, count=a, choice_option=False)  # opt.report_choice_option
    acc_count(y_test, y_predict)


def acc_count(y_test, y_predict):
    # Get all unique class labels
    unique_labels = sorted(set(y_test))  # Use sorted to ensure consistent order

    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    selected_labels = unique_labels[:6]

    selected_indices = [label_to_index[label] for label in selected_labels]

    cm = confusion_matrix(y_test, y_predict, labels=unique_labels)

    cm_subset = cm[selected_indices, :][:, selected_indices]

    correct_predictions = sum(cm[i, i] for i in selected_indices)
    total_samples = sum(sum(cm[i, :]) for i in selected_indices)
    total_samples = sum(sum(row) for row in cm[selected_indices, :])

    total_samples = sum(cm[i, :].sum() for i in selected_indices)
    accuracy = correct_predictions / total_samples

def report(output, target_names, count, choice_option=False):
    # Calculate arithmetic mean of classes
    geometric_precision, geometric_recall, geometric_f1_score, sum_support = 1., 1., 1., 0
    count_precision_number = 0
    count_recall_number = 0
    count_f1_number = 0
    for name in target_names:
        geometric_precision *= output[name]['precision']
        count_precision_number += 1
        geometric_recall *= output[name]['recall']
        count_recall_number += 1
        geometric_f1_score *= output[name]['f1-score']
        count_f1_number += 1
        sum_support += output[name]['support']
    dict_new_report = {'geometric mean': {'precision': 0., 'recall': 0., 'f1-score': 0., 'support': 0}}
    dict_new_report['geometric mean']['precision'] = math.pow(geometric_precision, 1 / count_precision_number)
    dict_new_report['geometric mean']['recall'] = math.pow(geometric_recall, 1 / count_recall_number)
    dict_new_report['geometric mean']['f1-score'] = math.pow(geometric_f1_score, 1 / count_f1_number)
    dict_new_report['geometric mean']['support'] = sum_support

    macro_avg_new_precision, macro_avg_new_recall, macro_avg_f1_score, sum_support = 0., 0., 0., 0
    count_precision_number = 0
    count_recall_number = 0
    count_f1_number = 0
    count_number = 0
    # dict_new_report['macro majority mean'] = {'precision': 0., 'recall': 0., 'f1-score': 0., 'support': 0}
    # dict_new_report['macro majority mean']['precision'] = geometric_precision/count_number
    # dict_new_report['macro majority mean']['recall'] = geometric_recall
    # dict_new_report['macro majority mean']['support'] = sum_support

    # Collect metrics to be printed
    report_name = []
    for key_name in (list(output[target_names[0]].keys()) + ['predict_number']):
        # print(key_name)
        report_name.append(key_name)
    # Classes to be printed

    classification_name = target_names + ['accuracy', 'macro avg', 'weighted avg',
                                          'geometric mean', ]
    dict_new_report_ = {}
    for name in classification_name:
        if name in output.keys():
            dict_new_report_[name] = output[name]
            if count != None and (name not in ['accuracy', 'macro avg', 'weighted avg']):
                if name in count.keys():
                    dict_new_report_[name]['predict_number'] = count[name]
                    output[name]['predict_number'] = count[name]
                else:
                    dict_new_report_[name]['predict_number'] = 0
                    output[name]['predict_number'] = 0
        else:
            dict_new_report_[name] = dict_new_report[name]
    dict_new = {}
    for key, value in dict_new_report_.items():
        value_list = []
        if isinstance(value, dict):
            for key_, value_ in value.items():
                if isinstance(value_, float):
                    value_ = round(value_, 3)
                value_list.append(value_)
            if len(value_list) != 5:
                value_list.append(['NULL'] * (5 - len(value_list)))
            dict_new[key] = value_list
        else:
            value_list = [value]
            value_list += ['NULL'] * (5 - len(value_list))
    report_name_all = ['name'] + list(output[target_names[0]].keys())
    table = PrettyTable(report_name_all)

    for key, value in dict_new.items():
        table.add_row([key] + value)
        # if choice_option == True:
        # if key == 'macro avg' or key == 'macro majority mean': # or key == 'geometric mean'
        # table.add_row([key] + value)
        # else:
        # table.add_row([key]+value)
    # table.align["name"] = "l"
    print(table)
    print('accuracy: {}'.format(output['accuracy']))

    return output


if __name__ == '__main__':
    print("test")

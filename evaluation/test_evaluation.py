import glob
import argparse
import pandas as pd
import copy
import os
import json
from sklearn.metrics.cluster import  adjusted_rand_score, normalized_mutual_info_score

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default="rico_seq2seq",
                 type=str, help="data name")
par.add_argument("-e", "--evaluation", default="nmi", choices=["nmi", "ari"],
                 type=str, help="Select the evaluation method(nmi, ari)")
args = par.parse_args()

clustering_algorithms = ['dec']
fusion_types = []
weights = []
with open('ground_truth_list.json', 'r') as f:
    ground_truth = json.load(f)

total_best_result = dict()

def NMI(data_name, ground_truth_labels, clustering_result, c_num):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth[c_num].keys():
        for g in ground_truth[c_num][g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    nmi_score = normalized_mutual_info_score(ground_truth_labels, clustering_resul_labels)
    if total_best_result[data_name] < nmi_score:
        total_best_result[data_name] = nmi_score

    return ("%.3f" % (nmi_score))

def ARI(data_name, ground_truth_labels, clustering_result, c_num):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth[c_num].keys():
        for g in ground_truth[c_num][g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    ari_score = adjusted_rand_score(ground_truth_labels, clustering_resul_labels)
    if total_best_result[data_name] < ari_score:
        total_best_result[data_name] = ari_score

    return ("%.3f" % (ari_score))

def save_csv_file(data_name, result_dict):
    result_path = "csv/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = "csv/" + args.evaluation + "/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    del result_dict["List"]
    df = pd.DataFrame(result_dict)
    df.to_csv(result_path + data_name + ".csv", index=False)

def run(data_name, c_num):
    total_best_result[data_name] = 0.0
    result_dict = dict()
    result_dict["List"] = list()
    for i, clustering_algorithm in enumerate(clustering_algorithms):
        result_dict[clustering_algorithm] = list()
        result_path = '../clustering/result/' + data_name + "/"

        clustering_result = dict()
        clustering_result_temp = dict()

        for dirs in glob.glob(result_path+"*"):
            with open(dirs, 'r') as fd:
                res = fd.read()
            clustering_result_temp[dirs.split("/")[-1]] = " ".join(res.split('\n')).split()

        clustering_result_list = list(clustering_result_temp.keys())
        clustering_result_list.sort()
        for c_key in clustering_result_list:
            clustering_result[c_key] = clustering_result_temp[c_key]

        ground_truth_labels = list()
        for v in ground_truth[c_num].values():
            ground_truth_labels.extend(v)

        if args.evaluation == "nmi":
            result_dict[clustering_algorithm].append(
                    NMI(data_name, ground_truth_labels, clustering_result, c_num))
        elif args.evaluation == "ari":
            result_dict[clustering_algorithm].append(
                    ARI(data_name, ground_truth_labels, clustering_result, c_num))

    save_csv_file(data_name, result_dict)

if __name__ == "__main__":
    run("idec_se_34_conv_dnn_jsd", "34")
    order_total_best_result = sorted(total_best_result.items(), reverse=True, key=lambda item: item[1])

    fd = open("csv/" + args.evaluation  + "/total_best_result.txt", "w")
    print("Total {} Result Length : {}".format(args.evaluation, len(total_best_result)))
    fd.write("Total {} Result Length : {}\n".format(args.evaluation, len(total_best_result)))

    for i, items in enumerate(order_total_best_result):
        print("Top{} = {} : {}".format(i+1, items[0], "%.3f"%items[1]))
        fd.write("Top{} = {} : {}".format(i+1, items[0], "%.3f"%items[1]+"\n"))
    fd.close()

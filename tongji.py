import matplotlib.pyplot as plt
import matplotlib
import json

path = 'result_final.json'
dataset_json = json.load(open(path, encoding='utf8'))


humor_list = []
nonhumor_list = []
sarcasm_list = []
nonsarcasm_list = []

count_all_sentences = 0
episode_sentence_num_list  = []

for i in range(1,11):
    count_humor = 0
    count_nonhumor = 0
    count_sarcasm = 0
    count_nonsarcasm = 0

    count_episode_sentence_num = 0
    for key ,value in dataset_json.items():

        if key.split('_')[0] == str(i):
            count_all_sentences += 1
            count_episode_sentence_num += 1

            if value['sarcasm'] == '0':
                count_nonsarcasm += 1
            if value['sarcasm'] == '1':
                count_sarcasm += 1
            if value['humor'] == '0':
                count_nonhumor += 1
            if value['humor'] == '1':
                count_humor += 1

    humor_list.append(count_humor)
    nonhumor_list.append(count_nonhumor)
    sarcasm_list.append(count_sarcasm)
    nonsarcasm_list.append(count_nonsarcasm)
    episode_sentence_num_list.append(count_episode_sentence_num)

a = sum(humor_list)
b = sum(nonhumor_list)
c = sum(sarcasm_list)
d = sum(nonsarcasm_list)
e = sum(episode_sentence_num_list)

label_list = [i for i in range(1,11)]

x = range(len(humor_list))
rects1 = plt.bar(x, height= nonhumor_list, width=0.45, alpha=0.8, color='orange', label="sarcasm")
rects2 = plt.bar(x, height=humor_list, width=0.45, color='blue', label="nonsarcasm", bottom=nonhumor_list)
# plt.ylim(0, 80)
plt.ylabel("sentences")
plt.xticks(x, label_list)
plt.xlabel("episode")
plt.legend()
plt.show()
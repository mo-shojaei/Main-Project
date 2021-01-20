from collections import defaultdict
import re
def get_frequency(filenames, saveFiles, saveFile):
    dics = [{},{}]
    significance = {}
    for i in range(0, len(filenames)):
        dics[i] = processFile(filenames[i])
    relevant_dic = {k: v for k, v in sorted(dics[0].items(), key=lambda item: item[1], reverse=True)}
    irrelevant_dic = {k: v for k, v in sorted(dics[1].items(), key=lambda item: item[1], reverse=True)}

    with open(saveFiles[0], "w+") as file:
        for i in relevant_dic:
            file.write(i + "  " + str(relevant_dic[i]) + "\n")
    with open(saveFiles[1], "w+") as file:
        for i in irrelevant_dic:
            file.write(i + "  " + str(irrelevant_dic[i]) + "\n")
    with open(saveFile, "w+") as file:
        for i in relevant_dic:
            val = relevant_dic[i] - dics[1][i]
            significance[i] = val
        significance = {k: v for k, v in sorted(significance.items(), key=lambda item: item[1], reverse=True)}
        for i in significance:
            file.write(i + "  " + str(significance[i]) + "\n")


def processFile(filename):
    dic = defaultdict(int)
    lst = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            lst.extend(re.split(r'[;,\s]\s*', line))
    for i in lst:
        i = i.strip()
        i = i.lower()
        dic[i] += 1
    return dic

get_frequency(["keywordsRelevant.txt", "keywordsIrrelevant.txt"], ["freqRel.txt","freqIrrel.txt"], "significance.txt")

#get_frequency(["keywordsRelVenmo.txt", "keywordsIrrrVenmo.txt"], "VenmoSig")
# get_frequency("keywordsIrrelevant.txt", "frequencyIrrelevant.txt")

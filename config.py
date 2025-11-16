def get_ids(file_path):
    id_list = []
    with open(file_path, "r") as f:
       for line in f:
           id_list.append(line.strip())    

    return id_list

def get_class_names(ids, words_file_path):
    """Find corresponding class names from words.txt based on ids"""
    id_to_name = {}
    
    with open(words_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    word_id = parts[0]
                    class_name = parts[1]
                    id_to_name[word_id] = class_name
    
    # Find corresponding class names
    class_names = []
    for word_id in ids:
        if word_id in id_to_name:
            class_names.append(id_to_name[word_id])
        else:
            class_names.append(f"Unknown ({word_id})")
    
    return class_names

# Get ids from tiny-imagenet-200 in any way
ids = sorted(get_ids("../imagenet/tiny-imagenet-200/wnids.txt"))[:4]



'''
0: n01443537 -> goldfish, Carassius auratus
1: n01629819 -> European fire salamander, Salamandra salamandra
2: n01641577 -> bullfrog, Rana catesbeiana
3: n01644900 -> tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
4: n01698640 -> American alligator, Alligator mississipiensis
5: n01742172 -> boa constrictor, Constrictor constrictor
6: n01768244 -> trilobite
7: n01770393 -> scorpion
'''


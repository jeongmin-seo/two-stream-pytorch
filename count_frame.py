import matplotlib.pyplot as plt

def count_frame(txt_path, dic):
    f = open(txt_path)
    for line in f.readlines():
        num_frame = int(line.split(' ')[1])
        if num_frame not in list(dic.keys()):
            dic[num_frame] = 1
        else:
            dic[num_frame] = dic[num_frame] + 1
    return dic

txt_path = '/home/jm/Two-stream_data/HMDB51/train_split1.txt'
dict = {}
dict = count_frame(txt_path=txt_path, dic=dict)
txt_path = '/home/jm/Two-stream_data/HMDB51/test_split1.txt'
dict = count_frame(txt_path, dict)

max_frame = max(list(dict.keys()))
print(max_frame, dict[max_frame])
print(sorted(list(dict.keys())))
plt.bar(list(dict.keys()), list(dict.values()))
plt.xlim(-1, max_frame)
plt.show()





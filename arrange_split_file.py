"""
This code is arrange train split file in JHMDB dataset.
"""

# requirememt library
import os
import re

save_root = "/home/jm/hdd/JHMDB"
file_root = os.path.join(save_root, "splits")
data_root = os.path.join(save_root, "Rename_Images")
class_name = sorted(os.listdir("/home/jm/hdd/JHMDB/Rename_Images"))[3:]
print(class_name)
for i in range(3):
    test_name = os.path.join(save_root, "test_split_%d.txt" %(i+1))
    train_name = os.path.join(save_root, "train_split_%d.txt" %(i+1))
    test_file = open(test_name, "w")
    train_file = open(train_name, "w")

    for j, file_name in enumerate(sorted(os.listdir(file_root))):

        if j%3 != i:
            continue

        read_name = os.path.join(file_root, file_name)
        read_file = open(read_name, 'r')


        for line in read_file.readlines():
            split_line = re.split("[ .]+", line)
            class_number = int(j/3)
            save_line = class_name[class_number] + "/" + split_line[0]
            last_image_name = sorted(os.listdir(os.path.join(data_root, save_line)))[-1]
            n_frame = int(re.split('[.]+', last_image_name)[0])
            save_line += " " + "%d" % n_frame + " " + "%d\n" % (j/3)

            if int(split_line[-1]) == 1:
                train_file.write(save_line)
            elif int(split_line[-1]) == 2:
                test_file.write(save_line)

        read_file.close()
    train_file.close()
    test_file.close()




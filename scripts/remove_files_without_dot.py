import os

for dirpath, _, files in os.walk('./'):
    for file_name in files:
        if not ('.' in file_name):
            # os.system("rm -rf {}".format(file_name))

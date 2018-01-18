import os
import re

root = './class_people'


# train = './train'
# for dir in os.listdir(root):
#     path = '{}/{}'.format(root, dir)
#     if os.path.isdir(path):
#         files = os.listdir(path)
#         files = list(filter(lambda f: f[-5:-4] != 'N' and not f.startswith('.'), files))
#         print(files)
#         for file in files:
#             filePath = '{}/{}'.format(path, file)
#             fileToRename = re.sub(r'_\w.jpg','_N.jpg', filePath)
#             if os.path.exists(fileToRename) and os.path.isfile(fileToRename):
#                 print('removing file ', filePath)
#                 os.remove(filePath)
#                 print('renaming file {} to {}'.format(fileToRename, filePath))
#                 os.rename(fileToRename, filePath)

def renameFiles(path, prefix, postfix, countFrom=1, version = 'F_'):
    files = os.listdir(path)
    for f in files:
        if not f.startswith('.DS'):
            os.rename('{}/{}'.format(path, f), '{}/{}{}{}.{}'.format(path,version, prefix, countFrom, postfix))
            countFrom += 1


renameFiles('./dummies/10_30', prefix='dummy', postfix='jpg', version='D1_')

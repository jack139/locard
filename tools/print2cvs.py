# coding:utf-8

import sys, os
import json

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python3 %s <path>" % sys.argv[0])
        sys.exit(2)

    filepath = sys.argv[1]

    file_list = os.listdir(filepath)
    file_list = sorted(file_list)

    #print('姓名,性别,民族,出生,地址,身份号码,文件名')

    for i in file_list:
        filename, ext = os.path.splitext(i)
        if ext!='.txt':
            continue

        with open(os.path.join(filepath, i), 'r') as f:
            data = json.load(f)

        print('"%s","%s","%s","%s","%s","\'%s","%s"'%(
            data['name'],
            data['sex'],
            data['nation'],
            data['birth'],
            data['addr'],
            data['idnum'],
            filename,
        ))

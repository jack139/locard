# coding:utf-8

import sys, os
import json
import cv2

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("usage: python3 %s <path> <json_path>" % sys.argv[0])
        sys.exit(2)

    filepath = sys.argv[1]
    jsonpath = sys.argv[2]

    file_list = os.listdir(filepath)
    file_list = sorted(file_list)

    for i in file_list:
        filename, ext = os.path.splitext(i)

        # 去色
        #im = cv2.imread(os.path.join(filepath, i))[:, :, ::-1] # 去色
        #cv2.imwrite(os.path.join(filepath, i),im)

        if not os.path.exists(os.path.join(jsonpath, filename+'.json')):
            print('need json!', i)
            continue

        with open(os.path.join(jsonpath, filename+'.json'), 'r') as f:
            data = json.load(f)

        if len(data['shapes'])!=2:
            print('shapes err!', i)
            continue

        if data['shapes'][0]['label']=='card' and data['shapes'][1]['label']=='photo':
            p1 = data['shapes'][0]['points']
            p2 = data['shapes'][1]['points']
        elif data['shapes'][0]['label']=='photo' and data['shapes'][1]['label']=='card':
            p1 = data['shapes'][1]['points']
            p2 = data['shapes'][0]['points']
        else:
            print('points err!', i)
            continue

        if p1[0][0]<p2[0][0] and p1[0][1]<p2[0][1] and p1[1][0]>=p2[1][0] and p1[1][1]>p2[1][1]:
            continue

        print('photo position err!', i)

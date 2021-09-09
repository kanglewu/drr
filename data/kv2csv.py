# -*- coding: UTF-8 -*-
# not used online
import re
CTRL_A='\x01'
CTRL_B='\x02'
CTRL_C='\x03'
#top N lines for test
testCnt=2000
fin = open('origin.txt', 'r')
fout = open('channel.test', 'w')
fout2 = open('channel.data', 'w')

columns=['item__is_ka','item__is_standard','item__match_type','item__match_score','item__image_cnt','item__item_price','item__jfy_exp_cate_pv_14d']
valuesMap={}
cnt=0
labelDict={}
labelDict['0']=0
labelDict['1']=0
for line in fin.readlines():
    #print line
    # print cnt
    array=line.strip().split('\t')
    user=array[7]
    floor_id=array[2]
    item=array[3]
    features_kv=array[4]
    features_list=array[6]
    label=array[5].split(';')[0]
    # if label=='0':
    #     if  not ('11'   in item ) :
    #         continue;
    labelDict[label]+=1

    #40个序列特征
    values=[]
    #print features_list
    for v in re.split(',|:|;|#',features_list):
        values.append(v)
    # print len(values)
    values.append(item)
    values.append(floor_id)
    values.append(user)

    #print features_kv
    vd={}
    for kv in features_kv.split(CTRL_A):
      kvArray = kv.split(CTRL_B)
      if len(kvArray)!=2:
          print kv
      k=kvArray[0]
      v=kvArray[1]
      if  k in columns:
        vd[k]=v
        if k not in valuesMap:
            valuesMap[k] = set()
        valuesMap[k].add(v)
    for c in columns:
        values.append(vd[c])

    values.append(label)
    if len(values) !=51:
        print cnt,'error'
    # print len(values)
    output=",".join(values)+"\n"
    cnt +=1
    if cnt<testCnt:
        fout.write(output)
    else:
        fout2.write(output)
fin.close()
fout.close()
fout2.close()

print labelDict
print 'pos_rate=',1.*labelDict['1']/(labelDict['1']+labelDict['0']+0.)
for k in valuesMap.keys():
    print k
    if len(valuesMap[k])<20:
        print k,valuesMap[k]
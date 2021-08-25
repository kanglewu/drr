CTRL_A='\x01'
CTRL_B='\x02'
CTRL_C='\x03'
#top N lines for test
testCnt=1000
fin = open('orign.txt', 'r')
fout = open('channel.test', 'w')
fout2 = open('channel.data', 'w')

columns=['is_ka','is_standard','match_type','match_score','image_cnt','item_price','jfy_exp_cate_pv_14d']
valuesMap={}
cnt=0
for line in fin.readlines():
    array=line.strip().split('\t')
    label=array[1].split(';')[0]
    kvs=array[2]
    values=[]
    for kv in kvs.split(CTRL_B):
      kvArray = kv.split(CTRL_C)
      k=kvArray[0]
      v=kvArray[1]
      if k in columns:
        values.append(v)
        if k not in valuesMap:
            valuesMap[k] = set()
        valuesMap[k].add(v)


    values.append(label)
    if len(values)==0:
        break
    output=",".join(values)+"\n"
    cnt +=1
    if cnt<testCnt:
        fout.write(output)
    else:
        fout2.write(output)
fin.close()
fout.close()
fout2.close()

for k in valuesMap.keys():
    if len(valuesMap[k])<20:
        print k,valuesMap[k]
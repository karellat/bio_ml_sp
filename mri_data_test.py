import mri_data

data = mri_data.MRI_DATA(sizes=(70, 20, 10))
print(data.train.data['images'].shape)
#print(data.train.data['images'][0])
print(data.dev.data['images'].shape)
print(data.dev.data['labels'])
print(data.dev.data['categories'])
print(sum(map(lambda x: 1 if x == 3 else -1, data.dev.data['labels']))/len(data.dev.data['labels']))
print(sum(map(lambda x: 1 if x == 1 else -1, data.dev.data['categories']))/len(data.dev.data['categories']))
print(len(data.dev.data['categories']))
#print(data.dev.data['categories'].shape)
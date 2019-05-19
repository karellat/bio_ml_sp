import mri_data

data = mri_data.MRI_DATA(sizes=(60, 20, 20))
print(data.train.data['images'].shape)
print(data.train.data['images'][0])
print(data.test.data['labels'].shape)
print(data.dev.data['categories'].shape)
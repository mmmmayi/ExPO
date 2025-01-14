invalid = open('/home2/mayi_data/dataset/voxceleb/voxceleb2/voxceleb2_wav/invalid_pre_pho2','r')
lines = invalid.readlines()
invalid_list = {}
for line in lines:
        line=line.strip('\n')
        parts = line.split('/')
        line = ('/').join(parts[1:])
        invalid_list[line]=0
origin = open('data/vox2_pooling/wav.scp','r').readlines()
new = open('data/vox2_pooling/new_wav.scp','w')
for line in origin:
    key = line.split(' ')[0]
    if key in invalid_list:
        continue
    else:
        new.write(line)

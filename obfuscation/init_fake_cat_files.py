import datasets

wfu_dataset = datasets.load_dataset('../wfudata', trust_remote_code=True)

cats = {
    'EMAIL':[],
    'HOSPITAL':[],
    'IPADDRESS':[],
    'LOCATION':[],
    'OTHER':[],
    'URL':[]
}

for i in range(len(wfu_dataset['train'])):
    phi = wfu_dataset['train']['phi'][i]
    text = wfu_dataset['train']['text'][i]
    prev_offset = None
    for phi_type, offset in zip(phi['types'], phi['offsets']):
        if phi_type in cats:
            cats[phi_type].append(text[offset[0]:offset[1]])

for key in cats.keys():
    with open(f'fake_{key.lower()}_list.txt', 'w') as fid:
        fid.writelines([x.strip().replace('\n','') + '\n' for x in cats[key]])

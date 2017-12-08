files = [open('train.txt', 'r'), open('test.txt', 'r'), open('valid.txt', 'r')]
ofiles = [open('relation2id.txt', 'w'), open('entity2id.txt', 'w')]

s = [set(), set()]

for thefile in files:
    for line in thefile:
        items = line.strip().split()
        if len(items) < 4 or items[3] == '1':
            s[1].add(items[0])
            s[0].add(items[1])
            s[1].add(items[2])
    thefile.close()

for i in range(2):
    iid = 0
    for item in s[i]:
        ofiles[i].write(item + '\t' + str(iid) + '\n')
        iid += 1
    ofiles[i].close()

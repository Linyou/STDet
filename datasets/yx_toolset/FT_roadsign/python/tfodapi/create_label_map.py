import os

# {'soa': 2334, 'ooa': 415, 'io': 4984, 'wo': 1023, 'roa': 456,
#  'np': 851, 'ors': 4909, 'cf': 1345, 'p10': 497, 'p11': 968,
#  'p26': 537, 'p20': 551, 'p23': 564, 'p19': 548, 'pne': 590,
#  'rn': 1071, 'ps': 572, 'loa': 457, 'pv': 0, 'p5': 529, 
#  'lo': 561, 'sloa': 475, 'cross': 2235, 'sroa': 538, 'tl': 3129,
#  'rg': 3167, 'pg': 732, 'sc': 2054, 'ph': 64, 'ro': 1746,
#   'pn': 1317, 'po': 1266, 'pl': 2056, 'pm': 509}

od_classes = ['io', 'wo', 'ors', 'p10', 'p11', 
              'p26', 'p20', 'p23', 'p19', 'pne', 
              'rn', 'ps', 'p5', 'lo', 'tl',
              'pg', 'sc', 'ro', 'pn', 'po',
              'pl', 'pm', 'ph']
seg_classes = ['roa', 'loa', 'soa', 'sloa', 'sroa',
			   'ooa', 'cf', 'rg', 'np', 'cross',
			   'ld','zyfgd','lcfgd','lmj','sfwl',
			   'sdwl','sfyl','sdyl','dfyl','sl']

file = open('ft_od_label_map.pbtxt', 'w')
for i, label in enumerate(seg_classes):
    line = 'item {\n'
    line += '\tid: ' + str(i + 1) + '\n'
    line += '\tname: "' + label + '"\n}\n'
    file.write(line)
file.close()
pred = ['xyusbafbjavud.jpg',
        'vbncmdjf__.jpg',
	    'bajvbevabu.jpg']
filename = "sample.txt"
# if use japanese
# with open('sample.txt', 'w', encoding='utf-8') as f:
print('save')
with open(filename, 'w') as file:
    for p in pred:
	    file.write(p+'\n')

print('open')
f = open(filename, "r")
print(f.read())
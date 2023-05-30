# sample2.py

import sample1 as pre_loaded

args = pre_loaded.args

# print the user's name and age
print('Your name is {} and you are {} years old.'.format(args.name, args.age))

print(pre_loaded.var1)
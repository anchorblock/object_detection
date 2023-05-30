# sample1.py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--age', type=int, required=True)

args = parser.parse_args()

# # print the user's name and age
# print('Your name is {} and you are {} years old.'.format(args.name, args.age))

var1 = args.age
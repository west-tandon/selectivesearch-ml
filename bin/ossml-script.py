import argparse
import json
import sys

import ossml.payoffs

parser = argparse.ArgumentParser(description='Machine Learning Package for Ordered Selective Search', prog='ossml')
commands = parser.add_subparsers(help='command', dest='command')
commands.required = True

# Train
parser_train = commands.add_parser('train')
parser_train.add_argument('type', type=str, choices=['impact', 'cost'])
parser_train.add_argument('json', type=argparse.FileType('r'), help='file with feature description in JSON format')
parser_train.add_argument('output', type=str)

# Predict
parser_predict = commands.add_parser('predict')
# TODO

args = parser.parse_args()

if args.type == 'impact':

    # Train impact
    ossml.payoffs.run_train(json.loads(args.json.read()), args.output)

elif args.type == 'cost':
    pass

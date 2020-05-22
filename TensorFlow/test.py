#from __future__ import print_function
import argparse
import sys

def main():
    print 12121

def main1():
    print "hello"

for i in sys.modules.keys():
    print i
    pass
    #print sys.modules[i]

sys.modules['__main__'].main1()

print sys.modules['__main__'].main()

#print bool(None) == True
#print bool(None) == False

#sys.exit(1)

'''
parser = argparse.ArgumentParser(description='test parsing arguments')

parser.add_argument('pos1', nargs='*')
parser.add_argument('pos2')
parser.add_argument('-o1')
parser.add_argument('-o2')
parser.add_argument('pos3', nargs='*')

print sys.argv

arg,w = parser.parse_known_args(sys.argv[1:])
print arg
print w
'''
# print parser.print_help()


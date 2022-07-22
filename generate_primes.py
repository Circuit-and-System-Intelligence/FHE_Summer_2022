#!/usr/bin/env python
# Date: 7/20//2022
# 
# This program will read a file of primes, create a list of primes and 
# save it as a binary. 

import pickle as pkl

def main():
	# this will open the prime text file, then read the primes
	# into a list, and then convert those primes into integers.
	# afterwards it will save the list into a binary file 

	with open('100000.txt','r') as f:
		line = f.readlines()
		line = line[0]
		primes = line.split(',')
		primes = [int(p) for p in primes]

		with open('prime.pickle','wb') as prim_file:
				pkl.dump( primes, prim_file )


if __name__ == '__main__':
	main()

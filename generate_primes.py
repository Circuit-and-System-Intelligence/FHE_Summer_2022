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

		with open('100000primes.pickle','wb') as prim_file:
				pkl.dump( primes, prim_file )

def SieveOfEratosthenes(num):
	# this function will generate all primes less
	# than given num

	try: 
		prime = [True]*(num+1)
	except:
		prime = [True for i in range(num+1)]

	p = 2
	while ( p*p <= num ):

		if prime[p] == True:
			for i in range(p*p, num+1, p):
				prime[i] = False
		p += 1
	
	prime.pop(0)
	prime.pop(0)

	primes = []
	for ind, p in enumerate(prime):
		if p:
			primes.append(ind+2)
	
	# print( primes )
	print( primes[-5:] )
	print( len(primes) )
	return primes

def Sieve(num):
	# find all primes up to and including num

	primes = [-1]*(2**32)
	primes[0] = 2
	sz = 1
	p = 3

	while ( p < num ):
		is_prime = True
		for prim in primes[:sz]:
			if prim*prim > p:
				break
			if p % prim == 0:
				is_prime = False

		if is_prime:
			primes.append(p)
			sz += 1

		p += 1

	return primes

def testSieve():
	p = SieveOfEratosthenes(1299827)

	with open('100000primes.pickle','rb') as f:
		primes = pkl.load(f)
		print(f'Sieve == prime.pickle: {p == primes}')

def generatePickle():
	# this will save generated primes to pickle file
	# primes = Sieve(2**20)
	primes = SieveOfEratosthenes(2**20)
	return
	with open('./bin/primes.pickle','wb') as f:
		pkl.dump( primes, f )

if __name__ == '__main__':
	# main()
	# SieveOfEratosthenes(2**23)
	# testSieve()
	generatePickle()


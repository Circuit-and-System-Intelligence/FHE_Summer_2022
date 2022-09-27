#!/usr/bin/env python
# Date: 7/25/2022
# Create a class to perform transformations
# into and out of a defined residue number system


class RNS(object):

	def __init__(self, primes=None):
		self.primes = primes
		if (self.primes == None):
			self.basic_primes()

		self.gen_P()

		self.get_inverses()

	def basic_primes(self):
		# this function will set 
		# primes to [2,3,5]
		self.primes = [2,3,5]

	def gen_P(self):
		# this will get the large modulus for
		# the given primes
		P = 1
		for p in self.primes:
			P *= p

		self.P = P
	
	def get_inverses(self):
		# this function will get the 
		# inverses for the primes
		self.T = []

		for prim in self.primes:
			x = self.extended_euclidean( self.P//prim, prim )
			self.T.append( x )

	def extended_euclidean(self, a, n):
		# this function will calculate the inverse of a mod n
		# such that t*a === 1 mod n
		# 
		# this will be adapted from pseudo-code on wikipedia

		t = 0
		newt = 1
		r = n
		newr = a

		while newr != 0:
			quo = r // newr
			t, newt = newt, (t - quo*newt)
			r, newr = newr, (r - quo*newr)

		if t < 0:
			t += n

		return t

	def to_RNS(self, x):
		# this will convert x into RNS

		xi = []
		for prim in self.primes:
			xi.append( x % prim )

		return xi

	def from_RNS(self, xi):
		# this will convert RNS back to base10 integers

		x = 0
		for r,prim,t in zip(xi,self.primes,self.T):
			x += (self.P//prim)*r*t

		x = x % self.P
		return x

	def mult(self, xi, yi):
		# this will perform a multiplication
		zi = [(i*j)%k for i,j,k in zip(xi,yi,self.primes)]
		return zi

	def add(self, xi, yi):
		# this will perform an addition
		zi = [(i+j)%k for i,j,k in zip(xi,yi,self.primes)]
		return zi


def main():

	rsys = RNS([3,5,7,11,13,17,19,23])
	# rsys = RNS([3,5,7,11])

	x = 318
	y = 116

	rx = rsys.to_RNS(x)
	ry = rsys.to_RNS(y)

	rz = rsys.mult(rx,ry)

	z = rsys.from_RNS(rz)
	print(f'x*y: {(x*y)%rsys.P}')
	print(f'z:   {z}')

if __name__ == '__main__':
	main()

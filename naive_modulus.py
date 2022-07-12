#!/usr/bin/env python
# Date: 6/27/2022
# Create a function to perform mod function with
# a loop consisting of simple operations
# 
# Implement a naive_modulus function
# and test it.

from random import randint

def naive_modulus(dividend, divisor):
	# this function will perform a loop
	# that will return dividend % divisor

	# sub_div = ( divisor ^ -1) + 1
	if ( dividend*divisor > 0 ):
		sub_divisor = divisor * -1
	else:
		sub_divisor = divisor

	if (divisor == 0):
		raise ZeroDivisionError

	loop = 0

	x = dividend
	y = divisor

	while True:
		if ( abs(dividend) < abs(divisor) ) and (dividend*divisor >= 0) :
			# print(f'loop = {loop}')
			# print(f'floor = {abs(x//y)}')
			assert dividend == x%y
			return dividend
		if loop > 2 ** 32:
			print(f'x: {x}')
			print(f'y: {y}')
			raise ArithmeticError
		dividend += sub_divisor
		loop += 1


def barrett(a, n):
	# this function will implement barrett reduction algorithm
	# to optimize modulus reduction

	a = int(a)
	n = int(n)

	# calculate k and q for reduction
	# k = nextpow2(n)
	k = 1 if n == 0 else (n-1).bit_length()
	'''
	k = 0
	mask = 1
	while mask < abs(n):
		mask = mask << 1
		k += 1
	'''
	
	two_k = k << 1
	# q = floor(bitshift(1,2*k)/n)
	q = int( (1 << two_k) // n )

	# t = bitshift(a*q, -2*k)
	t = (a*q) >> two_k

	# c = a - t*n
	c = a - (t*n)

	# if limit c from [0,2n) to [0,n)
	if c > n:
		c -= n
	
	assert c == a % n
	return c

class Montgomery():

	def __init__(self,n=7,powr=4):
		# define the n and r
		self.n = n
		self.r = 1 << powr
		self.logr = powr
		self.invr = self.inverse_modulus(self.r,self.n)
		self.k = (self.r*self.invr - 1) // self.n

	def inverse_modulus(self, a, n):
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

	def toMont(self, a):
		# this function will convert into the Montgomery system
		# return barrett( (a*self.r), self.n )
		return (a*self.r) % self.n

	def fromMont(self, a):
		# this function will convert out of the Montgomery system
		# return barrett( (a*self.invr), self.n )
		return ( a * self.invr ) % self.n

	def multiplication(self, a, b):
		# this will perform multiplication on two
		# numbers already in the Montgomery system
		# and return a number in the system
		x = a*b
	
		s = ( x * self.k ) & (self.r-1)

		t = x + (s*self.n)

		u = t >> self.logr

		if u > self.n:
			u -= self.n

		return u

def test_mont():
	# this function will test the montgomery system

	mont = Montgomery(n=63,powr=6)

	a = mont.toMont( 1234 )
	b = mont.toMont( 56 )

	c = mont.multiplication( a, b )

	print( mont.fromMont( c ) ) 


def main():
	# run a test of naive_modulus

	for i in range(10):
		x = randint(0,2**12)
		y = randint(2**6,2**8)
		x *= -1
		# y *= -1
		
		print(f'x: {x}\ty: {y}')
		#my_z = naive_modulus(x,y)
		my_z = barrett(x,y)
		print(f'my_z: {my_z}')
		z = x % y
		print(f'z: {z}')
		print(' ')

		if my_z != z:
			return
	print(f'the functions are equal')

if __name__ == '__main__':
	#main()
	test_mont()

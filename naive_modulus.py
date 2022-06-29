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
	k = 0
	mask = 1
	while mask < abs(n):
		mask = mask << 1
		k += 1
	
	two_k = k << 1
	# q = floor(bitshift(1,2*k)/n)
	q = int( (1 << two_k) / n )

	# t = bitshift(a*q, -2*k)
	t = (a*q) >> two_k

	# c = a - t*n
	c = a - (t*n)

	# if limit c from [0,2n) to [0,n)
	if c > n:
		c -= n
	
	assert c == a % n
	return c


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
	main()

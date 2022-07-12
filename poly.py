#!/usr/bin/env python3
# Date: 5/26/2022
# Create class for polynomials and ring polynomials
# 
# This program will create a polynomial class
# and different functions to perform operations
# on the different polynomials

class Poly(object):

	def __init__(self,poly=None):
		# initiliazed the polynomial to an array that is
		# passed in by the user. If nothing is passed then 
		# set the polynomial to 0
		self.poly = poly
		if (poly == None):
			self.poly = [0]
	
	def __getitem__(self, key):
		# returns the coefficient at given term
		# utilizes python's [] operator
		if key >= len( self ):
			return 0
		return self.poly[key]

	def __setitem__(self, key, value):
		# update the value of the term with 
		# python's [] operator
		self.poly[key] = value
		return

	def __iter__(self):
		# create an iter value so that for loops 
		# can work on the polynomial
		self.n = 0
		return self

	def __next__(self):
		# this method will help iterate through the 
		# polynomial with a for loop
		# go through the polynomial until it reaches
		# the end of the polynomial
		if self.n < self.size():
			res = self.poly[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration
	
	def __eq__(self,other):
		# this method will determine if two
		# polynomials are equal to each other
		if (type(other) != type(self)):
			return False
		
		if ( other == None ):
			return False

		se = self.copy()
		ot = other.copy()
		se.zero_deg()
		ot.zero_deg()

		if ( se.size() != ot.size() ):
			return False

		for i,j in zip(se,ot):
			if i != j:
				return False

		return True
	
	def next(self):
		# same function as __next__, was necessary for
		# Python 2 but is useless in Python 3
		if self.n < self.size():
			res = self.poly[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration

	def copy(self):
		# this function will create a new object and copy it
		copyarr = self.poly.copy()
		copy = Poly(copyarr)
		return copy

	def __add__(self,other):
		# add two polynomials together, new deg would 
		# be max(deg1,deg2)

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i + other
			return copy
		'''
		if (type(other) != type(self)):
			return NotImplemented
		'''

		sz = max(self.size(),other.size())
		res = [0] * sz
		for ind,i in enumerate(self.poly):
			res[ind] += i
		for ind,j in enumerate(other.poly):
			res[ind] += j

		return Poly(res)

	def __radd__(self,other):
		# add two polynomials together, new deg would 
		# be max(deg1,deg2)

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i + other
			return copy

		return NotImplemented


	def __sub__(self,other):
		# sub two polynomials together, new deg would 
		# be max(deg1,deg2)

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i - other
			return copy

		'''
		if (type(other) != type(self)):
			return NotImplemented
		'''

		sz = max(self.size(),other.size())
		res = [0] * sz
		for ind,i in enumerate(self.poly):
			res[ind] += i
		for ind,j in enumerate(other.poly):
			res[ind] -= j

		return Poly(res)

	def __mul__(self,other):
		# multiply two polynomials together

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i * other
			return copy

		'''
		if (type(other) != type(self)):
			return NotImplemented
		'''

		sz = self.deg() + other.deg() + 1
		res = [0] * sz

		# loop through the polynomials so that each 
		# term is multiplied through the other polynomial
		for ind,i in enumerate(self.poly):
			for jnd,j in enumerate(other.poly):
				res[ind+jnd] += (i*j)
		
		# use zero_deg to get rid of leading zeroes
		ret = Poly(res)
		ret.zero_deg()

		return ret

	def __rmul__(self,other):
		# multiply two polynomials together

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i * other
			return copy

		return NotImplemented

	def __truediv__(self,other):
		# div two polynomials together
		# return quotient and remainder
		# copy the polynomials so that the arguments
		# are not affected by this computation

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i / other
			return copy

		'''
		if (type(other) != type(self)):
			return NotImplemented
		'''

		se = self.copy()
		ot = other.copy()
		se.zero_deg()
		ot.zero_deg()
		quo = [0]*(se.deg()-ot.deg()+1)
		rem = [0]*(ot.deg())

		# if the divisor's degree is greater than the dividend's
		# then return the 0 polynomial and the dividend
		# as the divisor
		if ( ot.deg() > se.deg() ):
			copy = se.copy()
			return (Poly(),copy)

		copy = se.poly.copy()

		# set error if dividing by 0
		if ( ot[-1] == 0 and ot.size() == 1):
			raise ValueError("Cannot divide by 0")

		# use a for loop to find leading coefficient of quotient
		# then substract the dividend by the divisor*quotient
		# start from biggest term, going to smallest term
		for i in range(se.deg()-ot.deg(),-1,-1):
			coef = copy[ot.deg() + i] / ot[-1] 
			quo[i] = coef
			for j in range(ot.size()):
				copy[i+j] -= ot[j] * coef
		
		# add what remains from dividend to the remainder polynomial
		for i in range(ot.deg()):
			rem[i] += copy[i]

		Ret = Poly(rem)
		Ret.zero_deg()

		Quo = Poly(quo)
		Quo.zero_deg()

		return (Quo, Ret)
		#return (Poly(quo),Poly(rem))

	def __floordiv__(self,other):
		# div two polynomials together
		# return quotient and remainder
		# copy the polynomials so that the arguments
		# are not affected by this computation

		if (type(other) == int or type(other) == float):
			copy = self.copy()
			for ind,i in enumerate(copy):
				copy[ind] = i // other
			return copy

		'''
		if (type(other) != type(self)):
			return NotImplemented
		'''

		se = self.copy()
		ot = other.copy()
		se.zero_deg()
		ot.zero_deg()
		quo = [0]*(se.deg()-ot.deg()+1)
		rem = [0]*(ot.deg())

		# if the divisor's degree is greater than the dividend's
		# then return the 0 polynomial and the dividend
		# as the divisor
		if ( ot.deg() > se.deg() ):
			copy = se.copy()
			return (Poly(),copy)

		copy = se.poly.copy()

		# set error if dividing by 0
		if ( ot[-1] == 0 and ot.size() == 1):
			raise ValueError("Cannot divide by 0")

		# use a for loop to find leading coefficient of quotient
		# then substract the dividend by the divisor*quotient
		# start from biggest term, going to smallest term
		for i in range(se.deg()-ot.deg(),-1,-1):
			coef = copy[ot.deg() + i] // ot[-1] 
			quo[i] = coef
			for j in range(ot.size()):
				copy[i+j] -= ot[j] * coef
		
		# add what remains from dividend to the remainder polynomial
		for i in range(ot.deg()):
			rem[i] += copy[i]

		Ret = Poly(rem)
		Ret.zero_deg()

		Quo = Poly(quo)
		Quo.zero_deg()

		return (Quo, Ret)
		#return (Poly(quo),Poly(rem))

	def __mod__(self,mod):
		# mod the different coefficients of the polynomial
		copy = self.poly
		for ind,i in enumerate(copy):
			copy[ind] = i % mod

		return Poly(copy)

	def deg(self):
		# returns the highest degree of the polynomial
		# does not ignore leading 0's
		# e.g. (5 + 2*x + 0*x^2) would return 2
		return len(self.poly)-1

	def size(self):
		# returns the size of the polynomial
		# (the length of the list)
		return len(self.poly)

	def evaluate(self,x):
		# this will evaluate the polynomial given x
		ret = 0
		for ind,i in enumerate(self):
			ret += i * (x ** ind)

		return ret

	def __len__(self):
		# returns the size of the polynomial
		return len(self.poly)

	def polyprint(self):
		# print the list in the polynomial
		print(self.poly)
		return

	def __str__(self):
		# returns print version of polynomial
		return f'{self.poly}'

	def zero_deg(self):
		# this function will get rid of leading zeros in polynomial
		if (len(self.poly) < 2):
			return
		while ( self.poly[-1] == 0 and self.size() > 1):
			self.poly.pop(-1)
		return

	def floor(self):
		# this function will turn each of the coefficients into integers
		for ind,i in enumerate(self.poly):
			self.poly[ind] = int(i)
		return self

	def round(self):
		# this function will round each coefficient
		for ind,i in enumerate(self.poly):
			self.poly[ind] = round(i)
		return self

	def __round__(self):
		# this function will use python's round()
		cpy = self.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = round(i)
		return cpy

def main():
	# testing environment for the class poly
	X = Poly([4,1,11,10]) # 1 + 0*x + x^2 + 0*x^3 + x^4
	X.polyprint()

	Y = Poly([6,9,11,11,0]) # 1 + x + 0*x^2 + x^3
	Y.polyprint()

	# Z = X + Y
	Z = X * Y
	Z.polyprint()

	W = Poly([1,0,0,0,1]) # 1 + x^4
	W.polyprint()

	Q,R = Z / W
	Q.polyprint()
	R.polyprint()

	E = Poly([0,-1,1,1])

	A = R + E
	A = A % 13
	A.polyprint()


	print('\n\n')

	x = Poly([1,2,0,2,1])
	y = Poly([1,0,2])

	q,r = x / y

	print('x: ')#,end=' ')
	x.polyprint()
	print('y: ')#,end=' ')
	y.polyprint()
	print('q: ')#,end=' ')
	q.polyprint()
	print('r: ')#,end=' ')
	r.polyprint()

	print('\n\n')
	print('printing with iterators')
	for ind,i in enumerate(Z):
		print(ind,' ',i)

	return

def testing_copy():
	# testing the copy method in the class

	X = Poly([1,2,3])

	Y = X.copy()

	X.polyprint()
	Y.polyprint()

	X[0] = 10

	X.polyprint()
	Y.polyprint()

def test_equal():

	x = Poly([1,1,2,2])
	y = Poly([2,2,4,4])

	z = x + x

	print(f'X == Y: {x == y}')
	print(f'X == Z: {x == z}')
	print(f'Z == Y: {z == y}')

	print(f'X:',end=' ')
	x.polyprint()
	print(f'Y:',end=' ')
	y.polyprint()
	print(f'Z:',end=' ')
	z.polyprint()

def testing_add_int():

	x = Poly([0,1,2,3,4])

	#y = x * 5.5
	y = 2 * x

	#x.polyprint()
	#y.polyprint()
	print(f'x: {x}')
	print(f'y: {y}')

	return

def testing_eval():

	x = Poly([1,2,3,4])

	print( x.evaluate(1) )
	
	print( x.evaluate(3) )

	return

if __name__ == '__main__':
	#main()
	#testing_copy()
	#test_equal()
	#testing_add_int()
	testing_eval()
	pass

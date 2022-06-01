#!/usr/bin/env python3
# Author: Jeremy Stevens
# E-mail: jsteve22@nd.edu 
# Date: 5/26/2022
# Create class for polynomials and ring polynomials
# 
# This program will create a polynomial class
# and different functions to perform operations
# on the different polynomials

class Poly():

	def __init__(self,poly=None):
		self.poly = poly
		if (poly == None):
			self.poly = [0]
	
	def __getitem__(self, key):
		# returns the coefficient at given term
		# utilizes python's [] operator
		return self.poly[key]

	def __setitem__(self, key, value):
		# update the value of the term with 
		# python's [] operator
		self.poly[key] = value
		return

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if self.n < self.size():
			res = self.poly[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration
	
	def next(self):
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
		sz = max(self.size(),other.size())
		res = [0] * sz
		for ind,i in enumerate(self.poly):
			res[ind] += i
		for ind,j in enumerate(other.poly):
			res[ind] += j

		return Poly(res)

	def __sub__(self,other):
		# sub two polynomials together, new deg would 
		# be max(deg1,deg2)
		sz = max(self.size(),other.size())
		res = [0] * sz
		for ind,i in enumerate(self.poly):
			res[ind] += i
		for ind,j in enumerate(other.poly):
			res[ind] -= j

		return Poly(res)

	def __mul__(self,other):
		# mult two polynomials together
		#self.zero_deg()
		#other.zero_deg()
		sz = self.deg() + other.deg() + 1
		res = [0] * sz

		for ind,i in enumerate(self.poly):
			for jnd,j in enumerate(other.poly):
				res[ind+jnd] += (i*j)
		
		ret = Poly(res)
		ret.zero_deg()

		return ret

	def __truediv__(self,other):
		# div two polynomials together
		# return quotient and remainder
		#other.zero_deg()
		se = self.copy()
		ot = other.copy()
		se.zero_deg()
		ot.zero_deg()
		quo = [0]*(se.deg()-ot.deg()+1)
		rem = [0]*(ot.deg())

		if ( ot.deg() > se.deg() ):
			copy = se.copy()
			return (Poly(),copy)

		copy = se.poly.copy()

		if ( ot[-1] == 0 and ot.size() == 1):
			raise ValueError("Cannot divide by 0")

		for i in range(se.deg()-ot.deg(),-1,-1):
			coef = copy[ot.deg() + i] / ot[-1] 
			quo[i] = coef
			for j in range(ot.size()):
				copy[i+j] -= ot[j] * coef
		
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
		return len(self.poly)

	def polyprint(self):
		print(self.poly)
		return

	def zero_deg(self):
		# this function will get rid of leading zeros in polynomial
		if (len(self.poly) < 2):
			return
		while ( self.poly[-1] == 0 and self.size() > 1):
			self.poly.pop(-1)
		return

	def floor(self):
		for ind,i in enumerate(self.poly):
			self.poly[ind] = int(i)

		return self

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

	X = Poly([1,2,3])

	Y = X.copy()

	X.polyprint()
	Y.polyprint()

	X[0] = 10

	X.polyprint()
	Y.polyprint()

if __name__ == '__main__':
	main()
	#testing_copy()

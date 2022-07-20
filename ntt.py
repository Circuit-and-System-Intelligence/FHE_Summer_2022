#!/usr/bin/env python
# Date: 7/18/2022
# Create a function to perform NTT transform 
# and convolution multiplication

from poly import Poly

class NTT(object):

	def __init__(self,n):
		# define the size of the length of vector
		self.n = n 
		self.find_working_mod(649)
		self.find_generator()
		self.calc_prim_root()

		self.invn = self.extended_euclidean( self.n, self.N )
		self.invw = self.extended_euclidean( self.w, self.N )

	def find_working_mod(self, M):
		# find the working modulus for the
		# class. This will be a prime number
		# greater or equal to M

		self.k = 168
		self.N = 673

	def find_generator(self):
		# this will find a generator which
		# would help calculate a primitive 
		# root of unity
		self.g = 6

	def calc_prim_root(self):
		# this will calculate the primitive
		# root of unity needed to convert
		# into NTT
		self.w = (self.g ** self.k) % self.N
		self.w = 326

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

	def merge_NTT(self, p: Poly):
		# this will convert p into NTT

		if len(p) > self.n:
			raise ValueError("Input too large for NTT")

		psi = [0]*self.n

		for i in range(self.n):
			psi[i] = self.w ** self.bitrev( i )

		a = p.copy()
		# ret = Poly( ret.poly + ([0]*(self.n-len(p))) )

		'''
		m = 2
		while m <= self.n:
			wm = self.w ** ( self.n // m )
			w = 1
			for j in range(m//2):
				for k in range(0,self.n,m):
					t = w * ret[k+j+(m//2)]
					u = ret[k+j]
					ret[k+j] = u + t
					ret[k+j+(m//2)] = u - t 
				w = w * wm
			m = m * 2
		'''

		m = 1
		k = self.n // 2
		while m < self.n:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k - 1
				# wi = psi[ self.bitrev(m+i) ]
				wi = psi[ m+i ]
				for j in range(jFirst,jLast+1):
					# wrev = ( (self.w ** self.bitrev(m+i)) % self.N )
					l = j + k
					t = a[j]
					u = a[l] * wi
					a[j] = (t + u) % self.N
					a[l] = (t - u) % self.N

			m = m * 2
			k = k//2

		'''
		for i in range(self.n):
			for j in range(self.n):
				ret[i] += p[j] * (self.w ** (i*j))

			ret[i] = ret[i] % self.N
		'''

		return a 

	def NTT(self, p: Poly):
		# this will convert p to NTT

		if len(p) > self.n:
			raise ValueError("Input too large for NTT")

		ret = Poly([0]*self.n)

		for i in range(self.n):
			for j in range(self.n):
				ret[i] += p[j] * (self.w ** (i*j))

			ret[i] = ret[i] % self.N

		return ret

	def merge_iNTT(self, p: Poly):
		# convert inverse NTT

		if len(p) > self.n:
			raise ValueError("Input too large for iNTT")

		psi = [0]*self.n

		for i in range(self.n):
			psi[i] = self.invw ** self.bitrev(i)

		# ret = Poly([0]*self.n)
		a = p.copy()
		a = Poly( a.poly + ([0]*(self.n-len(p))) )

		m = self.n // 2
		k = 1
		while m > 1:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k - 1
				# wi = psi[ self.bitrev(m+i) ]
				wi = psi[ m+i ]
				for j in range(jFirst,jLast+1):
					l = j + k
					t = a[j]
					u = a[l]
					a[j] = ( t + u ) % self.N
					a[l] = (( t - u ) * wi) % self.N
			
			m = m // 2
			k = k * 2

		'''
		for i in range(self.n):
			for j in range(self.n):
				ret[i] += p[j] * (self.invw ** (i*j))

			ret[i] = ret[i] * self.invn
			ret[i] = ret[i] % self.N
		'''

		return a

	def iNTT(self, p: Poly):
		# convert inverse NTT

		if len(p) > self.n:
			raise ValueError("Input too large for iNTT")

		ret = Poly([0]*self.n)

		for i in range(self.n):
			for j in range(self.n):
				ret[i] += p[j] * (self.invw ** (i*j))

			ret[i] = ret[i] * self.invn
			ret[i] = ret[i] % self.N

		return ret

	def conv_mult(self, x: Poly, y: Poly):
		# compute the circular convolutional
		# multiplication of x and y

		z = Poly([0]*self.n)

		for i in range(self.n):
			z[i] = ( x[i] * y[i] ) % self.N

		return z 

	def bitrev(self, k: int):
		# return bit-reversed order of k
		
		revk = 0
		bit_len = (self.n-1).bit_length()

		for i in range( bit_len ):
			revk += (2 ** (bit_len-1-i) ) * ( ((2**i) & k ) >> i )

		return revk

def main():
	ntt = NTT(8)

	# a = Poly([6,0,10,7])
	a = Poly([1,2,3,4,5,6,7,8])
	# b = Poly([2,4,1,10])
	print(f'a: {a}')
	# print(f'b: {b}')
	print(' ')

	pre_a = a.copy()
	'''
	pre_a[0] *= (ntt.w) ** 0
	pre_a[1] *= (ntt.w) ** 1
	pre_a[2] *= (ntt.w) ** 2
	pre_a[3] *= (ntt.w) ** 3

	pre_a = pre_a % ntt.N
	'''

	n_a = ntt.NTT( pre_a )
	merge_a = ntt.merge_NTT( a )
	# nb = ntt.NTT( b )

	print(f'n_a:     {n_a}')
	print(f'merge_a: {merge_a}')
	# print(f'nb: {nb}')
	print(' ')

	ra = ntt.iNTT( n_a )
	mra = ntt.merge_iNTT( merge_a )

	# rb = ntt.iNTT( nb )

	print(f'ra: {ra}')
	print(f'mra: {mra}')
	# print(f'rb: {rb}')

def mult():
	ntt = NTT(8)

	# a = Poly([6,10,3,9])
	# b = Poly([4,4,5,8])
	# a = Poly([12,53,81,17])
	# b = Poly([14,8,23,10])
	a = Poly([13,15,19,20])
	b = Poly([4,8,10,11])
	print(f'a: {a}')
	print(f'b: {b}')
	print(' ')
	ai = Poly( a.poly + ( -1*a ).poly )
	bi = Poly( b.poly + ( -1*b ).poly )

	na = ntt.NTT( ai )
	nb = ntt.NTT( bi )

	nc = ntt.conv_mult( na, nb )

	c = ntt.iNTT( nc )

	c = c // 2

	print( c )

	c = Poly(c.poly[0:4])

	print(f'c: {c}')

	x = a * b
	quo,x = x // Poly([1,0,0,0,1])
	x = x % ntt.N
	print(f'x: {x}')

	# print(f'x: {x}')


if __name__ == '__main__':
	main()
	# mult()

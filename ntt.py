#!/usr/bin/env python
# Date: 7/18/2022
# Create a function to perform NTT transform 
# and convolution multiplication

from poly import Poly
import pickle as pkl
from random import randint

class NTT(object):

	def __init__(self,n,M=2**5):
		# define the size of the length of vector
		self.n = n 
		self.find_working_mod(M)
		self.find_generator()
		self.calc_prim_root()

		self.invn = self.extended_euclidean( self.n, self.N )
		self.invpsi = self.extended_euclidean( self.psi, self.N )
		self.invw = self.extended_euclidean( self.w, self.N )

	def find_working_mod(self, M):
		# find the working modulus for the
		# class. This will be a prime number
		# greater or equal to M

		# load prime numbers
		primes = []
		with open("prime.pickle",'rb') as f:
			primes = pkl.load( f )

		for ind, p in enumerate(primes):
			if p >= M:
				if (p-1)%(2*self.n) == 0:
					self.k = (p-1)//(2*self.n)
					self.N = p
					return

		raise AttributeError("No Prime for Modulus Found")
		self.k = 84
		self.N = 673

		# self.N = 536608769

	def find_generator(self):
		# this will find a generator which
		# would help calculate a primitive 
		# root of unity
		
		# load prime numbers
		primes = []
		with open("prime.pickle",'rb') as f:
			primes = pkl.load( f )

		# find factors of N-1
		factors = []
		sqrt_N = (self.N-1) ** 0.5
		for p in primes:
			if p > (sqrt_N):
				break

			if (self.N-1) % p == 0:
				factors.append( p )

		# find a generator
		for i in range(1,self.N-1):
			skip = False

			for f in factors:
				a = i ** ((self.N-1)//f)
				assert ((self.N-1)//f) == ((self.N-1)/f)
				
				if a % self.N == 1:
					skip = True

			if skip:
				continue

			self.g = i
			return
			# print( self.g )
			
		self.g = 653
		return

	def calc_prim_root(self):
		# this will calculate the primitive
		# root of unity needed to convert
		# into NTT
		self.psi = (self.g ** self.k) % self.N
		self.w = (self.g ** (self.k*2)) % self.N

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

	def inverse_mod(self, a, n):
		# this function will find the inverse of a such
		# that t*a === 1 mod n
		# 
		# n must be a prime number
		
		p = n

		t = (a**(p-2))%p
		return t

	def merge_NTT(self, p: Poly):
		# this will convert p into NTT

		if len(p) > self.n:
			raise ValueError("Input too large for NTT")

		psi = [0]*self.n

		for i in range(self.n):
			# q=536608769, n=2048
			# psi[i] = (284166 ** self.bitrev( i )) % self.N

			# q=673, n=8
			# psi[i] = (8 ** self.bitrev( i )) % self.N

			psi[i] = (self.psi ** self.bitrev(i)) % self.N

		a = p.copy()
		# ret = Poly( ret.poly + ([0]*(self.n-len(p))) )

		m = 1
		k = self.n // 2
		while m < self.n:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k
				# wi = psi[ self.bitrev(m+i) ]
				wi = psi[ m+i ]
				for j in range(jFirst,jLast):
					# wrev = ( (self.w ** self.bitrev(m+i)) % self.N )
					l = j + k
					t = a[j]
					u = a[l] * wi
					a[j] = (t + u) % self.N
					a[l] = (t - u) % self.N

			m = m * 2
			k = k//2

		return a 

	def merge_iNTT(self, p: Poly):
		# convert inverse NTT

		if len(p) > self.n:
			raise ValueError("Input too large for iNTT")

		invpsi = [0]*self.n

		for i in range(self.n):
			# q=536608769, n=2048
			# invpsi[i] = (208001377 ** self.bitrev(i)) % self.N

			# q=673, n=8
			# invpsi[i] = ( 589 ** self.bitrev(i)) % self.N

			invpsi[i] = (self.invpsi ** self.bitrev(i)) % self.N

		# ret = Poly([0]*self.n)
		a = p.copy()
		# a = Poly( a.poly + ([0]*(self.n-len(p))) )

		m = self.n // 2
		k = 1
		while m >= 1:
			for i in range(m):
				jFirst = 2 * i * k
				jLast = jFirst + k
				# wi = psi[ self.bitrev(m+i) ]
				wi = invpsi[ m+i ]
				for j in range(jFirst,jLast):
					l = j + k
					t = a[j]
					u = a[l]
					a[j] = ( t + u ) % self.N
					a[l] = (( t - u ) * wi) % self.N
			
			m = m // 2
			k = k * 2

		for i in range(self.n):
			a[i] = a[i] * self.invn
			a[i] = a[i] % self.N

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
	sz = 2**10
	ntt = NTT(sz,2**20)

	# a = Poly([6,0,10,7])

	arr = []
	brr = []
	for i in range(sz):
		arr.append( randint(0,ntt.N) )
		brr.append( randint(0,ntt.N) )

	a = Poly([1,2,3,4,5,6,7,8]+([0]*2040))
	b = Poly([1,1,1,1,1,1,1,1]+([0]*2039)+[4])
	a = Poly( arr )
	b = Poly( brr )
	# a = Poly([1,2,3,4,5,6,7,8])


	# b = Poly([2,1,3,4,1,3,5,2])
	print(f'a: {a}')
	print(' ')

	pre_a = a.copy()
	'''
	pre_a[0] *= (ntt.w) ** 0
	pre_a[1] *= (ntt.w) ** 1
	pre_a[2] *= (ntt.w) ** 2
	pre_a[3] *= (ntt.w) ** 3
	pre_a[4] *= (ntt.w) ** 4
	pre_a[5] *= (ntt.w) ** 5
	pre_a[6] *= (ntt.w) ** 6
	pre_a[7] *= (ntt.w) ** 7
	'''
	pre_a = pre_a % ntt.N

	# n_a = ntt.NTT( pre_a )
	merge_a = ntt.merge_NTT( a )
	merge_b = ntt.merge_NTT( b )
	merge_c = ntt.conv_mult( merge_a, merge_b )
	# nb = ntt.NTT( b )

	# print(f'n_a:     {n_a}')
	print(f'merge_a: {merge_a}')
	print(' ')

	# ra = ntt.iNTT( n_a )
	mra = ntt.merge_iNTT( merge_a )
	mrc = ntt.merge_iNTT( merge_c )

	# print(f'ra: {ra}')
	print(f'mra: {mra}')
	# print(f'mrc: {mrc}')

	print(f'mra==a: {mra==a}')

	# polynomial multiplication
	c = a * b
	fn = Poly( [1] + ([0]*(sz-1)) + [1] )
	quo,c = c // fn
	c = c % ntt.N
	# print(c)

	print(f'mrc==c: {mrc==c}')
	

def test_addition():
	sz = 2**5
	ntt = NTT(sz,2**15)

	# a = Poly([6,0,10,7])

	arr = []
	brr = []
	for i in range(sz):
		arr.append( randint(0,ntt.N) )
		brr.append( randint(0,ntt.N) )

	a = Poly( arr )
	b = Poly( brr )

	merge_a = ntt.merge_NTT( a )
	merge_b = ntt.merge_NTT( b )

	merge_c = merge_a + merge_b

	mc = ntt.merge_iNTT( merge_c )

	c = a + b
	c = c % ntt.N

	print(f'mc: {mc}')
	print(f'c:  {c}')
	print(f'c==mc: {c==mc}')

def mult():
	sz = 2**2
	ntt = NTT(sz,2**8)

	arr = []
	brr = []
	for i in range(sz):
		arr.append( randint(0,ntt.N//2) )
		brr.append( randint(0,ntt.N) )

	a = Poly( arr )
	a = a * 2
	b = Poly( brr )

	merge_a = ntt.merge_NTT( a )

	half = ntt.extended_euclidean( 2, ntt.N )

	merge_c = merge_a * half
	merge_c = merge_c % ntt.N

	mc = ntt.merge_iNTT( merge_c )

	ma = a // 2
	ma = ma % ntt.N

	print(f'a: {a}')
	print(f'ma: {ma}')
	print(f'mc: {mc}')
	print(f'ma==mc: {ma==mc}')


def ntt_circuit():
	# this function will act as a translation
	# of the circuit defined in paper

	# x = [1,2,3,4,5,6,7,8]
	x = [1,1,1,1,2,2,2,2]

	w = 64
	phi = 8
	invw = 326
	invphi = 589

	q = 673


	# stage 0
	for i in range(4,8):
		x[i] *= (phi ** 4)
		x[i] = x[i] % q

	y = x.copy()
	for i in range(0,4):
		x[i] = y[i] + y[i+4]
		x[i] = x[i] % q

	for i in range(0,4):
		x[i+4] = y[i+4] - y[i]
		x[i+4] = x[i+4] % q

	# stage 1
	for i in range(2):
		x[i+2] *= (phi ** 2)
		x[i+6] *= (phi ** 2) * (w ** 2)
		x[i+2] = x[i+2] % q
		x[i+6] = x[i+6] % q

	y = x.copy()

	for i in range(2):
		x[i]   = y[i] + y[i+2]
		x[i+2] = y[i+2] - y[i]
		x[i+4] = y[i+6] + y[i+4]
		x[i+6] = y[i+6] - y[i+4]
		x[i]   = x[i] % q
		x[i+2] = x[i+2] % q
		x[i+4] = x[i+4] % q
		x[i+6] = x[i+6] % q

	# stage 2
	x[1] *= (phi)*(w**0)
	x[3] *= (phi)*(w**2)
	x[5] *= (phi)*(w**1)
	x[7] *= (phi)*(w**3)
	x[1] = x[1] % q
	x[3] = x[3] % q
	x[5] = x[5] % q
	x[7] = x[7] % q

	y = x.copy()
	
	for i in range(0,8,2):
		x[i] = y[i] + y[i+1]
		x[i+1] = y[i+1] - x[i]
		x[i] = x[i] % q
		x[i+1] = x[i+1] % q
	
	print(f'after ntt: {x}')

	y = x.copy()
	# stage 0
	for i in range(0,8,2):
		x[i] = y[i] + y[i+1]
		x[i+1] = y[i+1] - x[i]
		x[i] = x[i] % q
		x[i+1] = x[i+1] % q

	x[1] *= (invphi)*(invw**0)
	x[3] *= (invphi)*(invw**2)
	x[5] *= (invphi)*(invw**1)
	x[7] *= (invphi)*(invw**3)
	x[1] = x[1] % q
	x[3] = x[3] % q
	x[5] = x[5] % q
	x[7] = x[7] % q

	# stage 1

	y = x.copy()
	for i in range(2):
		x[i]   = y[i] + y[i+2]
		x[i+2] = y[i+2] - y[i]
		x[i+4] = y[i+6] + y[i+4]
		x[i+6] = y[i+6] - y[i+4]
		x[i]   = x[i] % q
		x[i+2] = x[i+2] % q
		x[i+4] = x[i+4] % q
		x[i+6] = x[i+6] % q
	
	for i in range(2):
		x[i+2] *= (invphi ** 2)
		x[i+6] *= (invphi ** 2) * (invw ** 2)
		x[i+2] = x[i+2] % q
		x[i+6] = x[i+6] % q
	
	# stage 2

	y = x.copy()
	for i in range(0,4):
		x[i] = y[i] + y[i+4]
		x[i] = x[i] % q

	for i in range(0,4):
		x[i+4] = y[i+4] - y[i]
		x[i+4] = x[i+4] % q

	for i in range(4,8):
		x[i] *= (invphi ** 4)
		x[i] = x[i] % q

	print(f'after intt: {x}')

if __name__ == '__main__':
	# main()
	# test_addition()
	# ntt_circuit()
	mult()

	pass

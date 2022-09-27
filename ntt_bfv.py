#!/usr/bin/env python
# this program will be an attempt to create the bfv
# encrption scheme that is described in the BFV paper.

import numpy as np
from numpy.polynomial import polynomial as p
import random
import sys
from ntt import NTT
from rns import RNS

import pdb

from counter import OperationsCounter
#from counter import PolyCount as Poly
from poly import Poly
from bitint import Bitint

def main():
	# this main function is using the lpr() class
	# encryption scheme

	# create an encryption scheme instance
	lpr = LPR()

	# create plaintext you want to encrypt
	pt = 5

	# encrypt plaintext into ciphertext
	ct = lpr.encrypt(pt)

	# decrypt back into plaintext
	recovered_pt = lpr.decrypt(ct)

	# print results
	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')
	print( lpr.opcount )

	return

class NTT_BFV():

	def __init__(self,q=2**15,t=2,n=2**4,std=2,fn=None,h=64,security=128,bitwidth=32):
		"""
		this init method will initialize the important variables
		needed for this encryption scheme, including keys

		----- Variables -----
		q 	- Cipher Text Modulus
		t 	- Plain Text Modulus
		n 	- Degree of Ring Polynomial
		std - standard deviation for Gaussian distribution
		fn 	- Cyclotomic Polynomial from n
		
		----- Keys -----
		sk 	- secret key with hamming weight h=64
		pk 	- public key for encrypting messages
		rlk - relinearization key for reducing ciphertext size after multiplication 
		"""

		self.ntt = NTT(n,q)
		
		'''
		# n=2**5, q=2**15
		self.ntt.psi = 3213
		self.ntt.invpsi = 3145
		'''

		'''
		# n=2**10, q=2**15
		self.ntt.psi = 20237
		self.ntt.invpsi = 16233
		'''

		'''
		# n=2**10, q=2**10
		self.ntt.psi = 1945
		self.ntt.invpsi = 4050
		'''

		# n=2**5, q=2**15
		self.ntt.N = 32833
		self.ntt.psi = 4256
		self.ntt.invpsi = 18754

		'''
		# n=2**10, q>2**20
		self.ntt.N = 1054721
		self.ntt.psi = 794881
		self.ntt.invpsi = 961203
		'''

		self.q = self.ntt.N
		self.t = t
		self.n = n
		self.std = std
		self.h = h
		self.sec = security
		# this will set the polynomial for the ring, if not declared then will be
		# the polynomial 1 + x^n
		self.fn = fn
		if (self.fn == None):
			self.fn = [1] + [0]*((n)-1) + [1]
		self.fn = Poly(self.fn)

		# create operation counters for different methods in BFV
		self.gen_op_counters(bitwidth)

		# this will set the keys as none, but will then immediately generate
		# a public key, a private key, and a relinearization key
		self.sk = None
		self.pk = None
		self.rlk = None
		self.gen_keys()


	def gen_keys(self):
		"""
		Generate keys, generate secret key first then public 
		and relinearization	keys afterwards
		"""
		self.gen_sk()
		self.gen_pk()
		self.gen_rlk()
		
	def gen_sk(self):
		"""
		call the gen_binary_poly key to create a polynomial
		of only 0's and 1's for the secret key
		"""
		# self.sk = self.gen_binary_poly()
		# self.sk = self.gen_normal_poly()

		# set counter object for keys
		oc = self.counters['key']
		
		# set hamming weight of secret key
		sk = [1]*self.h
		sk = sk + [0]*(self.n-self.h)
		'''
		sk = [Bitint(1,32)]*self.h
		sk = sk + [Bitint(0,32)]*(self.n-self.h)
		'''
		np.random.shuffle( sk )
		self.sk = Poly( sk )
		self.sk = oc.merge_NTT( self.ntt, self.sk )
		# self.sk = self.ntt.merge_NTT( self.sk )
	
		return

	def gen_pk(self):
		"""
		Generate public key from secret key and random polynomials.

		----- Random Polynomials -----
		a <- Uniform Distribution over q
		e <- Normal Distribution with (mean = 0, standard deviation = self.std)

		----- Calculation ----- 
		a = -(a*sk + e)

		----- Result -----
		pk = (b , a)

		"""
		if (self.sk == None):
			return
		# set counter object for keys
		oc = self.counters['key']

		# generate a uniformly distributed polynomial with coefs
		# from [0,q)
		a = self.gen_uniform_poly()

		# generate a normally distributed polynomial with integers
		# generated from a center of 0 and std of 2
		e = self.gen_normal_poly()

		one = self.ntt.extended_euclidean( self.ntt.n, self.ntt.N )
		neg_one = one * -1
		neg_one = neg_one % self.ntt.N
		neg_one = -1

		# create a new polynomial _a which is -a
		_a = oc.poly_mul_num(a, neg_one) 
		#_a = a * -1

		# then set e = -e
		e = oc.poly_mul_num(e, neg_one) 
		#e = e * -1

		# breakpoint()
		# create b from (-a * sk) - e
		b = self.polyadd( self.polymult(_a, self.sk, oc),e, oc)

		# set the public key to the tuple (b,a)
		# or (-[a*sk + e], a)
		self.pk = (b,a)
		return
	
	def gen_rlk(self):
		"""
		Generate a relinearization key for ciphertext multiplication

		----- Random Polynomials -----
		ai <- Uniformat Distribution over q
		ei <- Normal Distribution with (mean = 0, standard deviation = self.std )

		----- Calculation -----
		T = 2
		bi = -(ai*sk + ei) + (T^i)*(sk^2)

		----- Result -----
		rlk = [(bi , ai)]

		"""
		# set counter object for keys
		oc = self.counters['key']

		# define T, set T == 2
		self.T = 2
		T = self.T

		# define l, l = logT(q)
		self.l = int(np.log(self.q)//np.log(T))+1
		l = self.l

		# this will hold different masked versions
		# of the keys with T^i for i=[0,l)
		self.rlk_V1 = []

		one = self.ntt.extended_euclidean( self.ntt.n, self.ntt.N )
		neg_one = one * -1
		neg_one = neg_one % self.ntt.N
		neg_one = -1

		for i in range(l):
			ei = self.gen_normal_poly()
			ai = self.gen_uniform_poly()

			# T^i * sk^2
			Ti = T ** i
			ss = oc.dotProduct(self.sk, self.sk)
			ss = self.ntt.merge_iNTT( ss )
			ss = ss * Ti
			ss = ss % self.q
			ss = self.ntt.merge_NTT( ss )

			# -(ai*sk + ei)
			b = oc.dotProduct(ai, self.sk)
			b = b + ei
			b = b * neg_one

			# b = -(ai*sk + ei) + (T^i)*(sk^2)
			b = b + ss
			b = b % self.q
			
			self.rlk_V1.append( (b,ai) )

	def gen_op_counters(self,bitwidth=32):
		"""
		this function will generate operation counters
		for encrypting, decrypting, key generation, 
		addition, multiplication, and relinearization
		"""

		# create operations counter object
		self.opcount = OperationsCounter(bitwidth)
		self.counters = {}
		self.counters['enc'] = OperationsCounter(bitwidth)
		self.counters['dec'] = OperationsCounter(bitwidth)
		self.counters['key'] = OperationsCounter(bitwidth)
		self.counters['add'] = OperationsCounter(bitwidth)
		self.counters['mul'] = OperationsCounter(bitwidth)
		self.counters['relin'] = OperationsCounter(bitwidth)
	
	def encrypt(self,pt=0):
		"""
		encode plaintext integer into a plaintext polynomial
		and then into ciphertext polynomials

		----- Arguments -----
		pt	- Plain Text Integer

		----- Random Polynomials -----
		u 	<- Binary Distribution
		e1	<- Normal Distribution with (mean = 0, standard deviation = self.std)
		e2	<- Normal Distribution with (mean = 0, standard deviation = self.std)

		----- Calculation -----
		m = pt * (q / t)
		c0 = pk[0]*u + e1 + m
		c1 = pk[1]*u + e2
		ct = (c0, c1)

		----- Output -----
		ct	- Ciphertext of plain text
	
		"""

		# set encryption counter object 
		oc = self.counters['enc']

		m = pt
		if ( type(pt) == int ):
			m = [pt]
			m = Poly(m)

		# m = Poly(m)
		m = oc.poly_mod( m, self.q ) #m = m % self.q
		#print( m )

		delta = oc.floor_div(self.q, self.t) 
		#delta = self.q // self.t
		scaled_m = m.copy()
		#scaled_m[0] = delta * scaled_m[0] % self.q
		scaled_m = oc.poly_mul_num( scaled_m, delta ) 
		#scaled_m = (scaled_m * delta) 
		scaled_m = oc.poly_mod( scaled_m, self.q ) 
		# scaled_m = self.ntt.merge_NTT( scaled_m )
		scaled_m = oc.merge_NTT( self.ntt, scaled_m )
		#scaled_m = scaled_m  % self.q
		# create a new m, which is scaled my q//t % q

		# generated new error polynomials
		e1 = self.gen_normal_poly()
		e2 = self.gen_normal_poly()
		u = self.gen_binary_poly()

		'''
		# set counters for each polynomial
		e1.oc = oc
		e2.oc = oc
		u.oc = oc
		self.pk[0].oc = oc
		self.pk[1].oc = oc
		'''

		# create c0 = pk[0]*u + e1 + scaled_m
		ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u, oc), e1, oc), scaled_m, oc)
		# ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u), e1), scaled_m)
		'''
		ct0 = self.pk[0] * u
		ct0 = ct0 + e1
		ct0 = ct0 + scaled_m
		quo,ct0 = ct0 // self.fn
		ct0 = ct0 % self.q
		'''

		# create c1 = pk[1]*u + e2
		ct1 = self.polyadd( self.polymult( self.pk[1], u, oc), e2, oc)
		#ct1 = self.polyadd( self.polymult( self.pk[1], u), e2)

		return (ct0, ct1)

	def decrypt(self,ct):
		"""
		decrypt the cipher text to get the plaintext equivalent

		----- Arguments -----
		ct	- Ciphertext

		----- Calculations -----
		m = ct[0] + ( ct[1] * sk )
		pt = m * (t / q)

		----- Output -----
		pt	- Plaintext integer of ciphertext

		"""

		# set decryption counter object
		oc = self.counters['dec']
		'''
		ct[0].oc = oc
		ct[1].oc = oc
		self.sk.oc = oc
		'''

		# scaled_pt = ct[1]*sk + ct[0]
		#scaled_pt = self.polyadd( self.polymult( ct[1], self.sk ), ct[0] )
		scaled_pt = self.polyadd( oc.dotProduct( ct[1] , self.sk ), ct[0], oc )
		# scaled_pt = self.ntt.merge_iNTT( scaled_pt )
		scaled_pt = oc.merge_iNTT( self.ntt, scaled_pt )
		# print( scaled_pt )

		tq = oc.true_div( self.t, self.q )
		scaled_pt = oc.poly_mul_num( scaled_pt, tq ) 
		# scaled_pt = scaled_pt * ( self.t / self.q)
		# scaled_pt = scaled_pt * self.t
		# scaled_pt = scaled_pt // self.q
		scaled_pt.round()
		scaled_pt = oc.poly_mod( scaled_pt, self.t ) 
		# scaled_pt = scaled_pt % self.t
		# print( scaled_pt )
		decrypted_pt = scaled_pt

		# return the first term of the polynomial, which is the plaintext
		return decrypted_pt
		#return int(decrypted_pt[0])

	def ctadd(self, x, y):
		"""
		Add two ciphertexts and return a ciphertext which
		should decrypt as the addition of plaintext inputs

		----- Arguments -----
		x	- Ciphertext for addition
		y	- Ciphertext for addition

		----- Calculation -----
		ct = x + y

		----- Output -----
		ct	- Ciphertext equivalent to x+y

		"""

		# set adder counter object
		oc = self.counters['add']

		ct0 = self.polyadd(x[0],y[0],oc)
		ct1 = self.polyadd(x[1],y[1],oc)

		ct = (ct0,ct1)

		return ct

	def ctmult(self, x, y):
		"""
		Add two ciphertexts and return a ciphertext which
		should decrypt as the multiplication of plaintext inputs.
		Multiplying two ciphertexts increase elements in ciphertext
		by one, so must relinearize ciphertext back to two elements

		----- Arguments -----
		x	- Ciphertext for multiplication
		y	- Ciphertext for multiplication

		----- Calculation -----
		c0 = x[0] * y[0]
		c1 = x[0] * y[1] + x[1] * y[0]
		c2 = x[1] * y[1]
		ct = relin( c0, c1, c2 )

		----- Output -----
		ct	- Ciphertext equivalent to x*y

		"""

		# set multiplier counter object
		oc = self.counters['mul']

		t = self.t

		i = [None,None]
		j = [None,None]
		i[0] = oc.merge_iNTT( self.ntt, x[0] )
		i[1] = oc.merge_iNTT( self.ntt, x[1] )
		j[0] = oc.merge_iNTT( self.ntt, y[0] )
		j[1] = oc.merge_iNTT( self.ntt, y[1] )

		'''
		i[0] = self.ntt.merge_iNTT( x[0] )
		i[1] = self.ntt.merge_iNTT( x[1] )
		j[0] = self.ntt.merge_iNTT( y[0] )
		j[1] = self.ntt.merge_iNTT( y[1] )
		'''

		# c0 = ct0[0]*ct1[0]
		# c0 = oc.dotProduct( x[0], y[0] ) #c0 = x[0] * y[0]
		# c0 = oc.poly_mod( c0, self.q ) #c0 = c0 % self.q

		d0 = oc.poly_mul_poly(i[0], j[0])
		quo, d0 = oc.poly_div_poly(d0, self.fn)
		d0 = oc.poly_mul_num(d0, self.t)
		d0 = oc.poly_div_num(d0, self.q)
		d0.round()
		d0 = oc.poly_mod(d0, self.q)

		'''
		print(f'c0: {self.ntt.merge_iNTT(c0)}')
		print(f'd0: {d0}')
		assert self.ntt.merge_iNTT(c0) == d0
		'''

		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		ca = oc.dotProduct( x[0], y[1] )
		cb = oc.dotProduct( x[1], y[0] )
		c1 = oc.poly_add_poly( ca, cb ) #c1 = (x[0]*y[1]) + (x[1]*y[0])
		c1 = self.ntt.merge_NTT( c1 )

		c1 = oc.poly_mod( c1, self.q ) #c1 = c1 % self.q

		da = oc.poly_mul_poly(i[0], j[1])
		db = oc.poly_mul_poly(i[1], j[0])
		d1 = oc.poly_add_poly(da, db)
		quo, d1 = oc.poly_div_poly(d1, self.fn)
		d1 = oc.poly_mul_num(d1, self.t)
		d1 = oc.poly_div_num(d1, self.q)
		d1.round()
		d1 = oc.poly_mod(d1, self.q)

		# c2 = ct0[1]*ct1[1]
		c2 = oc.dotProduct( x[1], y[1] ) #c2 = x[1] * y[1]
		c2 = oc.poly_mod( c2, self.q ) #c2 = c2 % self.q

		d2 = oc.poly_mul_poly(i[1], j[1])
		quo, d2 = oc.poly_div_poly(d2, self.fn)
		d2 = oc.poly_mul_num(d2, self.t)
		d2 = oc.poly_div_num(d2, self.q)
		d2.round()
		d2 = oc.poly_mod(d2, self.q)

		d0 = oc.merge_NTT(self.ntt, d0 )
		d1 = oc.merge_NTT(self.ntt, d1 )
		d2 = oc.merge_NTT(self.ntt, d2 )

		ret = self.relin(d0,d1,d2)

		r0 = d0
		r1 = oc.dotProduct( d1, self.sk )
		r2 = oc.dotProduct( d2, oc.dotProduct( self.sk, self.sk ) )

		r = r0 + r1 + r2
		r = self.ntt.merge_iNTT( r )
		r = r * self.t
		r = r / self.q
		r.round()
		r = r % self.t
		# print(f'{r}')

		return ret

	def relin(self,c0,c1,c2):
		"""
		Take a ciphertext with 3 polynomials and perform a relinearization
		with the relin key to return a ciphertext of 2 polynomials

		----- Arguments -----
		c0	- Polynomial from ctmult
		c1	- Polynomial from ctmult (sk)
		c2	- Polynomial from ctmult (sk^2)

		----- Calculation -----
		c2i = basechange( c2, T )[i]
		c20	= ( c2i * rlk_V1[i][0] ) 
		c21 = ( c2i * rlk_V1[i][1] ) 

		c0' = c0 + c20
		c1' = c1 + c21
		ct' = (c0',c1')

		----- Output -----
		ct'	- Ciphertext relinearized from 3 elements to 2 elements

		"""
		# set relin counter object
		oc = self.counters['relin']

		# c2 = self.ntt.merge_iNTT( c2 )
		c2 = oc.merge_iNTT( self.ntt, c2 )
		c2i = []

		for i in range(self.l):
			mask = self.T ** i

			ci = []
			for j in range(self.n):
				ci.append( (c2[j] & mask) >> i )

			ci = Poly( ci )
			ci = oc.merge_NTT( self.ntt, ci )
			c2i.append( ci )

		c20 = Poly()
		c21 = Poly()

		for i in range(self.l):
			x = oc.dotProduct(self.rlk_V1[i][0], c2i[i])
			
			y = oc.dotProduct(self.rlk_V1[i][1], c2i[i])

			c20 = oc.poly_add_poly(c20, x)
			c21 = oc.poly_add_poly(c21, y)

		# c20 = c20 % self.q
		# c21 = c21 % self.q

		_c0 = oc.poly_add_poly(c0, c20)
		_c1 = oc.poly_add_poly(c1, c21)

		_c0 = oc.poly_mod(_c0, self.q)
		_c1 = oc.poly_mod(_c1, self.q)

		return (_c0, _c1)

	def mod(self,poly):
		"""	
		calculate the modulus of poly by q
		with answer given back in range (-q/2,q/2]
		"""
		copy = poly.poly.copy()
		for ind,i in enumerate(copy):
			i = i % self.q
			if ( i > (self.q/2) ):
				copy[ind] = i - self.q

		return Poly(copy)

	def polyadd(self, x, y, oc=None):
		"""
		add two polynomials together and keep them 
		within the polynomial ring
		"""
		if ( oc == None ):
			oc = self.opcount
		z = oc.poly_add_poly( x, y ) 
		#z = x + y

		z = oc.poly_mod( z, self.q ) 
		#z = z % self.q

		return z

	def polymult(self, x, y, oc=None):
		"""
		multiply two polynomials together and keep them 
		within the polynomial ring
		"""
		if ( oc == None ):
			oc = self.opcount

		z = oc.dotProduct( x, y ) 

		z = oc.poly_mod( z, self.q ) 
		return z

	def gen_normal_poly(self,c=0,std=None):
		"""
		generate a random polynomial of degree n-1
		with coefficients selected from normal distribution
		with center at 0 and std of 2. Each term is rounded
		down to nearest integer
		"""
		if ( std == None ):
			std = self.std
		a = []
		for i in range(self.n):
			a.append( int(np.random.normal(c,std)) )
		a = Poly(a)
		a = a % self.q
		a = self.ntt.merge_NTT( a )
		return a

	def gen_binary_poly(self):
		"""
		generate a random polynomial of degree n-1
		with coefficients ranging from [0,1]
		"""
		a = []
		for i in range(self.n):
			a.append( np.random.randint(0,2) )
		a = Poly(a)
		a = a % self.q
		a = self.ntt.merge_NTT( a )
		return a

	def gen_uniform_poly(self,q=None):
		"""
		generate a random polynomial of degree n-1
		with coefficients ranging from [0,q)
		"""
		if (q == None):
			q = self.q
		a = []
		for i in range(self.n):
			a.append( random.randint(0,q) )
		a = Poly(a)
		a = a % self.q
		a = self.ntt.merge_NTT( a )
		return a

	def print_counter_info(self):
		"""
		this function will print out the operation
		costs for each function (enc,dec,ctadd,ctmult...)
		"""

		print('Encryption OpCount')
		print( self.counters['enc'] )
		print('\nDecryption OpCount')
		print( self.counters['dec'] )
		print('\nKeyGen OpCount')
		print( self.counters['key'] )
		print('\nAdd OpCount')
		print( self.counters['add'] )
		print('\nMul OpCount')
		print( self.counters['mul'] )
		print('\nRelin OpCount')
		print( self.counters['relin'] )
		
if __name__ == '__main__':
	main()
	pass

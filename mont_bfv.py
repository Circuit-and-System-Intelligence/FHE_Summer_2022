#!/usr/bin/env python
# this program will be an attempt to create the BFV
# encrption scheme that is described in the BFV paper.
#
# this will implement Montgomery reduction to decrease 
# operations within modulus space

import numpy as np
import random

from naive_modulus import Montgomery
from counter import OperationsCounter
from poly import Poly

class Mont_BFV():

	def __init__(self,q=2**15,t=2**8,n=2**4,std=2,fn=None,h=64,security=128,bitwidth=32):
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

		self.mont = Montgomery( (q-1)**3, (q**3).bit_length() )

		self.q = q
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
		
		# set hamming weight of secret key
		sk = [1]*self.h
		sk = sk + [0]*(self.n-self.h)
		np.random.shuffle( sk )
		self.sk = Poly( sk )

		for ind, n in enumerate(self.sk):
			self.sk[ind] = self.mont.toMont( n )
	
		return

	def gen_pk(self):
		"""
		Generate public key from secret key and random polynomials.

		----- Random Polynomials -----
		a <- Uniform Distribution over q
		e <- Normal Distribution with (mean = 0, standard deviation = self.std)

		----- Calculation ----- 
		b = -(a*sk + e)

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

		# create a new polynomial _a which is -a
		_a = oc.poly_mul_num(a, -1) #_a = a * -1

		# then set e = -e
		e = oc.poly_mul_num(e, -1) #e = e * -1

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
		a <- Uniformat Distribution over q*p
		e <- Normal Distribution with (mean = 0, standard deviation = sigma_prime )

		----- Calculation -----
		p = q ** 3
		sigma_prime = (alpha ** (1-sqrt(k))) * (q ** (k-sqrt(k))) * (B ** sqrt(k))
		b = -(a*sk + e) + p*(sk^2)

		----- Result -----
		rlk = (b , a)

		"""

		# set counter object for keys
		oc = self.counters['key']

		# bigger p means less noise (I think)
		self.p = (self.q - 3) ** 3

		# hardcode k for now
		k = 4

		sqrt_k = np.sqrt(k)

		# hardcode B for now
		# B = 20
		B = 9.2 * self.std

		# ALPHA is a constant based on security parameter
		ALPHA = 3.758

		sigma_prime = ( ALPHA ** (1-sqrt_k) ) * ( (self.q-1) ** (k-sqrt_k) ) * (B ** sqrt_k) / 9.2 

		# e = self.gen_normal_poly(std=sigma_prime)
		e = self.gen_normal_poly(std=0)
		a = self.gen_uniform_poly(q=self.p)

		# p * (sk^2)
		# ss = oc.poly_mul_poly(self.sk,self.sk) #ss = self.sk * self.sk
		ss = oc.polypolyMontMul(self.mont, self.sk, self.sk)
		quo,ss = oc.poly_div_poly(ss, self.fn) #quo,ss = ss / self.fn
		mp = oc.toMont( self.mont, self.p )
		ss = oc.polynumMontMul( self.mont, ss, mp )
		# ss = oc.poly_mul_num( ss, self.p ) #ss = ss * self.p

		# -(a*s + e)
		# b = oc.poly_mul_poly(a,self.sk) #b = a * self.sk
		b = oc.polypolyMontMul(self.mont, a, self.sk)
		quo,b = oc.poly_div_poly(b, self.fn) #quo,b = b / self.fn
		b = oc.poly_add_poly( b, e ) #b = b + e
		b = oc.poly_mul_num( b, -1 ) #b = b * -1
		quo,b = oc.poly_div_poly(b, self.fn) #quo,b = b / self.fn

		# -(a*s + e) + p*sk^2
		b = oc.poly_add_poly( ss, b ) #b = ss + b

		self.rlk = (b, a)
		return 

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

		# these would be use in the different functions

	
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

		m = [self.mont.toMont(pt)]
		m = Poly(m)
		m = oc.poly_mod( m, self.q ) #m = m % self.q
		#print( m )

		delta = oc.floor_div(self.q, self.t) #delta = self.q // self.t
		delta = oc.toMont( self.mont, delta )
		scaled_m = m.copy()
		#scaled_m[0] = delta * scaled_m[0] % self.q
		scaled_m = oc.polynumMontMul(self.mont, scaled_m, delta )

		# create a new m, which is scaled my q//t % q
		# generated new error polynomials
		e1 = self.gen_normal_poly()
		e2 = self.gen_normal_poly()
		u = self.gen_binary_poly()

		# create c0 = pk[0]*u + e1 + scaled_m
		ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u, oc), e1, oc), scaled_m, oc)

		# create c1 = pk[1]*u + e2
		ct1 = self.polyadd( self.polymult( self.pk[1], u, oc), e2, oc)

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

		scaled_pt = self.polyadd( self.polymult( ct[1] , self.sk, oc ), ct[0], oc )
		#print( scaled_pt )
		for ind, i in enumerate(scaled_pt):
			scaled_pt[ind] = oc.fromMont( self.mont, i )
			# scaled_pt[ind] = self.mont.fromMont( i )

		scaled_pt = scaled_pt % (self.q-1)
		#scaled_pt = oc.poly_mod( scaled_pt, self.q-1 ) #scaled_pt = scaled_pt % self.t
		print( scaled_pt )
		# scaled_pt = oc.poly_mod( scaled_pt, self.q ) #scaled_pt = scaled_pt % self.t
		tq = oc.true_div( self.t, self.q )

		scaled_pt = scaled_pt * tq

		scaled_pt.round()
		scaled_pt = oc.poly_mod( scaled_pt, self.t ) #scaled_pt = scaled_pt % self.t
		print( scaled_pt )
		decrypted_pt = scaled_pt

		# return the first term of the polynomial, which is the plaintext
		return int(decrypted_pt[0])

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

		# calculate terms for mult
		mt = oc.toMont( self.mont, self.t )
		mq = oc.toMont( self.mont, self.q-1 )
		# invmq = self.mont.inverse_modulus(mq,self.mont.n)
		invq = self.mont.inverse_modulus(self.q-1,self.mont.n)
		invmq = oc.toMont( self.mont, invq )
		# print(f'mq*invmq: {self.mont.fromMont(self.mont.multiplication(mq,invmq))}')

		# c0 = ct0[0]*ct1[0]
		# c0 = oc.poly_mul_poly( x[0], y[0] ) #c0 = x[0] * y[0]
		c0 = oc.polypolyMontMul( self.mont, x[0], y[0] )
		quo,c0 = oc.poly_div_poly( c0, self.fn ) #quo,c0 = c0 / self.fn
		c0 = oc.polynumMontMul( self.mont, c0, mt )
		c0 = oc.polynumMontMul( self.mont, c0, invmq )

		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		# ca = oc.poly_mul_poly( x[0], y[1] )
		# cb = oc.poly_mul_poly( x[1], y[0] )
		ca = oc.polypolyMontMul( self.mont, x[0], y[1] )
		cb = oc.polypolyMontMul( self.mont, x[1], y[0] )
		c1 = oc.poly_add_poly( ca, cb ) #c1 = (x[0]*y[1]) + (x[1]*y[0])

		quo,c1 = oc.poly_div_poly( c1, self.fn ) #quo,c1 = c1 / self.fn
		c1 = oc.polynumMontMul( self.mont, c1, mt )
		c1 = oc.polynumMontMul( self.mont, c1, invmq )

		# c2 = ct0[1]*ct1[1]
		# c2 = oc.poly_mul_poly( x[1], y[1] ) #c2 = x[1] * y[1]
		c2 = oc.polypolyMontMul( self.mont, x[1], y[1] )
		quo,c2 = oc.poly_div_poly( c2, self.fn ) #quo,c2 = c2 / self.fn
		c2 = oc.polynumMontMul( self.mont, c2, mt )
		c2 = oc.polynumMontMul( self.mont, c2, invmq )

		'''
		# check if relin is failing
		t0 = c0
		t1 = oc.polypolyMontMul( self.mont, c1, self.sk )
		quo,t1 = oc.poly_div_poly( t1, self.fn )
		ss = oc.polypolyMontMul( self.mont, self.sk, self.sk )
		quo,ss = oc.poly_div_poly( ss, self.fn )
		t2 = oc.polypolyMontMul( self.mont, c2, ss )
		quo,t2 = oc.poly_div_poly( t2, self.fn )


		scaled_pt = self.polyadd( t0, self.polyadd( t1, t2, oc ), oc )

		for ind, i in enumerate(scaled_pt):
			scaled_pt[ind] = oc.fromMont( self.mont, i )
		
		scaled_pt = oc.poly_mod( scaled_pt, self.q-1 ) #scaled_pt = scaled_pt % self.t
		print( scaled_pt )
		tq = oc.true_div( self.t, self.q )
		scaled_pt = scaled_pt * tq
		scaled_pt.round()
		scaled_pt = oc.poly_mod( scaled_pt, self.t ) #scaled_pt = scaled_pt % self.t
		print(scaled_pt)
		'''

		ret = self.relin(c0,c1,c2)

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
		c20	= ( c2 * rlk[0] ) / p
		c21 = ( c2 * rlk[1] ) / p

		c0' = c0 + c20
		c1' = c1 + c21
		ct' = (c0',c1')

		----- Output -----
		ct'	- Ciphertext relinearized from 3 elements to 2 elements

		"""

		# set relin object
		oc = self.counters['relin']

		mp = oc.toMont( self.mont, self.p )
		invp = self.mont.inverse_modulus( self.p, self.mont.n )
		invmp = oc.toMont( self.mont, invp )
		# invmp = self.mont.inverse_modulus( mp, self.mont.n )
		# print(f'mp = {mp}')
		# print(f'invmp = {invmp}')
		# print(f'mp*invmp = {mp*invmp % ((self.q-1)**3)}')
		# print(f'mp*invmp = {self.mont.fromMont(self.mont.multiplication(mp,invmp))}')

		c2 = oc.polynumMontMul( self.mont, c2, invmp )

		# c20 = (c2 * rlk[0]/p)
		# c20 = oc.poly_mul_poly( c2, self.rlk[0] ) #c20 = c2 * self.rlk[0]	
		c20 = oc.polypolyMontMul( self.mont, c2, self.rlk[0] )
		# c20 = oc.polynumMontMul( self.mont, c20, invmp )
		quo,c20 = oc.poly_div_poly( c20, self.fn ) #quo,c20 = c20 / self.fn

		# c21 = (c2 * rlk[1]/p)
		# c21 = oc.poly_mul_poly( c2, self.rlk[1] ) #c21 = c2 * self.rlk[1]	
		c21 = oc.polypolyMontMul( self.mont, c2, self.rlk[1] )
		# c21 = oc.polynumMontMul( self.mont, c21, invmp )
		quo, c21 = oc.poly_div_poly( c21, self.fn ) #quo,c21 = c21 / self.fn

		_c0 = oc.poly_add_poly( c0, c20 ) #_c0 = c0 + c20
		_c1 = oc.poly_add_poly( c1, c21 ) #_c1 = c1 + c21

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

		z = oc.poly_add_poly( x, y ) #z = x + y
		quo,rem = oc.poly_div_poly( z, self.fn ) #quo,rem = (z / self.fn)
		z = rem

		# z = oc.poly_mod( z, self.q ) #z = z % self.q
		#z = self.mod(z)
		return z

	def polymult(self, x, y, oc=None):
		"""
		multiply two polynomials together and keep them 
		within the polynomial ring
		"""
		if ( oc == None ):
			oc = self.opcount

		# z = oc.poly_mul_poly( x, y ) #z = x * y
		z = oc.polypolyMontMul( self.mont, x, y )

		'''
		z = Poly( [0]*(len(x)+len(y)) )

		for ind, i in enumerate(x):
			for jnd, j in enumerate(y):
				z[ind+jnd] += self.mont.multiplication( i, j )
		'''

		quo, rem = oc.poly_div_poly( z, self.fn ) #quo, rem = (z / self.fn)
		z = rem

		# z = oc.poly_mod( z, self.q ) #z = z % self.q
		#z = self.mod(z)
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
			a.append( self.mont.toMont( int(np.random.normal(c,std)) ) )
		a = Poly(a)
		return a

	def gen_binary_poly(self):
		"""
		generate a random polynomial of degree n-1
		with coefficients ranging from [0,1]
		"""
		a = []
		for i in range(self.n):
			a.append( self.mont.toMont( np.random.randint(0,2) ) )
		a = Poly(a)
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
			a.append( self.mont.toMont( random.randint(0,q) ) )
		a = Poly(a)

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
	pass


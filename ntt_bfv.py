#!/usr/bin/env python
# this program will be an attempt to create the bfv
# encrption scheme that is described in the BFV paper.

import numpy as np
from numpy.polynomial import polynomial as p
import random
import sys
from ntt import NTT

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
		self.ntt.psi = 3213
		self.ntt.invpsi = 3145
		self.ntt.psi = 20237
		self.ntt.invpsi = 16233

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

		# create a new polynomial _a which is -a
		_a = oc.poly_mul_num(a, -1) 
		#_a = a * -1

		# then set e = -e
		e = oc.poly_mul_num(e, -1) 
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
		a <- Uniformat Distribution over q*p
		e <- Normal Distribution with (mean = 0, standard deviation = sigma_prime )

		----- Calculation -----
		p = q ** 3
		sigma_prime = (alpha ** (1-sqrt(k))) * (q ** (k-sqrt(k))) * (B ** sqrt(k))
		b = -(a*sk + e) + p*(sk^2)

		----- Result -----
		rlk = (b , a)

		"""

		# create new ntt instance
		self.rlk_ntt = NTT(self.n,18395777)
		self.rlk_ntt.psi = 2923173
		self.rlk_ntt.invpsi = 1254665
		'''
		self.rlk_ntt = NTT(self.n,1204731713)
		self.rlk_ntt.psi = 269672302
		self.rlk_ntt.invpsi = 366977299
		'''

		# set counter object for keys
		oc = self.counters['key']

		# define p for relin2
		# bigger p means less noise (I think)
		self.p = self.rlk_ntt.N // self.ntt.N
		self.p += 1

		# hardcode k for now
		k = 2

		sqrt_k = np.sqrt(k)

		# hardcode B for now
		# B = 20
		B = 9.2 * self.std

		# ALPHA is a constant based on security parameter
		ALPHA = 3.758
		
		try: 
			sigma_prime = ( ALPHA ** (1-sqrt_k) ) * ( self.q ** (k-sqrt_k) ) * (B ** sqrt_k) / 9.2 
		except OverflowError:	
			sqrt_k = int(sqrt_k)
			sigma_prime = int(int( 4 ** (1-sqrt_k) ) * ( self.q ** (k-sqrt_k) ) * (int(B) ** sqrt_k) // 9) 

		e = self.gen_normal_poly(std=sigma_prime)
		a = self.gen_uniform_poly(q=self.q*self.p)
		e = self.ntt.merge_iNTT( e )
		a = self.ntt.merge_iNTT( a )
		sk = self.ntt.merge_iNTT( self.sk )

		e = Poly([0]*self.n)

		e = self.rlk_ntt.merge_NTT( e )
		a = self.rlk_ntt.merge_NTT( a )
		sk = self.rlk_ntt.merge_NTT( sk )

		# p * (sk^2)
		ss = oc.dotProduct(sk, sk)
		ss = oc.poly_mul_num( ss, self.p ) 
		ss = oc.poly_mod( ss, self.rlk_ntt.N )

		# -(a*s + e)
		b = oc.dotProduct(a,sk) 
		b = oc.poly_add_poly( b, e ) 
		b = oc.poly_mul_num( b, -1 ) 

		# -(a*s + e) + p*sk^2
		b = oc.poly_add_poly( ss, b ) 
		#b = ss + b

		b = oc.poly_mod( b, self.rlk_ntt.N ) 
		#b = b % (self.q * self.p)

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
		invq = self.ntt.extended_euclidean( self.q, self.ntt.N )

		x = [i.copy() for i in x]
		x[0] = oc.poly_mul_num(x[0], t)
		x[1] = oc.poly_mul_num(x[1], t)

		# c0 = ct0[0]*ct1[0]
		c0 = oc.dotProduct( x[0], y[0] ) #c0 = x[0] * y[0]
		# c0 = oc.poly_mul_num( c0, t ) #c0 = c0 * ( self.t / self.q )
		# c0 = oc.poly_mul_num( c0, invq ) #c0 = c0 * ( self.t / self.q )
		c0 = oc.poly_mod( c0, self.q ) #c0 = c0 % self.q

		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		ca = oc.dotProduct( x[0], y[1] )
		cb = oc.dotProduct( x[1], y[0] )
		c1 = oc.poly_add_poly( ca, cb ) #c1 = (x[0]*y[1]) + (x[1]*y[0])

		# c1 = oc.poly_mul_num( c1, t ) #c1 = c1 * ( self.t / self.q )
		# c1 = oc.poly_mul_num( c1, invq ) #c1 = c1 * ( self.t / self.q )
		c1 = oc.poly_mod( c1, self.q ) #c1 = c1 % self.q

		# c2 = ct0[1]*ct1[1]
		c2 = oc.dotProduct( x[1], y[1] ) #c2 = x[1] * y[1]
		# c2 = oc.poly_mul_num( c2, t ) #c2 = c2 * ( self.t / self.q )
		# c2 = oc.poly_mul_num( c2, invq ) #c2 = c2 * ( self.t / self.q )
		c2 = oc.poly_mod( c2, self.q ) #c2 = c2 % self.q

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

		'''
		c0 = self.ntt.merge_iNTT( c0 )
		c0 = self.rlk_ntt.merge_NTT( c0 )
		c1 = self.ntt.merge_iNTT( c1 )
		c1 = self.rlk_ntt.merge_NTT( c1 )
		'''
		c2 = self.ntt.merge_iNTT( c2 )
		c2 = self.rlk_ntt.merge_NTT( c2 )

		# invp = self.rlk_ntt.inverse_mod( self.p, self.rlk_ntt.N )
		invp = self.rlk_ntt.extended_euclidean( self.p, self.rlk_ntt.N )

		# c20 = (c2 * rlk[0]/p)
		c20 = oc.dotProduct( c2, self.rlk[0] ) #c20 = c2 * self.rlk[0]	
		c20 = oc.poly_mul_num( c20, invp ) #c20 = c20 / self.p
		c20 = self.rlk_ntt.merge_iNTT( c20 )
		c20 = self.ntt.merge_NTT( c20 )
		# c20.round()
		# c20 = oc.poly_mod( c20, self.q ) #c20 = c20 % self.q

		# c21 = (c2 * rlk[1]/p)
		c21 = oc.dotProduct( c2, self.rlk[1] ) #c21 = c2 * self.rlk[1]	
		c21 = oc.poly_mul_num( c21, invp ) #c21 = c21 / self.p
		c21 = self.rlk_ntt.merge_iNTT( c21 )
		c21 = self.ntt.merge_NTT( c21 )
		# c21.round()
		# c21 = oc.poly_mod( c21, self.q ) #c21 = c21 % self.q

		# c0' = c0 + c20
		# c1' = c1 + c21
		_c0 = oc.poly_add_poly( c0, c20 ) #_c0 = c0 + c20
		_c1 = oc.poly_add_poly( c1, c21 ) #_c1 = c1 + c21

		_c0 = oc.poly_mod( _c0, self.q ) #_c0 = _c0 % self.q
		_c1 = oc.poly_mod( _c1, self.q ) #_c1 = _c1 % self.q

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

#!/usr/bin/env python
# this program will be an attempt to create the lpr.es
# encrption scheme that is described in the BFV paper.
# This encryption scheme will then be transformed into 
# a SWHE which will then be turned into a FHE scheme
#
#
# this will also have some testing code for me to test
# polynomial ring operations

import numpy as np
from numpy.polynomial import polynomial as p
import random
import sys

from counter import OperationsCounter
from poly import Poly

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

class LPR():

	def __init__(self,q=2**15,t=2**8,n=2**4,std=2,fn=None,T=None):
		# this init method will initialize the important variables
		# needed for this encryption scheme
		self.q = q
		self.t = t
		self.n = n
		self.std = std
		# this will set the polynomial for the ring, if not declared then will be
		# the polynomial 1 + x^n
		self.fn = fn
		if (self.fn == None):
			self.fn = [1] + [0]*((n)-1) + [1]
		self.fn = Poly(self.fn)
		# this will set the variable T, as needed for relinearization1 for BFV
		self.T = T

		self.gen_op_counters()

		# this will set the keys as none, but will then immediately generate
		# a public key, a private key, and a relinearization key
		self.sk = None
		self.pk = None
		self.rlk = None
		self.gen_keys()


	def gen_keys(self):
		# calls the different functions to generate the keys
		self.gen_sk()
		self.gen_pk()
		self.gen_rlk()
		
	def gen_sk(self):
		# call the gen_binary_poly key to create a polynomial
		# of only 0's and 1's for the secret key
		self.sk = self.gen_binary_poly()
		#self.sk = self.gen_normal_poly()
		
		# set hamming weight of secret key to 64
		if (self.n > 128 ):
			sk = [1]*64
			sk = sk + [0]*(self.n-64)
			np.random.shuffle( sk )
			self.sk = Poly( sk )
		
		return

	def gen_pk(self,test=None):
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

		if (test == 1):
			a = Poly([71,239,2,243,73,213,85,184])
			e = Poly([0,-1,1,-1,2,-2,-3,-1])

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

		# set counter object for keys
		oc = self.counters['key']

		# define p for relin2
		# bigger p means less noise (I think)
		self.p = (self.q) ** 3

		# hardcode k for now
		k = 4

		sqrt_k = np.sqrt(k)

		# hardcode B for now
		# B = 20
		B = 9.2 * self.std

		# ALPHA is a constant
		ALPHA = 3.758

		sigma_prime = ( ALPHA ** (1-sqrt_k) ) * ( self.q ** (k-sqrt_k) ) * (B ** sqrt_k) / 9.2 

		e = self.gen_normal_poly(std=sigma_prime)
		a = self.gen_uniform_poly(q=self.q*self.p)

		# p * (sk^2)
		ss = oc.poly_mul_poly(self.sk,self.sk) #ss = self.sk * self.sk
		quo,ss = oc.poly_div_poly(ss, self.fn) #quo,ss = ss / self.fn
		ss = oc.poly_mul_num( ss, self.p ) #ss = ss * self.p

		# -(a*s + e)
		b = oc.poly_mul_poly(a,self.sk) #b = a * self.sk
		quo,b = oc.poly_div_poly(b, self.fn) #quo,b = b / self.fn
		b = oc.poly_add_poly( b, e ) #b = b + e
		b = oc.poly_mul_num( b, -1 ) #b = b * -1

		# -(a*s + e) + p*sk^2
		b = oc.poly_add_poly( ss, b ) #b = ss + b

		qp = oc.num_mul(self.q, self.p)
		b = oc.poly_mod( b, qp ) #b = b % (self.q * self.p)

		self.rlk = (b, a)
		return 

	def gen_op_counters(self):
		# this function will generate operation counters
		# for encrypting, decrypting, key generation, 
		# addition, multiplication, and relinearization

		# create operations counter object
		self.opcount = OperationsCounter()
		self.counters = {}
		self.counters['enc'] = OperationsCounter()
		self.counters['dec'] = OperationsCounter()
		self.counters['key'] = OperationsCounter()
		self.counters['add'] = OperationsCounter()
		self.counters['mul'] = OperationsCounter()
		self.counters['relin'] = OperationsCounter()

		# these would be use in the different functions

	
	def encrypt(self,pt=0,test=None):
		# encode plaintext into a plaintext polynomial
		# create polynomial m, which is pt%q

		# set encryption counter object 
		oc = self.counters['enc']

		m = [pt]
		m = Poly(m)
		m = oc.poly_mod( m, self.q ) #m = m % self.q
		#print( m )

		delta = oc.floor_div(self.q, self.t) #delta = self.q // self.t
		scaled_m = m.copy()
		#scaled_m[0] = delta * scaled_m[0] % self.q
		scaled_m = oc.poly_mul_num( scaled_m, delta ) #scaled_m = (scaled_m * delta) 
		scaled_m = oc.poly_mod( scaled_m, self.q ) #scaled_m = scaled_m  % self.q
		# create a new m, which is scaled my q//t % q
		# generated new error polynomials
		e1 = self.gen_normal_poly()
		e2 = self.gen_normal_poly()
		u = self.gen_binary_poly()
		if (test == 0):
			u = Poly([1,1,1,1,1,1,0,1])
			e1 = Poly([-2,-2,0,4,-3,-1,0,-4])
			e2 = Poly([1,-1,1,3,-1,-4,0,1])

		if (test == 1):
			u = Poly([0,0,0,1,0,1,1,1])
			e1 = Poly([6,0,1,1,-2,-2,-2,-2])
			e2 = Poly([0,-1,3,-2,2,1,-1,1])

		# create c0 = pk[0]*u + e1 + scaled_m
		ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u, oc), e1, oc), scaled_m, oc)

		# create c1 = pk[1]*u + e2
		ct1 = self.polyadd( self.polymult( self.pk[1], u, oc), e2, oc)

		return (ct0, ct1)

	def decrypt(self,ct):
		# decrypt the cipher text to get the plaintext equivalent

		# set decryption counter object
		oc = self.counters['dec']

		# scaled_pt = ct[1]*sk + ct[0]
		#scaled_pt = self.polyadd( self.polymult( ct[1], self.sk ), ct[0] )
		scaled_pt = self.polyadd( oc.poly_mul_poly( ct[1] , self.sk ), ct[0], oc )

		tq = oc.true_div( self.t, self.q )
		scaled_pt = oc.poly_mul_num( scaled_pt, tq ) #scaled_pt = scaled_pt * ( self.t / self.q)
		scaled_pt.round()
		scaled_pt = oc.poly_mod( scaled_pt, self.t ) #scaled_pt = scaled_pt % self.t
		print( scaled_pt )
		decrypted_pt = scaled_pt

		#decrypted_pt.polyprint()

		# return the first term of the polynomial, which is the plaintext
		return int(decrypted_pt[0])

	def decrypt3(self,ct):

		t0 = ct[0]
		t1 = self.polymult( self.sk, ct[1] )
		t2 = self.polymult( self.polymult( self.sk, self.sk ), ct[2] )

		scaled_pt = self.polyadd( self.polyadd( t2, t1), t0 )
		decrypted_pt = []
		# scale each coefficient by t/q % t
		for ind, i in enumerate( scaled_pt ):
			scaled_pt[ind] = (  round(i * self.t / self.q ) % self.t )
		
		# create a polynomial from the list
		decrypted_pt = Poly(decrypted_pt)

		scaled_pt.polyprint()

		return int(scaled_pt[0])


	def ctadd(self, x, y):
		# X and Y are two cipher texts generated
		# by this encrypted scheme

		# set adder counter object
		oc = self.counters['add']

		ct0 = self.polyadd(x[0],y[0],oc)
		ct1 = self.polyadd(x[1],y[1],oc)

		ct = (ct0,ct1)

		return ct

	def ctmult(self, x, y, test=None):
		# multiply cipher texts X and Y and return ciphertext X*Y
		# still work in progress, not working yet

		# set multiplier counter object
		oc = self.counters['mul']

		z = []
		z.append(x[0].copy())
		z.append(x[1].copy())

		# scale both polynomials in z by (t/q)
		for ind, num in enumerate(z[0]):
			nt = oc.num_mul( num, self.t )
			ntq= oc.true_div( nt, self.q )
			z[0][ind] = round(num * self.t / self.q)

		for ind, num in enumerate(z[1]):
			nt = oc.num_mul( num, self.t )
			ntq= oc.true_div( nt, self.q )
			z[1][ind] = round(num * self.t / self.q)


		# c0 = ct0[0]*ct1[0]
		#c0 = self.polymult( z[0], y[0] )
		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		#c1 = self.polyadd( self.polymult(z[0],y[1]), self.polymult(z[1],y[0]) )
		# c2 = ct0[1]*ct1[1]
		#c2 = self.polymult( z[1], y[1] )

		# c0 = ct0[0]*ct1[0]
		c0 = oc.poly_mul_poly( x[0], y[0] ) #c0 = x[0] * y[0]
		quo,c0 = oc.poly_div_poly( c0, self.fn ) #quo,c0 = c0 / self.fn
		tq = oc.true_div( self.t, self.q )
		c0 = oc.poly_mul_num( c0, tq ) #c0 = c0 * ( self.t / self.q )
		c0.round()
		c0 = oc.poly_mod( c0, self.q ) #c0 = c0 % self.q

		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		ca = oc.poly_mul_poly( x[0], y[1] )
		cb = oc.poly_mul_poly( x[1], y[0] )
		c1 = oc.poly_add_poly( ca, cb ) #c1 = (x[0]*y[1]) + (x[1]*y[0])

		quo,c1 = oc.poly_div_poly( c1, self.fn ) #quo,c1 = c1 / self.fn
		tq = oc.true_div( self.t, self.q )
		c1 = oc.poly_mul_num( c1, tq ) #c1 = c1 * ( self.t / self.q )
		c1.round()
		c1 = oc.poly_mod( c1, self.q ) #c1 = c1 % self.q

		# c2 = ct0[1]*ct1[1]
		c2 = oc.poly_mul_poly( x[1], y[1] ) #c2 = x[1] * y[1]
		quo,c2 = oc.poly_div_poly( c2, self.fn ) #quo,c2 = c2 / self.fn
		tq = oc.true_div( self.t, self.q )
		c2 = oc.poly_mul_num( c2, tq ) #c2 = c2 * ( self.t / self.q )
		c2.round()
		c2 = oc.poly_mod( c2, self.q ) #c2 = c2 % self.q

		if (test == 1):
			assert c0 == Poly([142,144,235,21,224,118,152,123])
			assert c1 == Poly([40,189,27,73,242,152,98,40])
			assert c2 == Poly([22,86,133,29,199,110,199,50])

		ret = self.relin(c0,c1,c2)

		return ret

		'''
		failed code, did not produce correct multiplication
		keeping commented out for later study with why it failed

		# calculate fc0 
		fc0 = self.polymult( x[0], y[0] )
		for ind, i in enumerate(fc0):
			fc0[ind] = round(i * self.t / self.q) #% self.q

		# calculate fc1
		t0 = self.polymult(x[1],y[0])
		t1 = self.polymult(x[0],y[1])
		fc1 = self.polyadd( t0, t1)
		for ind, i in enumerate(fc1):
			fc1[ind] = round(i * self.t / self.q) #% self.q

		# calculate fc2
		fc2 = self.polymult( x[1], y[1] )
		for ind, i in enumerate(fc2):
			fc2[ind] = round(i * self.t / self.q) #% self.q

		ret = self.relin1(fc0,fc1,fc2)
		return ret
		return (ret,(c0,c1,c2))
		'''

	def relin(self,c0,c1,c2):

		# set relin object
		oc = self.counters['relin']

		# c20 = (c2 * rlk[0]/p)
		c20 = oc.poly_mul_poly( c2, self.rlk[0] ) #c20 = c2 * self.rlk[0]	
		quo,c20 = oc.poly_div_poly( c20, self.fn ) #quo,c20 = c20 / self.fn
		c20 = oc.poly_div_num( c20, self.p ) #c20 = c20 / self.p
		c20.round()
		c20 = oc.poly_mod( c20, self.q ) #c20 = c20 % self.q

		# c21 = (c2 * rlk[1]/p)
		c21 = oc.poly_mul_poly( c2, self.rlk[1] ) #c21 = c2 * self.rlk[1]	
		quo, c21 = oc.poly_div_poly( c21, self.fn ) #quo,c21 = c21 / self.fn
		c21 = oc.poly_div_num( c21, self.p ) #c21 = c21 / self.p
		c21.round()
		c21 = oc.poly_mod( c21, self.q ) #c21 = c21 % self.q

		# c0' = c0 + c20
		# c1' = c1 + c21
		_c0 = oc.poly_add_poly( c0, c20 ) #_c0 = c0 + c20
		_c1 = oc.poly_add_poly( c1, c21 ) #_c1 = c1 + c21

		_c0 = oc.poly_mod( _c0, self.q ) #_c0 = _c0 % self.q
		_c1 = oc.poly_mod( _c1, self.q ) #_c1 = _c1 % self.q

		return (_c0, _c1)

	def mod(self,poly):
		# calculate the modulus of poly by q
		# with answer given back in range (-q/2,q/2]
		copy = poly.poly.copy()
		for ind,i in enumerate(copy):
			i = i % self.q
			if ( i > (self.q/2) ):
				copy[ind] = i - self.q

		return Poly(copy)

	def polyadd(self, x, y, oc=None):
		# add two polynomials together and keep them 
		# within the polynomial ring
		if ( oc == None ):
			oc = self.opcount
		z = oc.poly_add_poly( x, y ) #z = x + y
		quo,rem = oc.poly_div_poly( z, self.fn ) #quo,rem = (z / self.fn)
		z = rem

		z = oc.poly_mod( z, self.q ) #z = z % self.q
		#z = self.mod(z)
		return z

	def polymult(self, x, y, oc=None):
		# multiply two polynomials together and keep them 
		# within the polynomial ring
		if ( oc == None ):
			oc = self.opcount

		z = oc.poly_mul_poly( x, y ) #z = x * y
		quo, rem = oc.poly_div_poly( z, self.fn ) #quo, rem = (z / self.fn)
		z = rem

		z = oc.poly_mod( z, self.q ) #z = z % self.q
		#z = self.mod(z)
		return z

	def gen_normal_poly(self,c=0,std=None):
		# generate a random polynomial of degree n-1
		# with coefficients selected from normal distribution
		# with center at 0 and std of 2. Each term is rounded
		# down to nearest integer
		if ( std == None ):
			std = self.std
		a = []
		for i in range(self.n):
			a.append( int(np.random.normal(c,std)) )
		a = Poly(a)
		return a

	def gen_binary_poly(self):
		# generate a random polynomial of degree n-1
		# with coefficients ranging from [0,1]
		a = []
		for i in range(self.n):
			a.append( np.random.randint(0,2) )
		a = Poly(a)
		return a

	def gen_uniform_poly(self,q=None):
		# generate a random polynomial of degree n-1
		# with coefficients ranging from [0,q)
		if (q == None):
			q = self.q
		a = []
		for i in range(self.n):
			a.append( random.randint(0,q) )
		a = Poly(a)

		return a

	def poly_base_change(self,poly,q,T):
		# change the base of a polynomial and
		# return multiple polynomials of new base
		# haven't tested this, don't know if it works
		l = int(np.floor(np.log2(q)/np.log2(T)))
		cpy = poly.copy()
		base_poly = []

		for ind, i in enumerate(cpy):
			if i < 0:
				cpy[ind] = i + self.q

		for i in range(l+1):
			mask = []
			for jnd, j in enumerate(cpy):
				_Tpow = T ** (i+1)
				_T = T ** i
				num = j % _Tpow
				mask.append( int( num / _T ) )
				cpy[jnd] -= num

			base_poly.append( Poly(mask) )

		return base_poly

	def print_counter_info(self):
		# this function will print out the operation
		# costs for each function (enc,dec,ctadd,ctmult...)

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

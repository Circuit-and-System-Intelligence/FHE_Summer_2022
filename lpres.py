#!/usr/bin/env python
# this program will be an attempt to create the lpr.es
# encrption scheme that is described in the BFV paper.
# This encryption scheme will then be transformed into 
# a SWHE which will then be turned into a FHE scheme
#
#
# this will also have some testing code for me to test
# polynomial ring operations

from matplotlib.pyplot import sca
import numpy as np
from numpy.polynomial import polynomial as p
import sys

from sklearn.preprocessing import scale
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

	return

class LPR():

	def __init__(self,q=2**15,t=2**8,n=2**4,fn=None,T=None):
		# this init method will initialize the important variables
		# needed for this encryption scheme
		self.q = q
		self.t = t
		self.n = n
		# this will set the polynomial for the ring, if not declared then will be
		# the polynomial 1 + x^n
		self.fn = fn
		if (self.fn == None):
			self.fn = [1] + [0]*(n-1) + [1]
		self.fn = Poly(self.fn)
		# this will set the variable T, as needed for relinearization1 for BFV
		self.T = T
		if (self.T == None):
			# if not defined, set T as the square root of q, rounded to highest up
			self.T = int(np.ceil(np.sqrt(self.q)))

		# this will set the keys as none, but will then immediately generate
		# a public key, a private key, and a relinearization key
		self.sk = None
		self.pk = None
		self.rlk = None
		self.gen_keys()

	def gen_keys(self):
		# calls the different functions to generate the keys
		self.gensk()
		self.genpk()
		self.genrlk1()
		
	def gensk(self):
		# call the gen_binary_poly key to create a polynomial
		# of only 0's and 1's for the secret key
		self.sk = self.gen_binary_poly()
		#self.sk = self.gen_normal_poly()
		return

	def genpk(self,test=None):
		if (self.sk == None):
			return
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
		'''
		_a = []
		for i in a:
			_a.append(-1*i)
		_a = Poly(_a)
		'''
		_a = a * -1

		# then set e = -e
		'''
		for ind, i in enumerate(e):
			e[ind] = -1 * i
		'''
		e = e * -1

		# create b from (-a * sk) - e
		b = self.polyadd( self.polymult(_a, self.sk),e)

		# set the public key to the tuple (b,a)
		# or (-[a*sk + e], a)
		self.pk = (b,a)
		return
	
	def genrlk1(self,test=None):
		# use change of base rule for logs to calculate logT(q)
		# using log2 because most likely self.q and self.T are in base 2
		self.l = int(np.floor(np.log2(self.q)/np.log2(self.T)))

		# create the different masks for the rlk key
		self.rlk = []
		'''
		ss = self.polymult(self.sk,self.sk)
		'''

		if (test == 1):
			pregen = []
			pregen.append( ( Poly([182,174,245,89,3,176,60,162]) , Poly([0,3,0,-4,-2,1,2,-3]) ) )
			pregen.append( ( Poly([60,186,134,177,240,28,50,218]) , Poly([-2,1,3,-3,-4,1,-3,1]) ) )
			pregen.append( ( Poly([122,144,143,158,23,81,240,137]) , Poly([2,1,-2,-3,1,1,-1,-1]) ) )
			pregen.append( ( Poly([212,77,255,165,216,115,221,239]) , Poly([-1,-2,-1,0,-4,-3,1,1]) ) )
			pregen.append( ( Poly([191,190,56,62,149,118,52,95]) , Poly([1,-1,1,3,-1,-4,0,1]) ) )

		#a = self.gen_uniform_poly()
		#e = self.gen_normal_poly()
		for i in range(self.l+1):
			# generate the different random polynomials needed
			a = self.gen_uniform_poly()
			e = self.gen_normal_poly()
			if (test == 1):
				a = pregen[i][0]
				e = pregen[i][1]
			'''
			_a = a.copy()
			for jnd, j in enumerate(_a):
				_a[jnd] = -1 * j
			for jnd, j in enumerate(e):
				e[jnd] = -1 * j
			'''
			_a = a * -1
			e = e * -1

			b = self.polyadd( self.polymult(_a, self.sk), e)
			T = self.T ** i
			ss = self.sk * T
			'''
			s2 = ss.copy()
			for jnd, j in enumerate(s2):
				s2[jnd] = j * T 
			'''
			s2 = self.polymult( ss, self.sk )
			b = self.polyadd( b, s2 )
			self.rlk.append( (b,a) )
			
		#self.rlk = rlk.copy()
		return
	
	
	def encrypt(self,pt=0,test=None):
		# encode plaintext into a plaintext polynomial
		# create polynomial m, which is pt%q
		m = [pt]
		m = Poly(m)
		m = m % self.q

		delta = self.q // self.t
		scaled_m = m.copy()
		scaled_m[0] = delta * scaled_m[0] % self.q
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
		ct0 = self.polyadd( self.polyadd( self.polymult( self.pk[0], u), e1), scaled_m)

		# create c1 = pk[1]*u + e2
		ct1 = self.polyadd( self.polymult( self.pk[1], u), e2)

		return (ct0, ct1)

	def decrypt(self,ct):
		# decrypt the cipher text to get the plaintext equivalent

		# scaled_pt = ct[1]*sk + ct[0]
		#scaled_pt = self.polyadd( self.polymult( ct[1], self.sk ), ct[0] )
		scaled_pt = self.polyadd( ( ct[1] * self.sk ), ct[0] )
		'''
		decrypted_pt = []
		# scale each coefficient by t/q % t
		for ind, i in enumerate( scaled_pt ):
			decrypted_pt.append(  round(i * self.t / self.q ) % self.t )
		
		# create a polynomial from the list
		decrypted_pt = Poly(decrypted_pt)
		'''

		scaled_pt = scaled_pt * ( self.T / self.q)
		scaled_pt.round()
		scaled_pt = scaled_pt % self.t
		decrypted_pt = scaled_pt

		decrypted_pt.polyprint()

		# return the first term of the polynomial, which is the plaintext
		return int(decrypted_pt[0])

	def decrypt3(self,ct):
		#print('here')
		t0 = ct[0]
		t1 = self.polymult( self.sk, ct[1] )
		t2 = self.polymult( self.polymult( self.sk, self.sk ), ct[2] )
		#scaled_pt = self.polyadd( self.polymult( self.polymult( self.sk, self.sk ), ct[2] ), self.polyadd( self.polymult( ct[1], self.sk ), ct[0] ) )
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
		ct0 = self.polyadd(x[0],y[0])
		ct1 = self.polyadd(x[1],y[1])

		ct = (ct0,ct1)

		return ct

	def ctmult(self, x, y, test=None):
		# multiply cipher texts X and Y and return ciphertext X*Y
		# still work in progress, not working yet

		z = []
		z.append(x[0].copy())
		z.append(x[1].copy())

		# scale both polynomials in z by (t/q)
		for ind, num in enumerate(z[0]):
			z[0][ind] = round(num * self.t / self.q)

		for ind, num in enumerate(z[1]):
			z[1][ind] = round(num * self.t / self.q)


		# c0 = ct0[0]*ct1[0]
		c0 = self.polymult( z[0], y[0] )
		# c1 = ct0[0]*ct1[1] + ct0[1]*ct1[0]
		c1 = self.polyadd( self.polymult(z[0],y[1]), self.polymult(z[1],y[0]) )
		# c2 = ct0[1]*ct1[1]
		c2 = self.polymult( z[1], y[1] )

		c0 = x[0] * y[0]
		quo,c0 = c0 / self.fn
		c0 = c0 * ( self.t / self.q )
		c0.round()
		c0 = c0 % self.q

		c1 = (x[0]*y[1]) + (x[1]*y[0])
		quo,c1 = c1 / self.fn
		c1 = c1 * ( self.t / self.q )
		c1.round()
		c1 = c1 % self.q

		c2 = x[1] * y[1]
		quo,c2 = c2 / self.fn
		c2 = c2 * ( self.t / self.q )
		c2.round()
		c2 = c2 % self.q

		if (test == 1):
			assert c0 == Poly([142,144,235,21,224,118,152,123])
			assert c1 == Poly([40,189,27,73,242,152,98,40])
			assert c2 == Poly([22,86,133,29,199,110,199,50])

		ret = self.relin1(c0,c1,c2)
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

	def relin1(self,c0,c1,c2):
		# still work in progress, not completed
		# calculate c2T, which would be c2 in base T

		c2T = self.poly_base_change(c2,self.q,self.T)

		summ0 = Poly()
		summ1 = Poly()

		for i in range(self.l+1):
			#summ0 = self.polyadd( summ0, self.polymult(self.rlk[i][0], c2T[i] ) )
			#summ1 = self.polyadd( summ1, self.polymult(self.rlk[i][1], c2T[i] ) )
			summ0 = summ0 + ( self.rlk[i][0] * c2T[i] )
			summ1 = summ1 + ( self.rlk[i][1] * c2T[i] )
		
		q,summ0 = summ0 / self.fn
		q,summ1 = summ1 / self.fn

		_c0 = c0 + summ0
		_c1 = c1 + summ1

		_c0 = _c0 % self.q
		_c1 = _c1 % self.q
		
		#_c0 = self.polyadd( c0, summ0 )
		#_c1 = self.polyadd( c1, summ1 )

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

	def polyadd(self, x, y):
		# add two polynomials together and keep them 
		# within the polynomial ring
		z = x + y
		quo,rem = (z / self.fn)
		z = rem
		#for ind, i in enumerate(z):
		#	z[ind] = round(i)
		z = z % self.q
		#z = self.mod(z)
		return z

	def polymult(self, x, y):
		# multiply two polynomials together and keep them 
		# within the polynomial ring
		z = x * y
		quo, rem = (z / self.fn)
		z = rem
		#for ind, i in enumerate(z):
		#	z[ind] = round(i)
		z = z % self.q
		#z = self.mod(z)
		return z

	def gen_normal_poly(self,c=0,std=2):
		# generate a random polynomial of degree n-1
		# with coefficients selected from normal distribution
		# with center at 0 and std of 2. Each term is rounded
		# down to nearest integer
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
			a.append( np.random.randint(0,q) )
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

if __name__ == '__main__':
	main()
	pass

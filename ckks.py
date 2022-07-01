#!/usr/bin/env python
# Date: 6/8/2022
# Create a Class to emulate ckks encryption scheme 
# 
# This program will create a class for the ckks 
# fully homomorphic encryption scheme

from poly import Poly
from vector import Vector, Matrix, vdot, matmul, linalg
import numpy as np
import random

class CKKS():

	def __init__(self,M=8,delta=64,L=3,q0=2**12,h=2,std=3.2):
		self.M = M
		self.N = M // 2
		# encoding variables
		self.xi = np.exp( 2 * np.pi * 1j / self.M )
		self.delta = delta
		self.create_sigma_R_basis()
		# encryption variables
		self.gen_ring_poly()
		self.L = L
		self.q0= q0
		self.p = self.delta
		self.h = h
		self.std = std
		self.q = self.q0 * (self.p ** self.L) 
		self.P = (self.q ** 3) + 1
		self.key_gen()
		return

	def gen_ring_poly(self):
		# this will generate the cyclotomic polynomial
		fn = [1] + (self.N-1)*[0] + [1]
		self.fn = Poly( fn )
		return

	def key_gen(self):
		# generate secret key, public key, and evaluation key
		self.sk_gen()
		self.pk_gen()
		self.evk_gen()
		return

	def sk_gen(self):
		# generate secret key
		#self.sk = self.gen_binary(N=self.N)
		arr = np.random.choice([-1,1],size=self.h).tolist()
		arr = arr + (self.N - self.h) * [0]
		rng = np.random.default_rng()
		rng.shuffle( arr )
		self.sk = Poly( arr )
		return

	def pk_gen(self):
		# generate public key
		# pk = ( b , a )
		# a <- uniform dist
		# e <- normal dist
		# s <- secret key
		# b = -a*s + e
		a = self.gen_uniform(self.q,self.N)
		e = self.gen_normal(N=self.N)

		b = a * -1
		b = b * self.sk
		quo,b = b // self.fn
		b = b + e
		b = b % self.q
		#b = self.ring_mod( b, self.q )

		self.pk = ( b , a )
		return

	def evk_gen(self):
		# generate relinearization key
		# rlk = ( b' , a' )
		# a' <- uniform dist (P*q)
		# e  <- normal dist
		# s  <- secret key
		# b' = -a*s + e + P*s^2
		a = self.gen_uniform( self.q*self.P, self.N )
		e = self.gen_normal(N=self.N)

		ss = self.sk * self.sk
		quo,ss = ss // self.fn
		ss = ss * self.P
		
		b = a * -1
		b = b * self.sk
		quo,b = b // self.fn
		b = b + e

		b = b + ss
		
		#b = self.ring_mod( b, (self.q * self.P) )
		b = b % (self.q * self.P)

		self.evk = ( b, a )
		return

	def gen_binary(self,N=None):
		# generate a binary polynomial
		if (N == None):
			N = self.N
		arr = np.random.randint(-1,2,size=[1,N]).tolist()[0]
		return Poly( arr )

	def gen_normal(self,center=0,std=None,N=None):
		# generate a normal distribution of integers
		if (N == None):
			N = self.N
		if (std == None):
			std = self.std
		arr = np.random.normal(center,std,size=[1,N]).tolist()[0]
		arr = Poly( arr )
		arr.floor()
		return arr

	def gen_uniform(self,q,N=None):
		# generate a uniform distribution of integers
		if (N == None):
			N = self.N
		arr = []
		for i in range(N):
			arr.append( random.randint(0,q) )
		return Poly( arr )

	def gen_zo(self,p,N=None):
		# generate a zo distribution. {-1,0,1} with a 
		# p/2 probability for choosing -1,1 and a 1-p 
		# probability of choosing 0
		if (N == None):
			N = self.N
		arr = np.random.choice( [0,1,-1], size=[1,N], p=[1-p,p/2,p/2] ).tolist()[0]
		return Poly( arr )

	def encrypt(self, m):
		# encode a plaintext polynomial into a ciphertext polynomial
		for ind, i in enumerate(m):
			m[ind] = int( i.real )
		m = Poly( m )
		cm = m.copy()
		m = m.copy()
		#print( m )
		#m = m % self.q
		#print( self.q )
		#m = self.ring_mod( m, self.q )
		m = m % self.q

		# v <- ZO(0.5) , e0 <- normal dist, e1 < normal dist
		v = self.gen_zo( 0.5, self.N )
		e0 = self.gen_normal(N=self.N)
		e1 = self.gen_normal(N=self.N)

		# c0 = v * pk[0] + m + e0 (mod q)
		c0 = v * self.pk[0]
		quo,c0 = c0 // self.fn
		c0 = c0 + m
		c0 = c0 + e0
		c0 = c0 % self.q
		#c0 = self.ring_mod( c0, self.q )

		# c1 = v * pk[1] + e1 (mod q)
		c1 = v * self.pk[1]
		quo,c1 = c1 // self.fn
		c1 = c1 + e1
		c1 = c1 % self.q
		#c1 = self.ring_mod( c1, self.q )

		# calculate canonical inifity norm of m
		vn = self.canonical_inf_norm( cm )

		# calculate Bclean
		Bclean = (8*np.sqrt(2)*self.std*self.N) + (6*self.std*np.sqrt(self.N)) + (16*self.std*np.sqrt(self.h*self.N)) 

		return [ ( c0 , c1 ), self.L, vn, Bclean ]

	def decrypt(self, ct):
		# decrypt ciphertext to plaintext polynomial

		# m = ct[0] + ct[1] * s (mod q)
		m = ct[0][1] * self.sk
		quo,m = m // self.fn
		m = m + ct[0][0]
		#m = m % self.q
		q = (self.p ** ct[1] ) * self.q0 #q = (self.delta ** ct[1] ) * self.q0

		m = self.ring_mod( m, q )

		return m

	def rescale(self, ct):
		# rescale the ciphertext for level l to l-1
		print(f'ct[0][0]: {ct[0][0]}')
		print(f'self.p: {self.p}')
		c0= ct[0][0] // self.p #c0= ct[0][0] / self.delta
		c1= ct[0][1] // self.p #c1= ct[0][1] / self.delta
		c0 = round( c0 )
		c1 = round( c1 )
		l = ct[1] - 1
		q = (self.p ** l) * self.q0 #q = (self.delta ** l) * self.q0
		c0= self.ring_mod( c0, q )
		c1= self.ring_mod( c1, q )

		# calculate v
		v = ct[2] / self.p #v = ct[2] / self.delta

		# calculate b
		bscale = np.sqrt( self.N / 3 ) * ( 3 + (8 * np.sqrt( self.h ) ) )
		b = ( ct[3] / self.p ) + bscale #b = ( ct[3] / self.delta ) + bscale

		return [ (c0, c1), l, v, b ]

	def simple_rescale(self, ct):
		# rescale ciphertext without changing message m
		l = ct[1] - 1
		q = (self.p ** l) * self.q0 #q = (self.delta ** l) * self.q0
		c0 = ct[0][0] % q
		c1 = ct[0][1] % q

		return [ (c0, c1), l, ct[2], ct[3] ]


	def ct_add(self, ct0, ct1):
		# this will return a ciphertext polynomial
		# of the addition of two ciphertexts

		# calculate q ( q[l] )	
		q = (self.p ** ct0[1] ) * self.q0 #q = (self.delta ** ct0[1] ) * self.q0

		c0 = ct0[0][0] + ct1[0][0]
		#c0 = self.ring_mod( c0, q )
		c0 = c0 % q

		c1 = ct0[0][1] + ct1[0][1]
		#c1 = self.ring_mod( c1, q )
		c1 = c1 % q

		# calculate new v
		v = ct0[2] + ct1[2]
		
		# calculate new b
		b = ct0[3] + ct1[3]

		return [ ( c0, c1 ), ct0[1], v, b ]

	def ct_mult(self, ct0, ct1):
		# this will return a ciphertext polynomial
		# of multiplied ciphertexts

		# calculate q ( q[l] )	
		q = (self.p ** ct0[1] ) * self.q0 #q = (self.delta ** ct0[1] ) * self.q0

		c0 = ct0[0][0] * ct1[0][0]
		quo,c0 = c0 // self.fn
		#c0 = self.ring_mod( c0, q )
		c0 = c0 % q

		c1 = (ct0[0][0] * ct1[0][1]) + (ct0[0][1] * ct1[0][0])
		quo,c1 = c1 // self.fn
		#c1 = self.ring_mod( c1, q )
		c1 = c1 % q

		c2 = (ct0[0][1] * ct1[0][1])
		quo,c2 = c2 // self.fn
		#c2 = self.ring_mod( c2, q )
		c2 = c2 % q

		# test - decrypted of the three terms
		m0 = c0
		m1 = c1 * self.sk
		quo,m1 = m1 / self.fn
		m2 = c2 * self.sk
		m2 = m2 * self.sk
		quo,m2 = m2 / self.fn

		m = (m0) + (m1) + (m2) 
		m = self.ring_mod( m, q )
		z = self.decode( m )
		#print( z )

		# relinearize the three terms to two
		r0 = c2 * self.evk[0]
		quo,r0 = r0 // self.fn
		r0 = r0 // self.P
		r0 = round(r0)
		r0 = r0 + c0
		#r0 = self.ring_mod( r0, q )
		r0 = r0 % q

		r1 = c2 * self.evk[1]
		quo,r1 = r1 // self.fn
		r1 = r1 // self.P
		r1 = round(r1)
		r1 = r1 + c1
		#r1 = self.ring_mod( r1, q )
		r1 = r1 % q

		# calculate new v
		v = ct0[2] * ct1[2]
		
		# calculate new b
		bscale = np.sqrt( self.N / 3 ) * ( 3 + (8 * np.sqrt( self.h ) ) )
		bks = 8 * self.std * self.N / np.sqrt( 3 )
		bmult = (bks * q / self.P) + bscale
		b = (ct0[2]*ct1[3]) + (ct0[3]*ct1[2]) + (ct0[3]*ct1[3])

		return [ ( r0, r1 ), ct0[1], v, b]

	def ring_mod(self, p, q=None):
		# mod q with range (-q/2,q/2]
		q = self.q if q == None else q
		p = p.copy()
		p = p % q
		for ind, i in enumerate(p):
			if (i > q/2):
				p[ind] = i - q
		return p

	def canonical_inf_norm(self, p):
		# this function will return the canonical infinity norm
		# of the polynomial p
		p = p.copy()

		sp = self.sigma( p )
		#print( sp )

		mx = 0
		for i in sp:
			mx = max( mx, abs( i ) )

		return mx

	def create_sigma_R_basis(self):
		# this will create a sigma_R_basis matrix
		# which will help transform z to m
		N = self.M // 2
		roots = [ self.xi ** (2 * i + 1) for i in range(N) ]
		v = self.Vandermonde(roots,N)

		# set self.sigma_R_basis as CRT^-1
		self.sigma_R_basis = v.transpose()
		return 

	def Vandermonde(self,X,N):
		# this function will return a MxN Vandermonde matrix when given
		# a vector of x's

		vander = []
		# go through each of the different x's to create arrays
		for x in X:
			row = []

			# create the row [1, x, x^2, ..., x^n-1]
			for i in range(N):
				row.append( x ** i )

			vander.append( row )

		return Matrix(vander)

	def sigma_inverse(self, z):
		# this function will calculate the polynomial equivalent
		# of a vector z with M-th root of unity

		# Create the Vandermonde matrix
		N = self.M // 2
		roots = [ self.xi ** (2 * i + 1) for i in range(N) ]

		A = self.Vandermonde(X=roots,N=N)

		#print( A )
		#print( self.sigma_R_basis.transpose() )

		# solve the linear equation
		coeffs = linalg(A, z)
		#print(coeffs)

		# Turn into polynomial
		poly = Poly( coeffs )

		return poly

	def sigma(self, p):
		# this function will solve the polynomial with
		# the M-th root of unity

		N = self.M // 2
		outputs = []

		for i in range(N):
			x = self.xi ** (2*i + 1)
			out = p.evaluate(x)
			outputs.append(out)

		return Vector( outputs )

	def pi_inverse(self, z):
		# this function will extend the vector with its conjugates
		rev = z.copy()
		rev.reverse()
		for ind, i in enumerate(rev):
			rev[ind] = i.conjugate()

		arr = z + rev
		return Vector( arr )

	def pi(self, z):
		# this function will get half of the elements
		half = len(z)//2
		return Vector( z[:half] )

	def compute_basis_coordinates(self, z):
		# computes the coordinates of vector with respect to orthogonal lattice
		output = []
		for row in self.sigma_R_basis:
			nu = vdot(z, row)
			de = vdot(row, row)

			k = nu/de

			output.append( k.real )
		
		return Vector( output )

	def coordinate_wise_random_rounding(self, coordinates):
		# Rounds coordinates randomly

		r = [i - np.floor(i) for i in coordinates]
		f = [np.random.choice( [c,1-c], 1, p=[1-c,c] ) for c in r]

		f = Vector(f)
		rounded = coordinates - f
		rounded = [ int(coeff.real) for coeff in rounded ]
		return Vector( rounded )


	def sigma_R_discretization(self, z):
		# this function will turn a complex vector z into 
		# the coordinates of simga R

		coords = self.compute_basis_coordinates( z )
		rounds = self.coordinate_wise_random_rounding( coords )
		# change vector to Nx1 matrix
		rounds = Matrix( [rounds] )
		rounds = rounds.transpose()
		ret = matmul( self.sigma_R_basis.transpose(), rounds )
		return ret

	def encode(self, z):
		# this function will encode a complex vector into
		# an integer polynomial

		while ( len(z) < self.N/2 ):
			z.append( 0 )
		
		scaled = self.pi_inverse(z)
		scaled = scaled * self.delta

		R = self.sigma_R_discretization( scaled )

		R = R.transpose()
		R = Vector( R[0] )

		p = self.sigma_inverse( R )

		p = round(p)

		return p 

	def decode(self, p):
		# this function will take an integer polynomial and turn 
		# it back into a complex vector

		scaled_p = p / self.delta

		sig = self.sigma( scaled_p )

		ret = self.pi( sig )

		return ret


def main():

	es = CKKS(M=2**5,delta=2**10,q0=2**15,L=3)

	za = [ 1 + 2j, 3 - 4j ]
	zb = [ 1 + 0j, 0 + 1j ]

	# plaintext vectors
	ma = es.encode( za )
	mb = es.encode( zb )

	#z = es.decode( m )
	#print( z )

	# ciphertext polynomials
	ca = es.encrypt( ma )
	cb = es.encrypt( mb )
	print(f'ca.b = {ca[3]}')
	print(f'cb.b = {cb[3]}')
	print(f'es.delta = {es.delta}')
	print(f'ca.v = {ca[2]}')
	print(f'cb.v = {cb[2]}')

	cc = es.ct_add( ca, cb )
	print(f'cc.v = {cc[2]}')
	print(f'cc.b = {cc[3]}')
	print(' ')
	#cc = es.rescale( cc )
	#print(f'cc.v = {cc[2]}')
	#print(f'cc.b = {cc[3]}')

	mc = es.decrypt( cc )

	zc = es.decode( mc )
	print(f'za: {za}')
	print(f'zb: {zb}')
	print(f'zc: {zc}')

	return


def test():

	es = CKKS()

	N = es.M // 2
	roots = [ es.xi ** (2 * i + 1) for i in range(N) ]
	v = es.Vandermonde(roots,N)
	vp = v.transpose()

	scaled_z = [ 192 + 256j, 128 - 64j, 128 + 64j, 192 - 256j ]

	#print(vp)
	for i in vp:
		print(i)
	print('\n')

	real = []
	for i in range(4):
		nu = vdot(scaled_z, vp[i])
		de = vdot(vp[i], vp[i])
		print(f'nu: {nu}')
		print(f'de: {de}')
		k = nu/de
		print(k)
		real.append( int(k.real) )
	
	print(' ')
	print(real)

	return

if __name__ == '__main__':
  main()

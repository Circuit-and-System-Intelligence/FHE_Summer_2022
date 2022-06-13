#!/usr/bin/env python
# Date: 6/8/2022
# Create a Class to emulate ckks encryption scheme 
# 
# This program will create a class for the ckks 
# fully homomorphic encryption scheme

from poly import Poly
from vector import Vector, Matrix, vdot, matmul, linalg
import numpy as np

class CKKS():

	def __init__(self,M=8,delta=64):
		self.M = M
		self.xi = np.exp( 2 * np.pi * 1j / self.M )
		self.delta = delta
		self.check = 1
		self.create_sigma_R_basis()
		return

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
			out = p.calc(x)
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

	es = CKKS()

	m = es.encode( [ 1 + 2j, 3 - 4j ] )

	print( m )

	z = es.decode( m )

	print( z )

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

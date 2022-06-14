#!/usr/bin/env python
# Date: 6/14/2022
# Create functions to test the CKKS class
# 
# This program will create functions for 
# testing ct_add and ct_mult for ckks

from ckks import CKKS
from poly import Poly
from vector import Vector, Matrix, matmul, vdot

def main():
	test_ct_mult()
	test_ct_add()
	pass

def test_ct_mult():
	print('Testing ct_mult')

	es = CKKS(M=2**3, delta=2**7, q0=2**10, L=3)

	za = [ 1 + 2j, 3 - 4j ]
	zb = [ 1 + 0j, 0 + 1j ]

	# plaintext vectors
	ma = es.encode( za )
	mb = es.encode( zb )

	# ciphertext polynomials
	ca = es.encrypt( ma )
	cb = es.encrypt( mb )

	# ciphertext multiplication and rescaling
	cc = es.ct_mult( ca, cb )
	cc = es.rescale( cc )

	# decryption
	mc = es.decrypt( cc )

	# decoding into complex vector
	zc = es.decode( mc )

	print(f'za: {za}')
	print(f'zb: {zb}')
	print(f'za * zb = zc')
	print(f'zc: {zc}')
	print(' ')

	return

def test_ct_add():
	print('Testing ct_add')

	es = CKKS(M=2**3, delta=2**7, q0=2**10, L=3)

	za = [ 1.2 + 2j, 3 - 4j ]
	zb = [ 2.3 + 2j,-3 + 4j ]

	# plaintext vectors
	ma = es.encode( za )
	mb = es.encode( zb )

	# ciphertext polynomials
	ca = es.encrypt( ma )
	cb = es.encrypt( mb )

	# ciphertext addition
	cc = es.ct_add( ca, cb )

	# decryption
	mc = es.decrypt( cc )

	# decoding into complex vector
	zc = es.decode( mc )

	print(f'za: {za}')
	print(f'zb: {zb}')
	print(f'za + zb = zc')
	print(f'zc: {zc}')
	print(' ')

	return

if __name__ == '__main__':
	main()

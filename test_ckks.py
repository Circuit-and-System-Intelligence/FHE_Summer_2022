#!/usr/bin/env python
# Date: 6/14/2022
# Create functions to test the CKKS class
# 
# This program will create functions for 
# testing ct_add and ct_mult for ckks

from ckks import CKKS
from poly import Poly
# from counter import PolyCount as Poly
from vector import Vector, Matrix, matmul, vdot
import sys

def main():
	test_ct_mult()
	#test_ct_add()
	#test_encrypt()
	pass

def test_encrypt():
	print('Testing encryption')

	es = CKKS(M=2**2, delta=2**7, q0=2**8, h=1, L=1, std=3.2)
	print(f'es.p: {es.p}')
	print(f'es.delta: {es.delta}')

	za = [ 1 + 2j ]

	# plaintext vectors
	ma = es.encode( za )
	print( ma )

	# ciphertext polynomials
	ca = es.encrypt( ma )

	print(f'es.delta = {es.delta}')
	print(f'N+2*Bclean= {es.N + (2*ca[3])}')

	mc = es.decrypt( ca )
	print( mc )

	zc = es.decode( mc )
	print(f'za: {za}')
	print(f'zc: {zc}')
	return


def test():

	es = CKKS()

	return

def test_ct_mult():
	print('Testing ct_mult')

	es = CKKS(M=2**7, delta=2**20, q0=2**25, h=4, L=4)

	za = [ 1 + 2j, 3 - 4j ]
	zb = [ 1.5 + 0j, 0 + 1j ]

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
	print(f'zc: ',end='[')
	print(f'({zc[0].real:.3f}+{zc[0].imag:.3f}j)',end='')
	for ind,c in enumerate(zc):
		if ind == 0:
			continue
		print(f', ({c.real:.3f}+{c.imag:.3f}j)',end='')
	print(f']')
	print(' ')
	es.print_counter_info()

	return

def test_ct_add():
	print('Testing ct_add')

	es = CKKS(M=2**2, delta=2**4, q0=2**14, L=1)
	es.sk = Poly( [0,-1] )
	es.pk = ( Poly([-98, 64]), Poly([64, 98]) )

	za = [ 1.2 + 2j, 3 - 4j ]
	zb = [ 2.3 + 2j,-3 + 4j ]

	# plaintext vectors
	ma = es.encode( za )
	mb = es.encode( zb )
	print('mb')
	print(mb)
	print(' ')

	# ciphertext polynomials
	ca = es.encrypt( ma )
	cb = es.encrypt( mb )

	ca = [ (Poly([-82,92]),Poly([60,99])), ca[1], ca[2], ca[3] ]
	print( 'cb')
	print( cb[0][0] )
	print( cb[0][1] )
	print(' ')

	# ciphertext addition
	cc = es.ct_add( ca, cb )
	print('cc')
	print( cc[0][0] )
	print( cc[0][1] )
	print(' ')

	# decryption
	mc = es.decrypt( cc )
	print( mc )

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

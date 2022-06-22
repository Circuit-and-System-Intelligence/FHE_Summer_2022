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
	#test_ct_mult()
	#test_ct_add()
	test_encrypt()
	pass

def test_encrypt():
	print('Testing encryption')

	es = CKKS(M=2**2, delta=2**10, q0=2**15, h=1, L=2, std=3.2)

	#print(f'es.sk: {es.sk}')
	#print(f'es.pk[0]: {es.pk[0]}')
	#print(f'es.pk[1]: {es.pk[1]}')

	#za = [ 1 + 2j, 3 - 4j ]
	za = [ 1 + 2j ]
	#zb = [ 1 + 0j, 0 + 1j ]

	# plaintext vectors
	ma = es.encode( za )
	#mb = es.encode( zb )

	# ciphertext polynomials
	ca = es.encrypt( ma )
	#print(f'ca[0][0]: {ca[0][0]}')
	#print(f'ca[0][1]: {ca[0][1]}')
	#cb = es.encrypt( mb )
	print(f'es.delta = {es.delta}')
	print(f'N+2*Bclean= {es.N + (2*ca[3])}')

	#cc = es.ct_mult( ca, cb )
	#cc = es.rescale( cc )

	mc = es.decrypt( ca )

	zc = es.decode( mc )
	print(f'za: {za}')
	#print(f'zb: {zb}')
	print(f'zc: {zc}')

	return


def test():

	es = CKKS()

	return

def test_ct_mult():
	print('Testing ct_mult')

	es = CKKS(M=2**6, delta=2**14, q0=2**20, L=1)

	za = [ 1 + 2j, 3 - 4j ]
	zb = [ 1.5 + 0j, 0 + 1j ]
	za = [ 1 + 2j ]
	zb = [ 2 + 1j ]

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

	es = CKKS(M=2**2, delta=2**4, q0=2**4, L=1)
	es.sk = Poly( [0,-1] )
	es.pk = ( Poly([-98, 64]), Poly([64, 98]) )

	#za = [ 1.2 + 2j, 3 - 4j ]
	#zb = [ 2.3 + 2j,-3 + 4j ]
	za = [ 1 + 2j ]
	zb = [ 2 + 1j ]

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

from bfv import BFV
from mont_bfv import Mont_BFV
from poly import Poly
import numpy as np
import sys
import pickle
import pdb

#pdb.set_trace()

def main():
	func = mont_test
	generate_data()
	#mont_test()
	#main_test()
	#test_multiplication()
	#large_q()
	#mult_enc()
	# testing_many_multiplication()
	return

def demo():
	c = 0
	while (c != 'q'):
		print(f'Select a test to run: ')
		print(f'\t-1) Encryption/Decryption')
		print(f'\t-2) Addition')
		print(f'\t-3) Multiplication')
		print(f'\t-4) Base Change')
		print(f'\t-q) Enter \'q\' To Quit')
		c = sys.stdin.readline()
		c = c[0]
		print(' ')
		if c == '1':
			main_test()
		elif c == '2':
			test_addition()
		elif c == '3':
			test_multiplication()
		elif c == '4':
			test_base_change()
		print('\n\n\n')
	print(f'Goodbye!')
	return

def main_test():
	# this main function is using the lpr() class
	# encryption scheme
	
	# q= 2**15, t= 2**8, n=2**4
	es = BFV(q=2**22,n=2**5)

	# generate a plaintext
	pt = np.random.randint(0,2)

	# encrypt the plaintext
	ct = lpr.encrypt(pt)

	# decrypt the ciphertext
	recovered_pt = lpr.decrypt(ct)

	# print the results
	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')
	print(' ')
	lpr.print_counter_info()

	return

def test_addition():
	# this function will act as the test of adding two cipher texts

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**128,n=2**10,std=3.8)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctadd(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text addition',end=': ')
	print(f'{x} + {y} = {answer}')
	print(f'Addition test: { answer == (x^y) }')
	lpr.print_counter_info()
	return 1 if (answer == (x+y)) else 0

def test_multiplication():
	# this function will act as the test of adding two cipher texts

	# q= 2**30, t= 2**1, n=2**10
	# lpr = BFV(t=2,q=2**256,n=2**10,std=2.0)
	lpr = BFV(t=2**3,q=2**15,n=2**5,std=2.0,h=2**1)

	# generate the two random numbers between [0,1]
	#x = 0
	#y = 0
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	x = 2
	y = 3
	print(f'{x} and {y} are randomly generated')
	print(' ')

	# encrypt both ciphertexts
	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)
	print(f'ctx[0] {ctx[0]}')
	print(f'encryping {x}... => {ctx[0].poly[0:4]} {ctx[0].poly[30:32]}')
	print(f'encryping {y}... => {cty[0].poly[0:4]} {cty[0].poly[30:32]}')
	# print(f'{hex(ctx[0][0])}\n{hex(ctx[0][1])}\n{hex(ctx[0][30])}\n{hex(ctx[0][31])}\n')
	# print(f'{hex(cty[0][0])}\n{hex(cty[0][1])}\n{hex(cty[0][30])}\n{hex(cty[0][31])}\n')
	print(' ')

	# multiply the ciphertexts
	ctz = lpr.ctmult(ctx,cty)
	print(f'{ctz[0].poly[0:4]} {ctz[0].poly[30:32]}')
	# print(f'{hex(ctz[0][0])}\n{hex(ctz[0][1])}\n{hex(ctz[0][30])}\n{hex(ctz[0][31])}\n')
	print(' ')

	# decrypt the ciphertext
	answer = lpr.decrypt(ctz)
	answer = answer[0]
	print(f'decrypting {ctz[0][0] % 256}... => {answer}')
	print(' ')

	# print(f'cipher text multiplication',end=': ')
	# print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	# lpr.print_counter_info()
	return 1 if (answer == (x*y)) else 0

def testing_many_multiplication():

	# q= 2**15, t= 2**8, n=2**4
	# lpr = BFV(t=2,n=2**4,q=2**31)
	# lpr = BFV(t=2,q=2**65,n=2**5,h=2**4,std=3.4)
	lpr = BFV(t=2,q=2**100,n=2**3,h=2**2,std=2.0)

	a = 1
	b = 1

	cta = lpr.encrypt(a)
	ctb = lpr.encrypt(b)

	print(f'a: {a}')
	print(' ')

	for i in range(15):
		cta = lpr.ctmult(cta,cta)
		a = lpr.decrypt(cta)
		print(f'i({i}) a: {a}')
		print(' ')
		

	return

def test_func(n=1000,func=None):
	if (func == None):
		return
	
	count = 0
	for i in range(n):
		count += func()
		print(' ')
	
	print(f'\n')
	print(f'The function ran {n} times')
	print(f'{count} successes')
	print(f'{count/n} success rate')

def	demo_counter():
	# this function will act as the test of adding two cipher texts
	print('Small Addition n=2**4')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**38,n=2**4,std=3.8)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctadd(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text addition',end=': ')
	print(f'{x} + {y} = {answer}')
	print(f'Addition test: { answer == (x^y) }')
	lpr.print_counter_info()
	print(' ')
	print(' ')

	# this function will act as the test of adding two cipher texts
	print('Large Addition n=2**10')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**38,n=2**10,std=3.8)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctadd(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text addition',end=': ')
	print(f'{x} + {y} = {answer}')
	print(f'Addition test: { answer == (x^y) }')
	lpr.print_counter_info()
	print(' ')
	print(' ')

	print('Small Multiplication n=2**4')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**38,n=2**4,std=3.8)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	print('wow')
	ctz = lpr.ctmult(ctx,cty)
	print('here')

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	lpr.print_counter_info()
	print(' ')
	print(' ')

	# this function will act as the test of adding two cipher texts
	print('Large Multiplication n=2**10')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**30,n=2**10,std=2.0)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctmult(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	lpr.print_counter_info()
	print(' ')

	return

def small_q():
	# this function will act as the test of multiplying two cipher texts
	print('SMALL Q MULT')
	print('small q = 2**15')
	print('small n = 2**3')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**15,n=2**3,std=2.0)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctmult(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	lpr.print_counter_info()
	print(' ')
	return

def large_q():
	# this function will act as the test of multiplying two cipher texts
	print('LARGE Q MULT')
	print('large q = 2**256')
	print('large n = 2**7')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = BFV()
	lpr = BFV(t=2,q=2**256,n=2**7,std=2.0)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctmult(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	lpr.print_counter_info()
	print(' ')
	return

def generate_data():
	# this function will generate the data for increasing q and increasing d for rings (x^d + 1)

	print('input file name')
	fn = input()

	print('input qmin, qmax, qstep')
	qmin = int(input())
	qmax = int(input())
	qstep = int(input())
	print('input nmin, nmax, nstep')
	nmin = int(input())
	nmax = int(input())
	nstep = int(input())

	with open(f"data/{fn}.csv","w") as f:
		# f.write(f'q,d,enc_add,enc_mul,dec_add,dec_mul,key_add,key_mul\n')
		f.write(f't=2, h=16, std=2, q = 128\n')
		f.write(f'n,q,add,mul\n')
		for d in range(nmin,nmax,nstep):
			for q in range(qmin,qmax,qstep):
				print(f'q={q} d={d}')
				add_list = []
				add_max = 0
				add_min = 2 ** 256
				print('here')
				for j in range(25):
					lpr = BFV(t=2,q=2**q,n=2**d,std=2.0,h=16)
					x = np.random.randint(0,2)
					y = np.random.randint(0,2)
					ctx = lpr.encrypt(x)
					cty = lpr.encrypt(y)
					ctz = lpr.ctadd(ctx,cty)
					z = lpr.decrypt(ctz)[0]
					assert z == x^y

					enc_add = lpr.counters['enc'].add
					enc_mul = lpr.counters['enc'].mul

					dec_add = lpr.counters['dec'].add
					dec_mul = lpr.counters['dec'].mul

					key_add = lpr.counters['key'].add
					key_mul = lpr.counters['key'].mul

					add_add = lpr.counters['mul'].add
					add_mul = lpr.counters['mul'].mul

					add_count = enc_add+dec_add+key_add+add_add
					mul_count = enc_mul+dec_mul+key_mul+add_mul
					add_list.append( add_count )

					if add_count > add_max:
						add_max = add_count
					if add_count < add_min:
						add_min = add_count

				add_avg = sum( add_list ) / 100
				add_std = np.std( add_list )

				f.write(f'{q}, {d}, {add_avg}, {add_std}, {add_max}, {add_min}\n')
				# f.write(f'{q}, {d}, {enc_add+dec_add+key_add+add_add}, {enc_mul+dec_mul+key_mul+add_mul}\n')
				# print(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul}')
				# f.write(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul},{key_add},{key_mul},{add_add},{add_mul}\n')
				# f.write(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul},{key_add},{key_mul}\n')

def mont_test():
	# this function will test montgomery encryption scheme
	
	# q= 2**15, t= 2**8, n=2**4
	lpr = Mont_BFV(q=2**22,t=2,n=2**5,bitwidth=16)

	# generate a plaintext
	pt = np.random.randint(0,2)
	b = np.random.randint(0,2)

	# encrypt the plaintext
	ct = lpr.encrypt(pt)
	ctb = lpr.encrypt( b )

	ctmul = lpr.ctmult( ct, ctb )
	mul = lpr.decrypt( ctmul )
	#mul = pt^b

	# decrypt the ciphertext
	recovered_pt = lpr.decrypt(ct)
	# recovered_pt = pt

	# print the results
	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')
	print(' ')
	print(f'pt*b = mul: {pt}*{b}={mul}')
	print(f'{mul == (pt*b)}')
	lpr.print_counter_info()

	return 1 if pt == recovered_pt else 0

	
def mult_enc():
	# define values for decryption with binary circuit
	Sr = 5
	k = 99
	n = 2 ** 6 # defined as d in bfv paper
	x = 100 # defined as n in bfv paper. q = 2 ^ x

	lprA = BFV(q=2**x,t=2,n=n,h=2**2)
	lprB = BFV(q=2**x,t=2,n=n,h=2**2)
	'''
	lpr = BFV(t=2,q=2**100,n=2**6,h=2**2,std=2.0)
	'''
	# lprA = BFV(q=2**20,t=2,n=2**5)
	# lprB = BFV(q=2**20,t=2,n=2**5)

	a = 1
	a = Poly([1,0,0,1,1,1,0,0])
	one = Poly([1])
	PolyA = a.copy()

	ctA = lprA.encrypt(a)
	ctOne = lprA.encrypt(one)
	ctA = lprA.ctmult( ctA, ctOne )

	# encrypt ctA in encryption B
	# ctA will have the top five bits of each coeff
	# encrypted into different ciphertexts in lprB
	ctB = [ [], [] ]
	for num in ctA[0]:
		ctB[0].append([])
		for i in range(Sr):
			ni = num >> ( x - Sr )
			ni = ni >> i
			npoly = ni & 1
			# npoly = toBinaryPoly( ni )
			ctB[0][-1].append( lprB.encrypt(npoly) )
	for num in ctA[1]:
		ctB[1].append([])
		for i in range(Sr):
			ni = num >> ( x - Sr )
			ni = ni >> i
			npoly = ni & 1
			# npoly = toBinaryPoly( ni )
			ctB[1][-1].append( lprB.encrypt(npoly) )

	# encrypt secret key of A
	skA = []
	for num in lprA.sk:
		skA.append( lprB.encrypt( int(num) ) )

	skB = []
	for num in lprB.sk:
		skB.append( lprA.encrypt( int(num) ) )
	
	# switch from A to B
	a_to_b = testBootStrap( lprB, skA, ctA, n, Sr, x, k )
	# multiply in B
	a_to_b = lprB.ctmult( a_to_b, lprB.encrypt( 1 ) )
	# switch from B to A
	b_to_a = testBootStrap( lprA, skB, a_to_b, n, Sr, x, k )
	# decrypt from A
	dec = lprA.decrypt( b_to_a )

	print(f'dec:   {dec}')
	print(f'PolyA: {PolyA}')

	return

	# now we have the encrypted ciphertext
	# and encrypted secret key

	one_B = lprB.encrypt( 1 )
	print(f'ctB[0]: {len(ctB[0])}')
	print(f'n: {n}')

	bitmat = []
	ctnew = []
	for i in range(n):
		bitmat = []

		# get c0 into bitmat
		bitmat.append( ctB[0][i] ) 

		negadd = 0
		# get c1 * sk into bitmat
		for j in range(n):
			bitmat.append( [] )
			for k in range(Sr):
				app = lprB.ctmult( skA[j], ctB[1][i-j][k] )
				bitmat[-1].append( app )

			if j > i:
				negadd += 1
				for k in range(Sr):
					bitmat[-1][k] = lprB.ctadd( bitmat[-1][k], one_B )

		negadd = toBinaryPoly( negadd )
		# ctneg = lprB.encrypt( negadd )

		bitmat.append( [] )
		for k in range(Sr):
			encn = lprB.encrypt( negadd[k] )
			if i == 0:
				decapp = lprB.decrypt( encn )
				print(f'({k}) decapp: {decapp}')
			bitmat[-1].append( encn )

		if i == 0: 
			print(f'negadd: {negadd}')
			for b in range(n+2):
				dec0 = lprB.decrypt( bitmat[b][0] )[0]
				dec1 = lprB.decrypt( bitmat[b][1] )[0]
				dec2 = lprB.decrypt( bitmat[b][2] )[0]
				dec3 = lprB.decrypt( bitmat[b][3] )[0]
				dec4 = lprB.decrypt( bitmat[b][4] )[0]
				s = 0
				s += (2 ** 0) * dec0
				s += (2 ** 1) * dec1
				s += (2 ** 2) * dec2
				s += (2 ** 3) * dec3
				s += (2 ** 4) * dec4
				print(f'\t{dec0} {dec1} {dec2} {dec3} {dec4} - {s}')

		# sum bitmat 
		for j in range( n ):
			r0 = [0]*Sr
			r1 = [0]*Sr
			for k in range(Sr):
				r0[k] = lprB.ctadd( bitmat[0][k], bitmat[1][k] )
				r0[k] = lprB.ctadd( r0[k], bitmat[2][k] )

				if k == 0:
					r1[k] = lprB.encrypt(0)
				else:
					a = bitmat[0][k-1]
					b = bitmat[1][k-1]
					c = bitmat[2][k-1]

					a = lprB.ctadd( a, one_B )
					b = lprB.ctadd( b, one_B )
					c = lprB.ctadd( c, one_B )

					ab = lprB.ctmult( a, b )
					bc = lprB.ctmult( b, c )
					ac = lprB.ctmult( a, c )

					r1[k] = lprB.ctadd( ab, bc )
					r1[k] = lprB.ctadd( r1[k], ac )
					r1[k] = lprB.ctadd( r1[k], one_B )

			bitmat[0] = r0
			bitmat.append( r1 )
			bitmat.pop(1)
			bitmat.pop(1)

		# final sum
		finalbits = []
		finalbits.append( lprB.ctadd( bitmat[0][0], bitmat[1][0] ) ) 
		for j in range(Sr-1):
			carrybit = lprB.ctmult( bitmat[0][j], bitmat[1][j] )
			appendbit = lprB.ctadd( bitmat[0][j+1], bitmat[1][j+1] )
			appendbit = lprB.ctadd( appendbit, carrybit )
			finalbits.append( appendbit )

		if i == 0:
			dec0 = lprB.decrypt( finalbits[0] )[0]
			dec1 = lprB.decrypt( finalbits[1] )[0]
			dec2 = lprB.decrypt( finalbits[2] )[0]
			dec3 = lprB.decrypt( finalbits[3] )[0]
			dec4 = lprB.decrypt( finalbits[4] )[0]
			s = 0
			s += (2 ** 0) * dec0
			s += (2 ** 1) * dec1
			s += (2 ** 2) * dec2
			s += (2 ** 3) * dec3
			s += (2 ** 4) * dec4
			print('sum')
			print(f'\t{dec0} {dec1} {dec2} {dec3} {dec4} - {s}')

		# append bit to ctnew
		m = lprB.ctadd( finalbits[Sr-1], finalbits[Sr-2] )
		ctnew.append( m )

	# reconstruct ciphertext in B
	zero = lprB.encrypt( 0 )

	for i in range(n):
		pow2 = 2 ** i
		pow2 = toBinaryPoly( pow2 )
		ctpow2 = lprB.encrypt( pow2 )
		add = lprB.ctmult( ctnew[i], ctpow2 )
		zero = lprB.ctadd( zero, add )
	
	decz = lprB.decrypt( zero )
	print(f'decz:    {decz}')

	recPoly = []
	for dct in ctnew:
		recPoly.append( lprB.decrypt( dct )[0] )

	recPoly = Poly( recPoly )
	print(f'recPoly: {recPoly}')
	print(f'PolyA:   {PolyA}')
	print(f'recPoly == PolyA: {recPoly==PolyA}')
		
	# return

	'''
	Plaintext manual decrypt
	'''
	print('\n')
	a = PolyA

	Sr = 5
	k = 99
	n = 2 ** 3 # defined as d in bfv paper
	x = 100 # defined as n in bfv paper. q = 2 ^ x

	'''
	# decrypt ctB from encryption B
	for ind,c in enumerate(ctB[0]):
		dec = lprB.decrypt( c )
		ctB[0][ind] = fromBinaryPoly( dec )
	for ind,c in enumerate(ctB[1]):
		dec = lprB.decrypt( c )
		ctB[1][ind] = fromBinaryPoly( dec )

	ctB[0] = Poly( ctB[0] )
	ctB[1] = Poly( ctB[1] )

	newa = lprA.decrypt(ctB)
	'''

	# manual decryption
	d0 = ctA[0].copy()
	d1 = ctA[1].copy()

	# right shift
	# 1 Level
	for ind, i in enumerate(d0):
		d0[ind] = i >> (x - Sr)

	for ind, i in enumerate(d1):
		d1[ind] = i >> (x - Sr)


	mandec = d1 * lprA.sk
	quo, mandec = mandec // lprA.fn
	mandec += d0
	mandec = mandec % (2**Sr)
	# print(f'mandec: {mandec}')

	# "create" matrix (a list) of integers for each coefficient
	# start with only first coefficient for now
	sumints = []
	bind = 0
	sumints.append( d0[bind] )
	# print('appendints: ')
	negadd = 0
	for i in range(n):
		appendint = lprA.sk[i] * d1[bind-i]
		# print(f'\t {appendint:2} = {lprA.sk[i]} * {d1[n-i-1]}')
		# print('\t', appendint )
		#appendint = appendint & ( (2**Sr) - 1 )

		if i > (bind):
			appendint = ( ((2**Sr)-1 ) ^ appendint )
			negadd += 1

		sumints.append( appendint )
	sumints.append( negadd & ((2**Sr)-1) )
	# sum the list using just sum for now
	w = sum( sumints )
	for ssint in sumints:
		print(f'\t{ssint}')
	# sumints = [ 9, 5, 17 ]
	for ie in range( n ):
		i = 0
		r0 = 0
		r1 = 0
		for j in range(Sr):
			bitpos = 2 ** j
			ai = sumints[i] & bitpos
			bi = sumints[i+1] & bitpos
			ci = sumints[i+2] & bitpos
			r0 += ( ai ^ bi ^ ci )

			if j == 0:
				bitr = 1
				ac = 1
				bc = 1
				cc = 1
			else:
				bitr = bitpos >> 1
				ac = sumints[i] & bitr
				bc = sumints[i+1] & bitr
				cc = sumints[i+2] & bitr

				ac = ac ^ bitr
				bc = bc ^ bitr
				cc = cc ^ bitr

				ac = ac << 1
				bc = bc << 1
				cc = cc << 1

			a_b = ac & bc
			b_c = bc & cc
			a_c = ac & cc
		
			r1 += bitpos ^ ( a_b ^ b_c ^ a_c )

		sumints[0] = r0
		sumints.pop(1)
		sumints.pop(1)
		sumints.append(r1)
		#sumints[1] = r0
		# sumints.pop(2)

	manw = sum(sumints)
	print(f'manw: {manw}')

	print(f'w: {w}')
	
	w = manw
	# keep it mod Sr
	w = w & ( (2**Sr) - 1 )
	print(f'w: {w}')

	# get wb
	wb = w & ( 2 ** (k - x + Sr - 1) )
	wb = wb >> ( k - x + Sr - 1)
	print(f'wb: {wb}')

	# get m0
	m0 = w >> ( k - x + Sr )
	m0 = m0 ^ wb

	print(f'm0: {m0}')

	newa = lprA.decrypt(ctA)
	'''
	'''

	print(' ')
	print(f'a: {a}')
	print(f'newa: {newa}')

	print(f'a == newa: {a == newa}')

	return 1 if newa == a else 0

def toBinaryPoly(x: int):
	pow2 = 1
	i = 0
	p = []
	while pow2 <= x:
		p.append( (x & pow2) >> (i) )
		pow2 = pow2 << 1
		i += 1
	
	return Poly( p )

def fromBinaryPoly(p: Poly):
	ret = 0
	for ind,i in enumerate(p):
		ret += (2**ind) * i
	return ret 

def ctBitSwap(ct, i, j):
	t = ct[0][i]
	ct[0][i] = ct[0][j]
	ct[0][j] = t

	temp = ct[1][i]
	ct[1][i] = ct[1][j]
	ct[1][j] = temp
	return 

def testBootStrap( lpr, skEnc, ct, n, Sr, x, k ):
	# this is a test of bootstrapping

	# encrypt ct in encryption lpr
	# ct will have the top five bits of each coeff
	# encrypted into different ciphertexts in lpr
	ctB = [ [], [] ]
	for num in ct[0]:
		ctB[0].append([])
		for i in range(Sr):
			ni = num >> ( x - Sr )
			ni = ni >> i
			npoly = ni & 1
			# npoly = toBinaryPoly( ni )
			ctB[0][-1].append( lpr.encrypt(npoly) )
	for num in ct[1]:
		ctB[1].append([])
		for i in range(Sr):
			ni = num >> ( x - Sr )
			ni = ni >> i
			npoly = ni & 1
			# npoly = toBinaryPoly( ni )
			ctB[1][-1].append( lpr.encrypt(npoly) )

	# encrypt secret key of A
	skA = skEnc
	
	# now we have the encrypted ciphertext
	# and encrypted secret key
	one_B = lpr.encrypt( 1 )

	bitmat = []
	ctnew = []
	for i in range(n):
		bitmat = []

		# get c0 into bitmat
		bitmat.append( ctB[0][i] ) 

		negadd = 0
		# get c1 * sk into bitmat
		for j in range(n):
			bitmat.append( [] )
			for k in range(Sr):
				app = lpr.ctmult( skA[j], ctB[1][i-j][k] )
				bitmat[-1].append( app )

			if j > i:
				negadd += 1
				for k in range(Sr):
					bitmat[-1][k] = lpr.ctadd( bitmat[-1][k], one_B )

		negadd = toBinaryPoly( negadd )
		# ctneg = lpr.encrypt( negadd )

		bitmat.append( [] )
		for k in range(Sr):
			encn = lpr.encrypt( negadd[k] )
			if i == 0:
				decapp = lpr.decrypt( encn )
				print(f'({k}) decapp: {decapp}')
			bitmat[-1].append( encn )

		if i == 0: 
			print(f'negadd: {negadd}')
			for b in range(n+2):
				dec0 = lpr.decrypt( bitmat[b][0] )[0]
				dec1 = lpr.decrypt( bitmat[b][1] )[0]
				dec2 = lpr.decrypt( bitmat[b][2] )[0]
				dec3 = lpr.decrypt( bitmat[b][3] )[0]
				dec4 = lpr.decrypt( bitmat[b][4] )[0]
				s = 0
				s += (2 ** 0) * dec0
				s += (2 ** 1) * dec1
				s += (2 ** 2) * dec2
				s += (2 ** 3) * dec3
				s += (2 ** 4) * dec4
				print(f'\t{dec0} {dec1} {dec2} {dec3} {dec4} - {s}')

		# sum bitmat 
		for j in range( n ):
			r0 = [0]*Sr
			r1 = [0]*Sr
			for k in range(Sr):
				r0[k] = lpr.ctadd( bitmat[0][k], bitmat[1][k] )
				r0[k] = lpr.ctadd( r0[k], bitmat[2][k] )

				if k == 0:
					r1[k] = lpr.encrypt(0)
				else:
					a = bitmat[0][k-1]
					b = bitmat[1][k-1]
					c = bitmat[2][k-1]

					a = lpr.ctadd( a, one_B )
					b = lpr.ctadd( b, one_B )
					c = lpr.ctadd( c, one_B )

					ab = lpr.ctmult( a, b )
					bc = lpr.ctmult( b, c )
					ac = lpr.ctmult( a, c )

					r1[k] = lpr.ctadd( ab, bc )
					r1[k] = lpr.ctadd( r1[k], ac )
					r1[k] = lpr.ctadd( r1[k], one_B )

			bitmat[0] = r0
			bitmat.append( r1 )
			bitmat.pop(1)
			bitmat.pop(1)

		# final sum
		finalbits = []
		finalbits.append( lpr.ctadd( bitmat[0][0], bitmat[1][0] ) ) 
		for j in range(Sr-1):
			carrybit = lpr.ctmult( bitmat[0][j], bitmat[1][j] )
			appendbit = lpr.ctadd( bitmat[0][j+1], bitmat[1][j+1] )
			appendbit = lpr.ctadd( appendbit, carrybit )
			finalbits.append( appendbit )

		if i == 0:
			dec0 = lpr.decrypt( finalbits[0] )[0]
			dec1 = lpr.decrypt( finalbits[1] )[0]
			dec2 = lpr.decrypt( finalbits[2] )[0]
			dec3 = lpr.decrypt( finalbits[3] )[0]
			dec4 = lpr.decrypt( finalbits[4] )[0]
			s = 0
			s += (2 ** 0) * dec0
			s += (2 ** 1) * dec1
			s += (2 ** 2) * dec2
			s += (2 ** 3) * dec3
			s += (2 ** 4) * dec4
			print('sum')
			print(f'\t{dec0} {dec1} {dec2} {dec3} {dec4} - {s}')

		# append bit to ctnew
		m = lpr.ctadd( finalbits[Sr-1], finalbits[Sr-2] )
		ctnew.append( m )

	# reconstruct ciphertext in B
	zero = lpr.encrypt( 0 )

	for i in range(n):
		pow2 = 2 ** i
		pow2 = toBinaryPoly( pow2 )
		ctpow2 = lpr.encrypt( pow2 )
		add = lpr.ctmult( ctnew[i], ctpow2 )
		zero = lpr.ctadd( zero, add )
	
	decz = lpr.decrypt( zero )

	return zero

if __name__ == '__main__':
  main()

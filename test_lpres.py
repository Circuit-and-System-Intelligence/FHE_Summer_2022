from lpres import LPR
from poly import Poly
import numpy as np
import sys
import pickle

def main():
	func = test_multiplication
	#small_q()
	#large_q()
	#generate_data()
	#test_addition()
	testing_many_multiplication()
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
	lpr = LPR(q=2**215,T=4,t=2,n=2**10)

	# generate a plaintext
	#pt = 5
	#pt = np.random.randint(0,50)
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
	#lpr = LPR()
	lpr = LPR(t=2,q=2**128,n=2**10,T=4,std=3.8)

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
	lpr = LPR(t=2,q=2**38,n=2**10,T=4,std=3.8)

	# generate the two random numbers between [0,1]
	#x = 0
	#y = 0
	x = np.random.randint(0,2)
	y = np.random.randint(0,2)
	print(f'{x} and {y} are randomly generated')

	# encrypt both ciphertexts
	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	# multiply the ciphertexts
	ctz = lpr.ctmult(ctx,cty)

	# decrypt the ciphertext
	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	print( lpr.opcount )
	return 1 if (answer == (x*y)) else 0

def test_base_change():
	# this function will test the base changing 
	# function in the lpr() class

	# will first test this with the base as 10
	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR(T=10)

	print(f'Public Key[0]:',end='\t')
	lpr.pk[0].polyprint()

	base = lpr.poly_base_change(lpr.pk[0],lpr.q,lpr.T)

	print(f'New polynomials for base change')
	for p in base:
		p.polyprint()

	# recreate the original polynomial using the 
	# power of the different base change

	new = Poly()

	for ind,p in enumerate(base):
		cpy = p.copy()
		for jnd,c in enumerate(cpy):
			cpy[jnd] = c * (lpr.T ** ind)

		new = new + cpy

	print(f'Recovered:', end='\t')
	#new = lpr.mod(new)
	new.polyprint()

	print(f'{new == lpr.pk[0]}')
	#print(f'q = {lpr.q}')
	return

def testing_many_multiplication():

	# q= 2**15, t= 2**8, n=2**4
	# lpr = LPR(t=2,n=2**4,q=2**31)
	lpr = LPR(t=2,q=2**18,n=2**5,T=4,std=1.2)

	a = 1
	b = 0

	cta = lpr.encrypt(a)
	ctb = lpr.encrypt(b)

	ctc = lpr.ctmult(cta,ctb)

	c = lpr.decrypt(ctc)

	print(f'a: {a}')
	print(f'b: {b}')
	print(f'c: {c}')

	return

	for i in range(b):
		ctbase = lpr.ctmult(ctbase,cta)
		ans = lpr.decrypt(ctbase)
		print(ans)
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


def test_lpr_calc():

	lpr = LPR(q=2**8, n = 2 ** 3, t = 2, T = 4)

	# set the keys to presets already
	lpr.sk = Poly([1,0,1,1,0,0,0,1])

	# testing if pk is generated correctly
	lpr.gen_pk(test=1)

	assert (lpr.pk[0] == Poly([210,33,97,33,153,141,42,228]))
	assert (lpr.pk[1] == Poly([71,239,2,243,73,213,85,184]))

	# testing if rlk is generated correctly
	#lpr.gen_rlk1(test=1)
	lpr.gen_rlk2()

	assert (lpr.rlk[0][0] == Poly([229,32,80,76,13,63,8,250]))
	assert (lpr.rlk[1][0] == Poly([210,207,198,84,244,230,10,229]))
	assert (lpr.rlk[2][0] == Poly([101,87,32,146,42,145,229,182]))
	assert (lpr.rlk[3][0] == Poly([10,0,194,146,147,73,148,113]))
	assert (lpr.rlk[3][1] == Poly([212,77,255,165,216,115,221,239]))
	assert (lpr.rlk[4][0] == Poly([168,14,165,215,236,76,88,214]))

	# testing encryption
	m0 = 1
	ct0 = lpr.encrypt(pt=m0,test=0)

	assert (ct0[0] == Poly([218,92,152,210,154,112,15,132]) )
	assert (ct0[1] == Poly([59,8,100,216,230,240,159,104]) )

	m1 = 1
	ct1 = lpr.encrypt(pt=m1,test=1)

	assert (ct1[0] == Poly([86,187,214,131,132,35,46,235]) )
	assert (ct1[1] == Poly([71,108,58,210,15,61,112,130]) )

	# test addition

	ct2 = lpr.ctadd(ct0,ct1)
	assert (ct2[0] == Poly([48,23,110,85,30,147,61,111]) )
	assert (ct2[1] == Poly([130,116,158,170,245,45,15,234]) )
	#madd = lpr.decrypt(ct2)
	#print(madd)

	# test mult

	ct3 = lpr.ctmult(ct0,ct1,test=1)
	#assert (ct3[0] == Poly([126,232,169,195,171,35,195,110]) )
	assert (ct3[0] == Poly([116,236,168,194,182,26,182,113]) )
	assert (ct3[1] == Poly([162,154,155,6,193,114,61,171]) )
	mmult = lpr.decrypt(ct3)
	print(mmult)

	
	c0 = Poly([142,144,235,21,224,118,152,123])
	c1 = Poly([40,189,27,73,242,152,98,40])
	c2 = Poly([22,86,133,29,199,110,199,50])

	check1 = c0 + ( c1 * lpr.sk ) + ( c2 * lpr.sk * lpr.sk )
	quo,check1 = check1 / lpr.fn
	check1 = check1 % ( 2 ** 8 )

	check2 = ct3[0] + ( ct3[1] * lpr.sk )
	quo,check2 = check2 / lpr.fn
	check2 = check2 % ( 2 ** 8 )

	check1.polyprint()
	check2.polyprint()

	return


def test_manual_decrypt():

	sk = Poly([1,0,1,1,0,0,0,1])
	t = 2
	q = 256
	ct0 = ( Poly([218, 92, 152, 210, 154, 112, 15, 132]), Poly([59, 8, 100, 216, 230, 240, 159, 104]) )
	ct1 = ( Poly([86,187,214,131,132,35,46,235]), Poly([71,108,58,210,15,61,112,130]) )
	fn = Poly([1,0,0,0,0,0,0,0,1])

	c0 = ct0[0] * ct1[0]
	quo,c0 = c0 / fn
	c0 = c0 * ( t / q )
	c0.round()
	c0 = c0 % q

	c1 = (ct0[0] * ct1[1]) + (ct0[1] * ct1[0])
	quo,c1 = c1 / fn
	c1 = c1 * ( t/q )
	c1.round()
	c1 = c1 % q

	c2 = ct0[1] * ct1[1]
	quo,c2 = c2 / fn
	c2 = c2 * ( t / q )
	c2.round()
	c2 = c2 % q

	c0.polyprint()
	c1.polyprint()
	c2.polyprint()
	print(' ')


	dec = (c0) + (c1 * sk) + (c2 * sk * sk)
	dec.polyprint()
	print(' ')
	quo,dec = dec / fn
	dec.polyprint()
	print(' ')
	dec = dec % 256
	dec.polyprint()
	print(' ')
	dec = dec / 128
	dec.polyprint()
	print(' ')
	dec.round()
	dec.polyprint()
	print(' ')

	return

def	demo_counter():
	# this function will act as the test of adding two cipher texts
	print('Small Addition n=2**4')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = LPR()
	lpr = LPR(t=2,q=2**38,n=2**4,T=4,std=3.8)

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
	#lpr = LPR()
	lpr = LPR(t=2,q=2**38,n=2**10,T=4,std=3.8)

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
	#lpr = LPR()
	lpr = LPR(t=2,q=2**38,n=2**4,T=4,std=3.8)

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
	#lpr = LPR()
	lpr = LPR(t=2,q=2**30,n=2**10,T=4,std=2.0)

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
	#lpr = LPR()
	lpr = LPR(t=2,q=2**15,n=2**3,T=4,std=2.0)

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
	print('large q = 2**20')
	print('large n = 2**3')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = LPR()
	lpr = LPR(t=2,q=2**20,n=2**3,T=4,std=2.0)

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

	with open("bfv_data.csv","w") as f:
		f.write(f'q,d,enc_add,enc_mul,dec_add,dec_mul,key_add,key_mul\n')
		for q in range(32,33):
			for d in range(3,11):
				lpr = LPR(t=2,q=2**q,n=2**d,std=2.0)
				#for n in range(1000):
				x = np.random.randint(0,2)
				# y = np.random.randint(0,2)
				ctx = lpr.encrypt(x)
				# cty = lpr.encrypt(y)
				# ctz = lpr.ctadd(ctx,cty)
				z = lpr.decrypt(ctx)
				assert z == x

				enc_add = lpr.counters['enc'].add
				enc_mul = lpr.counters['enc'].mul
				enc_mod = lpr.counters['enc'].mod

				dec_add = lpr.counters['dec'].add
				dec_mul = lpr.counters['dec'].mul
				dec_mod = lpr.counters['dec'].mod

				key_add = lpr.counters['key'].add
				key_mul = lpr.counters['key'].mul
				key_mod = lpr.counters['key'].mod

				add_add = lpr.counters['add'].add
				add_mul = lpr.counters['add'].mul
				add_mod = lpr.counters['add'].mod

				# print(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul}')
				# f.write(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul},{key_add},{key_mul},{add_add},{add_mul}\n')
				f.write(f'{q},{d},{enc_add},{enc_mul},{dec_add},{dec_mul},{key_add},{key_mul}\n')


if __name__ == '__main__':
  main()

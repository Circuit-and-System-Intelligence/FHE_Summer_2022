from bfv import LPR
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
	#test_multiplication()
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
	lpr = LPR(t=2,q=2**60,n=2**11,T=4,std=2.0)

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

def testing_many_multiplication():

	# q= 2**15, t= 2**8, n=2**4
	# lpr = LPR(t=2,n=2**4,q=2**31)
	lpr = LPR(t=2,q=2**65,n=2**5,T=4,std=3.4)

	a = 1
	b = 0

	cta = lpr.encrypt(a)

	print(f'a: {a}')
	print(' ')

	for i in range(10):
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
	print('large q = 2**256')
	print('large n = 2**10')

	# q= 2**15, t= 2**8, n=2**4
	#lpr = LPR()
	lpr = LPR(t=2,q=2**256,n=2**10,std=2.0)

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

	with open("data/bfv_std.csv","w") as f:
		# f.write(f'q,d,enc_add,enc_mul,dec_add,dec_mul,key_add,key_mul\n')
		f.write(f'n,q,enc\n')
		for d in range(10,11):
			for q in range(16,257,16):
				print(f'q={q} d={d}')
				lpr = LPR(t=2,q=2**q,n=2**d,std=2.0)
				skip = False
				for n in range(100):
					x = np.random.randint(0,2)
					# y = np.random.randint(0,2)
					ctx = lpr.encrypt(x)
					# cty = lpr.encrypt(y)
					# ctz = lpr.ctmult(ctx,cty)
					z = lpr.decrypt(ctx)
					if (z != x):
						f.write(f'{d},{q},False\n')
						skip = True
						break
					#assert z == x

				if skip:
					continue

				f.write(f'{d},{q},True\n')
				continue
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

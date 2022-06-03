from lpres import LPR
from poly import Poly
import numpy as np
import sys

def main():
	#main_test()
	#test_addition()
	#test_multiplication()
	#test_two_mult()
	#testing_many_addition()
	demo()
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
	lpr = LPR()

	# generate a plaintext
	#pt = 5
	pt = np.random.randint(0,50)

	# encrypt the plaintext
	ct = lpr.encrypt(pt)

	# decrypt the ciphertext
	recovered_pt = lpr.decrypt(ct)

	# print the results
	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')

	return

def test_addition():
	# this function will act as the test of adding two cipher texts

	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR()

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,50)
	y = np.random.randint(0,50)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctadd(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text addition',end=': ')
	print(f'{x} + {y} = {answer}')
	print(f'Addition test: { answer == (x+y) }')
	return 1 if (answer == (x+y)) else 0
	return

def test_multiplication():
	# this function will act as the test of adding two cipher texts

	#lpr = LPR(q=2**14,t=2**5,T=5)
	#lpr = LPR(n=2**7,t=2**4,q=2**20)
	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR(t=2**5,n=2**0,T=2**4)

	# generate the two random numbers
	#x = 0
	#y = 0
	x = np.random.randint(0,6)
	y = np.random.randint(0,6)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctmult(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication',end=': ')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	#print(f'x == dec(enc(x)): {x == lpr.decrypt(ctx)}')
	#print(f'y == dec(enc(y)): {y == lpr.decrypt(cty)}')
	#print(f'x+y == dec(enc(x+y)): {x+y == lpr.decrypt(lpr.ctadd(ctx,cty))}')
	return 1 if (answer == (x*y)) else 0

def test_base_change():
	# this function will test the base changing 
	# function in the lpr() class

	# will first test this with the base as 10
	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR(T=5)

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

def test_two_mult():

	#lpr = LPR(t=2**10,q=2**19,T=2*2)
	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR(t=2**5,n=2**0)
	print(' ')

	print("testing low multiplication")
	a = 0
	b = 0
	cta = lpr.encrypt(a)
	ctb = lpr.encrypt(b)

	ctc = lpr.ctmult(cta,ctb)
	ctd = lpr.ctadd(cta,ctb)

	#lpr.decrypt3(ctc3)
	c = lpr.decrypt(ctc)
	d = lpr.decrypt(ctd)
	print(f'{a} * {b} = {c}')
	print(f'{(a*b)==c}')
	print(f'{a} + {b} = {d}')
	print(f'{(a+b)==d}')

	print('\ntesting high multiplication')
	a = 5
	b = 5
	cta = lpr.encrypt(a)
	ctb = lpr.encrypt(b)

	ctc = lpr.ctmult(cta,ctb)
	ctd = lpr.ctadd(cta,ctb)

	#lpr.decrypt3(ctc3)
	c = lpr.decrypt(ctc)
	d = lpr.decrypt(ctd)
	print(f'{a} * {b} = {c}')
	print(f'{(a*b)==c}')
	print(f'{a} + {b} = {d}')
	print(f'{(a+b)==d}')
	return

def testing_many_addition():

	# q= 2**15, t= 2**8, n=2**4
	lpr = LPR(t=2**7,n=2**1)

	a = 2
	b = 40
	base = 0

	cta = lpr.encrypt(a)
	ctbase = lpr.encrypt(base)

	for i in range(b):
		ctbase = lpr.ctadd(ctbase,cta)
		ans = lpr.decrypt(ctbase)
		print(ans)

	return


if __name__ == '__main__':
  main()
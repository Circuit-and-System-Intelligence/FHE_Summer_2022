from lpres import LPR
from poly import Poly
import numpy as np
import sys

def main():
  #main_test()
  #test_base_change()
  #test_multiplication()
  test_two_mult()
  return

def main_test():
	# this main function is using the lpr() class
	# encryption scheme

	test_addition()
	print('\n')
	test_multiplication()
	print('\n')
	#lpr_test()
	#ring_test()
	
	n = 2**4
	q = 2**15
	t = 2**8
	lpr = LPR(q=q,t=t,n=n)

	#lpr.sk.polyprint()
	#lpr.pk[0].polyprint()
	#lpr.pk[1].polyprint()
	#print( lpr.pk )

	pt = 5
	#pt = np.random.randint(0,50)

	ct = lpr.encrypt(pt)


	#ct[0].polyprint()
	#ct[1].polyprint()

	recovered_pt = lpr.decrypt(ct)

	print(f'original pt: {pt}\trecovered pt: {recovered_pt}')
	print(f'{pt==recovered_pt}')

	return

def test_addition():
	# this function will act as the test of adding two cipher texts

	lpr = LPR(t=2 ** 5)

	# generate the two random numbers
	#x = 1
	#y = 2
	x = np.random.randint(0,10)
	y = np.random.randint(0,10)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctadd(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text addition')
	print(f'{x} + {y} = {answer}')
	print(f'Addition test: { answer == (x+y) }')
	if (answer == (x+y) ):
		return 1
	else:
		return 0
	return

def test_multiplication():
	# this function will act as the test of adding two cipher texts

	#lpr = LPR(q=2**14,t=2**5,T=5)
	#lpr = LPR(n=2**7,t=2**4,q=2**20)
	lpr = LPR()

	# generate the two random numbers
	x = 0
	y = 0
	#x = np.random.randint(1,3)
	#y = np.random.randint(1,3)
	print(f'{x} and {y} are randomly generated')

	ctx = lpr.encrypt(x)
	cty = lpr.encrypt(y)

	ctz = lpr.ctmult(ctx,cty)

	answer = lpr.decrypt(ctz)

	print(f'cipher text multiplication')
	print(f'{x} * {y} = {answer}')
	print(f'Multiplication test: { answer == (x*y) }')
	print(f'x == dec(enc(x)): {x == lpr.decrypt(ctx)}')
	print(f'y == dec(enc(y)): {y == lpr.decrypt(cty)}')
	print(f'x+y == dec(enc(x+y)): {x+y == lpr.decrypt(lpr.ctadd(ctx,cty))}')
	return

def test_base_change():
	# this function will test the base changing 
	# function in the lpr() class

	# will first test this with the base as 10
	lpr = LPR(T=2)

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
	new = lpr.mod(new)
	new.polyprint()

	print(f'{new == lpr.pk[0]}')

	print(f'q = {lpr.q}')

	return

def test_two_mult():

	#lpr = LPR(t=2**10,q=2**19,T=2*2)
	lpr = LPR()
	print(' ')

	for i in range(10):

		print("testing low multiplication")
		a = 0
		b = 0
		cta = lpr.encrypt(a)
		ctb = lpr.encrypt(b)

		ctc = lpr.ctmult(cta,ctb)

		c = lpr.decrypt(ctc)
		print(f'{a} * {b} = {c}')
		print(f'{(a*b)==c}')

		print('\ntesting high multiplication')
		a = 8
		b = 8
		cta = lpr.encrypt(a)
		ctb = lpr.encrypt(b)

		ctc = lpr.ctmult(cta,ctb)

		c = lpr.decrypt(ctc)
		print(f'{a} * {b} = {c}')
		print(f'{(a*b)==c}')
	return


if __name__ == '__main__':
  main()
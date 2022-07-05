#!/usr/bin/env python
# Date: 6/22/2022
# Create a Class to count all of the operations in ckks and lpres
# 
# This file will hold a class to count all of the operations used
# when running programs

from poly import Poly
from naive_modulus import naive_modulus
from naive_modulus import barrett

class OperationsCounter():

	def __init__(self,bitwidth=32):
		self.add = 0
		# self.sub = 0
		self.mul = 0
		self.div = 0
		self.mod = 0
		self.bitwidth = bitwidth
		self.addbits = {}
		self.mulbits = {}
		self.divbits = {}
		self.modbits = {}
		return

	# count a simple addition between two 
	# numbers, e.g. float or int
	def num_add(self, x, y):
		# self.add += 1
		self.add += ( (x+y).bit_length() // self.bitwidth ) + 1
		self.append_addbits(x+y)
		return x + y

	# count a simple subtraction between two
	# numbers, e.g. float or int
	def num_sub(self, x, y):
		# self.add += 1
		self.add += ( (x-y).bit_length() // self.bitwidth ) + 1
		self.append_addbits(x-y)
		return x - y

	# count a simple multiplication between two
	# numbers, e.g. float or int
	def num_mul(self, x, y):
		# self.mul += 1
		self.mul += ( (x*y).bit_length() // self.bitwidth ) + 1
		self.append_mulbits(x*y)
		return x * y

	# count a simple division between two
	# numbers, e.g. float or int
	def true_div(self, x, y):
		self.div += 1
		return x / y

	# count a simple division between two
	# numbers, e.g. float or int
	def floor_div(self, x, y):
		# self.div += 1
		self.div += ( (x//y).bit_length() // self.bitwidth ) + 1
		self.append_divbits(x//y)
		return x // y

	# count a simple modulus between two
	# numbers, e.g. float or int
	def num_mod(self, x, y):
		#return self.naive_modulus_count(x,y)
		return self.barrett_count(x,y)
		'''
		self.mod += 1
		return x % y
		'''

	# count operations of adding number to polynomial
	def poly_add_num(self, p, c):
		# self.add += len( p )
		ret = p + c
		for r in ret:
			self.add += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_addbits( r )
		return p + c

	# count operations of subtracting number to polynomial
	def poly_sub_num(self, p, c):
		# self.add += len( p )
		ret = p - c
		for r in ret:
			self.add += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_addbits( r )
		return p - c

	# count operations of multiplying number to polynomial
	def poly_mul_num(self, p, c):
		# self.mul += len( p )
		ret = p * c
		for r in ret:
			self.mul += ( (int(r)).bit_length() // self.bitwidth ) + 1
			self.append_mulbits( int(r) )
		return p * c

	# count operations of dividing number to polynomial
	def poly_div_num(self, p, c):
		self.div += len( p )
		ret = p // c
		for r in ret:
			self.div += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_divbits( int(r) )
		return p // c

	# count operations of modding number to polynomial
	def poly_mod(self, p, c):
		cpy = p.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = self.barrett_count(i,c)
			#cpy[ind] = self.naive_modulus_count(i,c)
		return cpy
		'''
		self.mod += len( p )
		return p % c
		'''

	# count operations of adding polynomials
	def poly_add_poly(self, p0, p1):
		# get the size of bigger polynomial
		# l = max( len(p0), len(p1) )
		# self.add += l
		ret = p0 + p1
		for r in ret:
			self.add += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_addbits( r )
		return p0 + p1

	# count operations of subtracting polynomials
	def poly_sub_poly(self, p0, p1):
		# get the size of bigger polynomial
		# l = max( len(p0), len(p1) )
		# self.add += l
		ret = p0 - p1
		for r in ret:
			self.add += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_addbits( r )
		return p0 - p1

	# count operations of multiplying polynomials
	def poly_mul_poly(self, p0, p1):
		# get the sizes of both polynomials
		sz0 = len( p0 )
		sz1 = len( p1 )

		# self.mul += ( sz0 * sz1 )
		# self.add += ( sz0 * sz1 )

		ret = p0 * p1
		for r in ret:
			self.add += ( (r).bit_length() // self.bitwidth ) + 1
			self.mul += ( (r).bit_length() // self.bitwidth ) + 1
			self.append_mulbits( int(r) )

		return p0 * p1

	# count operations of dividing polynomials
	def poly_div_poly(self, p0, p1):
		# get the sizes of both polynomials
		sz0 = len( p0 )
		sz1 = len( p1 )

		dif = sz0 - sz1 + 1

		self.div += dif
		# self.mul += ( dif * sz1 )
		# self.add += ( dif * sz1 )

		for i in range(dif):
			for num in p1:
				self.add += ( (num*p0[i]).bit_length() // self.bitwidth ) + 1
				self.mul += ( (num*p0[i]).bit_length() // self.bitwidth ) + 1

		return p0 // p1

	# implement a mod function with addition and comparison
	def naive_modulus_count(self, x, y):
		# the naive_modulus would loop for x//y times, which
		# will include one addition operation for each loop
		self.add += int(abs( x // y ))
		self.mod += 1
		# if there was a cmp counter, it would be 1 + x//y
		# self.comp += 1 + (x // y) 
		# return barrett(x,y) 
		return naive_modulus(x,y)

	def barrett_count(self, x, y):
		# this will count the number of operations for
		# barrett modulus reductions
		# self.mul += 2
		# self.add += 2
		self.mul += 2 * ( ( (x*y).bit_length() // self.bitwidth) + 1 )
		self.add += 2 * ( ( (x+y).bit_length() // self.bitwidth) + 1 )
		self.mod += 1
		return barrett(x,y) 

	def append_addbits(self, num):
		# this function will add the bitlength
		# of num to the addbits dict
		bit = num.bit_length()
		if bit not in self.addbits:
			self.addbits[bit] = 1
		else:
			self.addbits[bit] += 1
		return

	def append_mulbits(self, num):
		# this function will mul the bitlength
		# of num to the mulbits dict
		bit = num.bit_length()
		if bit not in self.mulbits:
			self.mulbits[bit] = 1
		else:
			self.mulbits[bit] += 1
		return

	def append_divbits(self, num):
		# this function will div the bitlength
		# of num to the divbits dict
		bit = num.bit_length()
		if bit not in self.divbits:
			self.divbits[bit] = 1
		else:
			self.divbits[bit] += 1
		return

	def append_modbits(self, num):
		# this function will mod the bitlength
		# of num to the modbits dict
		bit = num.bit_length()
		if bit not in self.modbits:
			self.modbits[bit] = 1
		else:
			self.modbits[bit] += 1
		return

	# generate string to print information about current count
	def __str__(self):
		addstr = f'Add: {self.add}'
		# substr = f'Sub: {self.sub}'
		mulstr = f'Mul: {self.mul}'
		divstr = f'Div: {self.div}'
		modstr = f'Mod: {self.mod}'

		addbits = f'Addbits: {self.addbits}'
		mulbits = f'Mulbits: {self.mulbits}'
		modbits = f'Modbits: {self.modbits}'

		output = f'{addstr}\n{mulstr}\n{modstr}'
		# output += f'\n{divstr}'
		# output += f'\n{addbits}\n{mulbits}\n{modbits}'
		return output

def main():
	a = Poly( [ 1, 2, 3] )
	b = Poly( [ 3, 2, 1] )

	oc = OperationsCounter()


	quo, rem = a / b
	oc.poly_div_poly(a,b)

	rem = rem % 3
	oc.poly_mod( rem ) 

	print( quo )
	print( rem )
	print( oc )

if __name__ == '__main__':
	main()

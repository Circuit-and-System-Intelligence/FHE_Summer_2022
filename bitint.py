#!/usr/bin/env python
# Date: 7/5/2022
# Create a class to hold both an integer and it's bit size
# 
# This class is to calculate the maximum bit arithmetic needed
# in the different encryption schemes

class Bitint():

	def __init__(self,value,size):
		self.value = value
		self.size = size

	def __add__(self,other):
		# add two integers together, only two bitints 
		# can be added together

		if (type(other) == int):
			value = self.value + other
			size = max( self.size, other.bit_length() )
			return Bitint( value, size )

		# prevents other types from being added with Bitint
		if (type(self) != type(other)):
			return NotImplemented

		return Bitint( self.value + other.value, max(self.size,other.size)+1 )

	def __radd__(self,other):
		if (type(other) != int):
			return NotImplemented

		value = self.value + other
		size = max( self.size, other.bit_length() )
		return Bitint( value, size )

	def __sub__(self,other):

		if (type(other) == int):
			value = self.value - other
			size = max( self.size, other.bit_length() )
			return Bitint( value, size )

		# prevents other types from being subtracted with Bitint
		if (type(self) != type(other)):
			return NotImplemented

		return Bitint( self.value - other.value, max(self.size,other.size)+1 )
	
	def __mul__(self,other):

		if (type(other) == int):
			value = self.value * other
			size = self.size*(other.bit_length())
			return Bitint( value, size )

		# prevents other types from being multiplied with Bitint
		if (type(self) != type(other)):
			return NotImplemented

		return Bitint( self.value * other.value, self.size+other.size )

	def __rmul__(self,other):
		if (type(other) != int):
			return NotImplemented

		value = self.value * other
		size = self.size*(other.bit_length())
		return Bitint( value, size )

	def __truediv__(self,other):

		# prevents other types from being divided with Bitint
		if (type(self) != type(other)):
			return NotImplemented

		return Bitint( self.value / other.value, max(self.size,other.size) )

	def __floordiv__(self,other):
		
		if (type(other) == int):
			value = self.value // other
			size = max(self.size,other.bit_length())
			return Bitint( value, size )

		# prevents other types from being divided with Bitint
		if (type(self) != type(other)):
			return NotImplemented

		return Bitint( self.value // other.value, max(self.size,other.size) )

	def __mod__(self,mod):
		return Bitint( self.value % mod.value, mod.size )

	def __len__(self):
		return self.size

	def __str__(self):
		return f'{self.value}'

	def __lt__(self, other):
		if (type(other) == int):
			return True if self.value < other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value < other.value else False

	def __le__(self, other):
		if (type(other) == int):
			return True if self.value <= other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value <= other.value else False

	def __gt__(self, other):
		if (type(other) == int):
			return True if self.value > other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value > other.value else False

	def __ge__(self, other):
		if (type(other) == int):
			return True if self.value >= other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value >= other.value else False

	def __eq__(self, other):
		if (type(other) == int):
			return True if self.value == other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value == other.value else False

	def __ne__(self, other):
		if (type(other) == int):
			return True if self.value != other else False

		if (type(self) != type(other)):
			return NotImplemented

		return True if self.value != other.value else False

	def __abs__(self):
		return Bitint( abs(self.value), self.size )

	def __int__(self):
		return int( self.value )

def min_bit_size(n: int):
	k = 0
	mask = 1
	while mask <= abs(n):
		mask = mask << 1
		k += 1
	return k
	

def main():

	a = Bitint(4, 5)
	b = Bitint(5, 5)

	c = a + b

	print(f'a: {a} {len(a)}')
	print(f'b: {b} {len(b)}')
	print(f'c: {c} {len(c)}')

	print(f'a < b: {a < b}')
	
	pass

if __name__ == '__main__':
	main()

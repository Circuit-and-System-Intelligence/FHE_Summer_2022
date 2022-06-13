# Date: 6/9/2022
# Create a class of vectors and matrices
# 
# This program will hold the contents of a vector and 
# matrix class

import numpy as np

class Vector():

	def __init__(self,terms:list):
		# initialize the vector with the terms given
		self.vec = terms.copy()

	def __getitem__(self, key):
		# returns the term at the indexed value
		return self.vec[key]

	def __setitem__(self, key, value):
		# update the value of the term
		self.vec[key] = value
		return

	def __iter__(self):
		# create an iter value set to 0
		self.n = 0
		return self

	def __next__(self):
		# loop through the vector's contents
		if self.n < len(self):
			res = self.vec[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration

	def __eq__(self,other):
		# check if two vectors are equal to each other

		if ( type(other) != type(self)):
			return False

		if ( len(other) != len(self)):
			return False

		scpy = self.copy()
		ocpy = other.copy()

		for i,j in zip(scpy,ocpy):
			if i != j:
				return False

		return True

	def __str__(self):
		# return a string interpretation of the print value
		return f'{self.vec}'
	
	def size(self):
		# this will provide the size of the vector
		return len(self.vec)

	def __len__(self):
		# return the length of the vector
		return len(self.vec)

	def __add__(self,other):
		# add two vectors together
		# or add constant to a vector
		if ( type(other) == int or type(other) == float or type(other) == complex ):
			cpy = self.copy()
			for ind, i in enumerate(cpy):
				cpy[ind] = i + other
			return cpy

		if ( type(other) != type(self) ):
			return NotImplemented

		if ( len(other) != len(self) ):
			raise ArithmeticError

		ret = []
		scpy = self.copy()
		ocpy = other.copy()
		for i,j in zip(scpy,ocpy):
			ret.append( i + j )

		return Vector(ret)

	def __radd__(self,other):
		# add a constant to a vector
		if ( type(other) != int and type(other) != float and type(other) != complex ):
			return NotImplemented

		cpy = self.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = i + other
		return cpy

	def __mul__(self,other):
		# multiply a constant to the vector
		if ( type(other) != int and type(other) != float and type(other) != complex ):
			return NotImplemented

		cpy = self.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = i * other
		return cpy

	def __rmul__(self,other):
		# multiply a constant to the vector
		if ( type(other) != int and type(other) != float and type(other) != complex ):
			return NotImplemented

		cpy = self.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = i * other
		return cpy

	def __sub__(self,other):
		# sub two vectors together
		# or sub a constant to a vector
		if ( type(other) == int or type(other) == float or type(other) == complex ):
			cpy = self.copy()
			for ind, i in enumerate(cpy):
				cpy[ind] = i - other
			return cpy

		if ( type(other) != type(self) ):
			return NotImplemented

		if ( len(other) != len(self) ):
			raise ArithmeticError

		ret = []
		scpy = self.copy()
		ocpy = other.copy()
		for i,j in zip(scpy,ocpy):
			ret.append( i - j )

		return Vector(ret)


	def __truediv__(self,other):
		if ( type(other) != int and type(other) != float and type(other) != complex ):
			return NotImplemented

		cpy = self.copy()
		for ind, i in enumerate(cpy):
			cpy[ind] = i / other

		return cpy



	def copy(self):
		# return a copy of the vector
		cpy = self.vec.copy()
		return Vector(cpy)


class Matrix():

	def __init__(self,mat):
		self.matrix = self.createMatrix(mat)

	def createMatrix(self,mat):
		arr = []
		for i in mat:
			arr.append( i.copy() )
		return arr

	def __getitem__(self, key):
		# returns the term at the indexed value
		return self.matrix[key]

	def __setitem__(self, key, value):
		# update the value of the term
		self.matrix[key] = value
		return

	def __iter__(self):
		# create an iter value set to 0
		self.n = 0
		return self

	def __next__(self):
		# loop through the matrix's contents
		if self.n < len(self):
			res = self.matrix[self.n]
			self.n += 1
			return res
		else:
			raise StopIteration
	
	def __str__(self):
		# return a string interpretation of the matrix
		return f'{self.matrix}'

	def __len__(self):
		# return the lenth of the matrix
		return len(self.matrix)

	def size(self):
		# return a tuple of the matrix size
		return ( len(self.matrix), len(self.matrix[0]) )

	def transpose(self):
		# transpose the matrix
		cpy = []
		sz = self.size()
		for i in range(sz[1]):
			cpy.append( [] )
			for j in range(sz[0]):
				cpy[i].append( self[j][i] )
		
		return Matrix(cpy)

	def copy(self):
		# return a copy of the matrix
		cpy = []
		for i in self.matrix:
			cpy.append( i.copy() )
		return Matrix( cpy )


# this function will calculate vector dot products
def vdot(a: Vector, b: Vector):
	ret = 0

	if ( len(a) != len(b) ):
		raise ArithmeticError

	a = a.copy()
	b = b.copy()

	try:
		for ind,i in enumerate(b):
			b[ind] = i.conjugate()
	except TypeError:
		pass

	for i,j in zip(a,b):
		ret += (i*j)

	return ret

# compute the matrix multiplication
def matmul(a: Matrix, b: Matrix):
	asz = a.size()
	bsz = b.size()

	if ( asz[1] != bsz[0] ):
		# if a col != b row then no matmul possibly
		raise ArithmeticError

	nsz = ( asz[0], bsz[1] )

	new = [ [0]*bsz[1] for i in range(asz[0]) ]

	for i in range(asz[0]):
		for j in range(bsz[1]):
			sm = 0
			for k in range(asz[1]):
				sm += (a[i][k] * b[k][j])
			new[i][j] += sm
	
	return Matrix(new)

# solve the system of linear equations
def linalg(a: Matrix, b: Vector):

	a = a.copy()
	b = b.copy()
	for ind, i in enumerate(a):
		a[ind] = Vector(i)
	
	(rows,cols) = a.size()
	
	# make the first triangle of 0's in the lower left
	for i in range(rows-1):
		r = a[i].copy()

		lead = r[i]

		# reduce the leading coefficient to 1
		r = r / lead
		b[i] = b[i] / lead

		a[i] = r

		for j in range(i+1,cols):
			dif = a[j][i] * -1

			a[j] = a[j] + (r * dif)
			b[j] = b[j] + (b[i] * dif)
	
	# create the second triangle of 0's in upper right
	for i in reversed(range(rows)):
		r = a[i].copy()

		lead = r[i]

		# reduce the leading coefficient to 1
		r = r / lead
		b[i] = b[i] / lead

		a[i] = r
		for j in reversed(range(i)):
			dif = a[j][i] * -1

			a[j] = a[j] + (r * dif)
			b[j] = b[j] + (b[i] * dif)

	return b

def main():

	#vec()
	#mat()
	test_linalg()

	return

def vec():

	a = Vector([1,2,3])
	b = Vector([1,2,3])
	c = Vector([1,2,3,4])

	print( vdot(a,b) )
	print( vdot(a,a) )

	print( vdot([1,1],[5,5]) )

	return

def mat():

	a = Matrix( [ [1, 2, 3, 4] ,
				[ 1, 2, 3, 4 ] ] )

	print(f'a: {a}')
	print(a.size())

	c = a.transpose()
	print(f'c: {c}')
	print(c.size())

	d = matmul(a,c)
	print(f'd: {d}')
	print(d.size())

	e = matmul(c,a)
	print(f'e: {e}')
	print(e.size())

	return

def test_linalg():

	a = [ ]
	b = [ ]

	a = np.random.randint( 0, 50, size=[5, 5] ).tolist()
	b = np.random.randint( 0, 50, size=[1,5] ).tolist()

	b = b[0]

	ma = Matrix( a )
	mb = Vector( b )

	mc = linalg(ma, mb)

	na = np.array( a )
	nb = np.array( b )

	nc = np.linalg.solve(na,nb)

	print(mc)
	print(nc)

	return

if __name__ == '__main__':
	main()
	pass

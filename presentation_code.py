from bfv import BFV

# create an Encryption Scheme object with custom parameters
es = BFV(q=2**22,n=2**5)

# define plaintext parameters
x, y = 1, 0

# encrypt x 
ctx = es.encrypt(x)
print(f'x: {x}')

# encrypt y
cty = es.encrypt(y)
print(f'y: {y}')

# multiply ciphertext
ctz = es.ctmult( ctx, cty )

# decrypt the ciphertext product
z = es.decrypt(ctz)																												[0]

print(f'z: {z}')
print(f'x*y==z: {z == (x*y)}')

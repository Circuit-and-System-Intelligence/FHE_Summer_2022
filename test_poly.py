
import numpy as np
from numpy.polynomial import polynomial as p
from poly import Poly

def main():
  test_div()
  return

def test_div():

  np_divee = np.random.randint(0,2 ** 10,size=[1,15])
  np_divor = np.random.randint(0,2 ** 15,size=[1,15])
  #np_divor = np.array( [ [1,0,0,0,0,0,0,0,1]  ] )

  np_quo, np_rem = np.polydiv( np_divee[0], np_divor[0] )

  a = np_divee.tolist()[0]
  a.reverse()
  b = np_divor.tolist()[0]
  b.reverse()
  poly_divee = Poly(a)
  poly_divor = Poly(b)

  poly_quo, poly_rem = poly_divee / poly_divor
  
  print(np_divee)
  print(np_divor)
  print(' ')
  poly_divee.polyprint()
  poly_divor.polyprint()

  print(' ')
  print(' ')

  print(np_quo)
  print(np_rem)
  print(' ')
  poly_quo.polyprint()
  poly_rem.polyprint()
  print(' ')

  # round the quotients and remainders
  np_quo = np.round(np_quo)
  np_rem = np.round(np_rem)
  poly_quo.round()
  poly_rem.round()

  quo = poly_quo.poly.copy()
  quo.reverse()
  rem = poly_rem.poly.copy()
  rem.reverse()

  print( np.array_equal( np_quo, quo ) )
  print( np.array_equal( np_rem, rem ) )

  return

if __name__ == '__main__':
  main()
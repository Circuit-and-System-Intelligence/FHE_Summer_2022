#!/usr/bin/env python
# Date: 6/8/2022
# Create a Class to emulate ckks encryption scheme 
# 
# This program will create a class for the ckks 
# fully homomorphic encryption scheme

from poly import Poly
import numpy as np

class ckks():

  def __init__(self,M=8,delta=64):
    self.M = M
    self.delta = delta
    return


def main():

  es = ckks()

  return

if __name__ == '__main__':
  main()
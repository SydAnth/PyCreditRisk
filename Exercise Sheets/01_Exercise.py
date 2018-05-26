# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:03:08 2018
SPL Exercise sheet 1
"""


import numpy as np
import pandas as pd

###############################################################################
# 1. Create two strings with words ``slumdog'' and ``millionaire''. 
#    Print them in separate lines and in one line (using "paste").
##############################################################################

var1 = 'slumdog'
var2 = 'millionaire'
print(var1)
print(var2) #two-lines

print(var1,var2,sep=' ') # one-line

print('%s %s was not a good movie' %(var1,var2))
#insert string into pre-defined text

###############################################################################
# 2. Create a row vector "a" with the elements 1,3,5,7,11,13,17 and 19.
###############################################################################

a = [1] #Create a Python list
print(a)
lower = 0
upper = 20
print("Prime numbers between %s and %s are:" %(lower,upper))
for num in range(lower,upper + 1):
   # prime numbers are greater than 1
   if num > 1:
       for i in range(2,num):
           if (num % i) == 0:
               break
           elif (num == 2):
               break
       else:
            print(num)
            a.append(num) #Append vector by each prime number
            
a = np.array(a) #Turns it into a NumPy Array 

#How to differentiate between Row and Column Vectors in Python ?
a = np.array([1,3,5,7,11,13,17,19]).transpose()

###############################################################################
# 3. Generate a column vector "c" with 1,4,9,16,..,64 = 1^2, 2^2, 3^2,.., 8^2.
###############################################################################
c= np.arange(1,9,1) * np.arange(1,9,1)
c = np.power(np.arange(1,9,1),2) 

#np.arange create a vector same as seq in R
#However the upper bound is excluding


###############################################################################
# 4. Generate a column vector "b" with 2,..., 2^8.
###############################################################################
b = np.array(np.power(np.full((8,),2),np.arange(1,9,1)))
#np.full creates a row vecter of 2s

###############################################################################
# 5. Find the positions where elements of "b" and b"c" coincide (use "which")
###############################################################################
np.where(b==c)
###############################################################################
# 6. Create a matrix "M.c" with the first column vector "b" and the second 
#    column vector "c". Print the dimension. Print the seventh row of "M.c".
###############################################################################
Mc =np.column_stack((b,c))
#np.where is the Python equivalent to R which function
#np.where(Mc[:,0]==Mc[:,1])
#Returns Values 
Mc[np.where(b==c)]
Mc.shape
Mc[7,:]


###############################################################################
# 7. Create a matrix "M.r" with the first row vector "a" and the second row 
#    vector "b". Rename the rows of "M.r" to "a" and b" and the columns to 
#     S, T, ... , Y, Z.
###############################################################################
# Original Hint.: These are the letters 19 to 26 of the alphabet. 
#        The alphabet is provided by constants "letters" and "LETTERS".

# The ascii_uppercase library operates similarily
# to the LETTERS in R but requires a list wrapper 
from string import ascii_uppercase
letters = list(ascii_uppercase[16:24])
Mr = pd.DataFrame(np.row_stack((a,b)),columns=letters)




###############################################################################
# 8. Print the matrix "M.r" without the column W.
###############################################################################
select = [x for x in Mr.columns if x != "W"]
Mr[select]

###############################################################################
# 9. Print elements of "M.r" larger than 12.
###############################################################################

#  Boolean Subselction creats NaN the dropna statement
#  cleans this up
Mr[(Mr>=12)].dropna(axis=1, how='any')

#2 Methods of subselecting Dataframes using booleans 
Mrpd=pd.DataFrame(np.row_stack((a,b)),index=['a','b'],columns=letters)
Mrpd.where(Mrpd>=12) #or 
Mrpd[Mrpd>=12]

###############################################################################
# 10. Compute the values of the function $y=e^{-x}$ for an equidistant grid 
#     for x from -3 to 3 with an stepwidth of 0.5.
###############################################################################
np.exp(-np.arange(-3,3.5,.5))

###############################################################################
# 11. Create a vector "d" which contains the numbers from 1 to 100 
#     and another vector "e" which contains 100 elements equal to 7.
###############################################################################
d=np.arange(1,101,1)
e=np.full((100,),7)


###############################################################################
# 12. Create a 10x10 matrix "D" which contains the numbers 1 to 100 (filled
#     by columns) and another 10x10 matrix "E" which contains the numbers 
#     1, 1/2, 1/3..., 1/100. (filled by rows).
#############################################################################
#Important Note np.arange and reshape creates
#matrices by row use order ='F' to fill by column
D = np.arange(1,101).reshape(10,10,order='F')
E = np.divide(np.full((100,),1),np.arange(1,101)).reshape((10,10))
###############################################################################
# 13. Calculate the the sum D + E, the difference D-E, the (matrix!) product
#     D and E.  And the product of elements (d_{i,j} . e_{i,j})_{i,j=1...10}.
###############################################################################
multi=  D*E #elementwise multiplication
matmulti = np.matmul(D,E) #inner product
D+E
D-E
###############################################################################
# 14. Print the diagonal elements of the matrix $P = D \cdot E$.
###############################################################################
np.diag(D*E)


###############################################################################
# 15. Compute the difference of the functions y_1 = x^5+x^4+x^3+x^2+x+1
#     and y_2=1+x(1+x\(1+x(1+x(1+x)))) for some x.
#     What is the difference between both methods?
###############################################################################
x=5000000
y1 = x**5+x**4+x**3+x**2+x+1
y2 = 1+x*(1+x*(1+x*(1+x*(1+x))))
y1,y2

abs(y2 - y1)

# The issue doesn"t exist in Python


###############################################################################
# 16. Assuming $\{2,3,5,3,2,5,7,4,2,5\}$ are prices, 
#     calculate log returns (by two methods, one using "diff").
###############################################################################
prices =[2,3,5,3,2,5,7,4,2,5]

ret = np.diff(np.log(prices))


###############################################################################
# 17. From vector 
#    {1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6\}
#   find the points where values change.
###############################################################################
v = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 5, 5]
#np.roll lets you shift through a vector
np.where(np.roll(v,1)!=v)

###############################################################################
# 18. Define matrix A = 
#           -1.00 , 3.71 , 2.80 , 0.01 , 1.19
#            0.40 ,-1.81 ,-1.96 , 1.84 , 1.74
#           -4.30 , 1.71 , 0.68 , 0.11 , 3.44
#            0.03 , 3.90 , 0.41 , 0.02 , 1.05
#            0.24 ,-0.01 , 2.10 , 2.87 ,-3.57
###############################################################################



A=       np.array([[-1.00,3.71,2.80,0.01,1.19],
                   [0.40,-1.81,-1.96,1.84,1.74],
                   [-4.30,1.71,0.68,0.11,3.44],
                   [0.03,3.90,0.41,0.02,1.05],
                   [0.24,-0.01,2.10,2.87,-3.57]])
    



###############################################################################
# 19. Find the determinant.
##############################################################################
    
np.linalg.det(A)    

###############################################################################
# 20. Find inverse an multiply by the original (test whether we get the Identity matrix).
###############################################################################
A_inv = np.linalg.inv(A)
np.round(np.matmul(A_inv,A),5)
#Notice that if you don't round you will not
#get zeros at the End !


###############################################################################
# 21. Switch the upper triangles in the following matrices
#      B =  1  &  1  &  1\\
#           1  &  2  &  3\\
#           1  &  3  &  6
#
#      C = 1  &  4  &  7\\
#          2  &  5  &  8\\
#          3  &  6  &  9
###############################################################################
B = np.array([[1,1,1],[1,2,3],[1,3,6]])
C = np.arange(1,10).reshape(3,3,order='F')


np.triu(B,1)+np.tril(C)
np.triu(C,1)+np.tril(B)
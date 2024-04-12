import numpy, random

lr = 1 # leaning rate
bias = 1 # value of bias
# wights generated in a list (3 in total, 2 neurons and 1 for bias)
weights = [random.random(), random.random(), random.random()]

print(f"weights[0]= {weights[0]}")
print(f"weights[1]= {weights[1]}")
print(f"weights[2]= {weights[2]}")

sigmoid = lambda x: 1/(1+numpy.exp(-x))  # Activation function (here Heaviside)

# classification between 2 groups
def perceptron(input1, input2, outputExpected):
  print(f"input1= {input1}, input2= {input2}, outputExpected= {outputExpected}")

  outputP = input1*weights[0] + input2*weights[1] + bias*weights[2]
  print(f"outputP= {outputP}, sigmoid= {sigmoid(outputP)}")
  outputP = sigmoid(outputP)

  # calculate the error
  error = outputExpected - outputP
  print(f"error= {error}")

  weights[0] += error*input1*lr
  weights[1] += error*input2*lr
  weights[2] += error*bias*lr

  print(f"weights[0]= {weights[0]}")
  print(f"weights[1]= {weights[1]}")
  print(f"weights[2]= {weights[2]}")
  print("----- END FUNCTION ----")

# training phase
for i in range(4) : # <input1> , <input2> => outputExpected
  perceptron(1,1,1)  # True    or  True    => True
  perceptron(1,0,1)  # True    or  False   => True
  perceptron(0,1,1)  # False   or  True    => True
  perceptron(0,0,0)  # False   or  False   => False
  print(f"----- END LOOPING {i} ---- \n")

# running perceptior
  
x = int(input())
y = int(input())
outputP = x*weights[0] + y*weights[1] + bias*weights[2]
if outputP > 0 : #activation function
   outputP = 1
else :
   outputP = 0
print(x, "or", y, "is : ", outputP)

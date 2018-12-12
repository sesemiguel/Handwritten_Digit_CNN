# Implementation of a Convolutional Neural Network for Handwritten Digit Recognition

### Overview

This software implementation was created for the purpose of simulation. The simulation was intended for a hardware model of the said CNN with the same application, which is discussed in another study. 

This convolutional neural network (CNN) has 6 main portions:

1. Initialization
2. Input Layer
3. Convolutional Layer
4. Output Layer
5. Classification
6. Training

The data set used was the MNIST database of handwritten digits. It includes 60,000 training digits and 10,000 test digits.

## Initialization
The initialization process was intended for scaling purposes of the inputs and dimensions of the layers in the CNN. This enables the network to:

* Accept different sizes of images
* Accept different sizes and numbers of filters
* Adjust the layer dimensions from the said input images and filters.

The filters and weights are all initialized as random values.

```python
# Random values of filters
input_filter = np.random.rand(depth, f_rows, f_cols)

# Random values of weights
convolved_nodes_to_output_nodes = np.random.rand(total_weights, outputs)
```

The convolution output dimensions are equated through:

`conv_dim = (input_dim - filter_dim) + 1`

While the total number of fully connected (FC) weights are equated through:

`total_weights = depth * conv_rows * conv_cols`

The dimension of the convolutional layer is initialized through:

`conv_layer = ([depth, conv_rows, conv_cols])`

## Input Layer
The input layer accepts the digits from the MNIST database through CSV files. These CSV files are accepted through the function *genfromtxt* with comma as the delimiter. A function was created as shown below.

```python
def pd_csv_to_2darray(input_filename):
	try:
		return np.genfromtxt(input_filename, delimiter=',')
	except IOError:
		print("File not available!")`
```

## Convolutional Layer
The convolutional layer uses the function *signal.convolve* from SciPy. It has a mode of *valid* which means that the convolution has a stride of 1 with no padding. Convolution happens one depth at a time

```python
# Convolution
	for i in range(0, input_filter.shape[0]):
		convolved_nodes[i] = signal.convolve(input_img, input_filter[i], mode="valid")
```

This layer also includes the non-linear activation function of the convolved entities. The Rectified Linear Unit (ReLU) was used on this implementation. It accepts the positive values and turns all negative values into 0.

```python
# ReLU activation implementation
def relu_activation(data_array):
    return data_array * (data_array>0)
```

It will then be flattened for the next layer

```python
# Flattening of sigmoid activated convolution layer
	convolved_nodes_sigmoid_flat = convolved_nodes_sigmoid.reshape(1,total_weights)
```

## Output Layer
The output layer contains the connection of the ReLU activated convolved nodes and the output layer nodes. These nodes have weights which were randomized. The NumPy function *matmul* was used. This function simply gets the dot product of the two matrices.

```python
# Fully connected layer
	output_nodes_flat = np.matmul(convolved_nodes_sigmoid_flat, convolved_nodes_to_output_nodes)
```

## Classification
To classify the results, the softmax activation function is used. It is similar to a sigmoid function but it is used for classification the prediction of the network.

```python
def softmax(x):
    A = x-np.max(x)
    e_x = np.exp(A)
    return e_x / e_x.sum(axis=0), A
```

## Training
Backpropagation techniques, such as error calculation of errors and stochastic gradient descent (SGD), were used. Series of error calculations and weight/filter adjustments were monitored per layer. Below is the implementation of the error calculation based on the target outputs

```python
# Error calculation on Output Layer
	softmax_output_row = np.transpose(softmax_output)
	error_array = error_calculation(set_target(target_value), softmax_output_row)
```

Below is the implementation of the error calculation on the output layer

```
# Error * Sigmoid backpropagation formula
	A = error_array * relu_deriv(softmax_output_row)
	
	# Transpose result A
	A = np.transpose(A)

	# SOP of A and Nodes in flat form
	B = np.matmul(A,convolved_nodes_sigmoid_flat)

	# Transpose result B
	B = np.transpose(B)
```

To update the weights, SGD is applied.

```python
	# Updating of FC weights --- Cost Function
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_to_output_nodes[i][j] = convolved_nodes_to_output_nodes[i][j] - L_rate * B[i][j]
```

The values of the convolutional layer nodes are adjusted to prepare for the updating of the filters.

```python
# Updating of Convolution layer nodes
	for i in range(0,total_weights):
		for j in range(0,outputs):
			convolved_nodes_sigmoid_flat[0][i] = (softmax_output_row[0][j] * convolved_nodes_to_output_nodes[i][j]) + temp 
			temp = convolved_nodes_sigmoid_flat[0][i]
		temp = 0
```

The updated convolutional layer nodes are convolved with the input image. The result is subjected to SGD for the calibration of the filters.

```python
# Updating the filters
	for x in range(0, input_filter.shape[0]):
		input_filter[x] = input_filter[x] + (signal.convolve(input_img, convolved_nodes[x], mode="valid")) * L_rate
```

## Contributors
1. CHAN, Zion Eric O.
2. FALLAR, Mac Excel S.
3. RAMOS, Patrick Julian M.
4. SESE, Miguel Karlo D.

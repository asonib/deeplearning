# LeNet Architecture
<pre>
  Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 28, 28, 32)        832       
_________________________________________________________________
activation_4 (Activation)    (None, 28, 28, 32)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 64)        51264     
_________________________________________________________________
activation_5 (Activation)    (None, 14, 14, 64)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 250)               784250    
_________________________________________________________________
dense_4 (Dense)              (None, 10)                2510      
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0         
=================================================================
Total params: 838,856
Trainable params: 838,856
Non-trainable params: 0
_________________________________________________________________
</pre>

<h4> Loss/Validation Loss</h4>

![Image description](https://raw.githubusercontent.com/asonib/deeplearning/master/ConvoNet/LeNet/graphs/loss.png)

<h4> Accuracy/Validation Accuracy</h4>

![Image description](https://raw.githubusercontent.com/asonib/deeplearning/master/ConvoNet/LeNet/graphs/accuracy.png)

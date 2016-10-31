# Learn To Tensorflow

This repository contains mini programming challenges
aimed to gradually stretch your working skills and 
understanding of tensorflow basics.

Each challenge has a corresponding solution in the solutions folder.
If you need a hint or some starter code to give you structure there are templates in the template folder which have only the outline of the solution files.


## How to use this resource
These are skills building exercises.  It is suggested that you go through the challenges in Levels where each Level becomes increasingly more difficult.  If a Level takes you more than three hours to complete then consider repeating the Level before going onto the next Level.

### Level 1 - Beginner
Try to do the challenges using as much reference as you need such as the templates, searching for information on the web, and referencing the solution as needed.

### Level 2 - Template user
Same as Level 1 but do not reference the solution folder.

### Level 3 - Web researcher
Same as Level 2 but also do not refer to the template folder.

### Level 4 - Solo
Do all the challenges without any reference including searching the web.

### Beyond Level 4 - The Graduate
If you've completed Level 4 within three hours then consider yourself graduated.  It's time to move onto more complex real life applications.

Consider visiting https://www.tensorflow.org and going through each tutorial in a similar manner, graduating yourself from Level 1 up to Level 5.  There is a unix bash script in the solutions folder **solutions/convert-solution-to-template.sh** that you could use on any code to convert it to a template, like so:
```
./solutions/convert-solution-to-template.sh < solution-file-name.py > template-file-name.py
```



## The challenges

### 00-count.py
Create a very basic tensorflow program that adds one to a variable and uses a loop to run the addition ten times.


### 01-multiply.py
Multiply two matricies together.


### 02-xor-1d.py
Solve the xor problem using
```
input  = [[0,0],[1,1],[1,0],[0,1]]
output = [[0],  [0],  [1],  [1]]
```
And break the problem into the following layers
- the 1x2 input layer
- 2x3 + bias hidden sigmoid layer
- 3x1 + bias sigmoid output layer
- calculate loss as the sum of the squares of y - y_
- use gradient decent (set to 1.0) to minimize loss
Run iteratively 500 times and print all the variable data.


### 03-regularization.py
Add regularization to 02-xor-1d.py

One type of regularization is to minimize the values of the transformation matricies, such as the as the average of the square of m1 and m2.


### 04-xor-2d.py
Solve the xor problem using
```
input  = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
output = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]
```


### 05-feed.py
Replicate 04-xor-2d, but instead of using constants for input and output, use feeds.


### 06-save.py
Improve 05-feed.py to save the session information at the end of training and to use the saved session information if it exists instead of training.


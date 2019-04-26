# Getting Started Lessons

The objective of these lessons are to gently introduce some core concepts and basic usage of Tensorflow.  There are several common misunderstandings that hold independent learners back, such as the difference between the Tensorflow code and the Tensorflow graph, which can take weeks or months to figure out and can be very fustrating to learn on ones own.

The following concepts will be explored:

- executing the *Code* vs the *Graph*
- reversing the *Graph*
- a definition of *Model*, *Learning Rate*, and *Loss*
- a better way to structure code
- how tensorflow can be used to *Learn* variables in a *Graph*

For each lesson open the corresponding file, read the code and the commends, then run and modify the code to get a feel for it.

## Lesson 0 - A review

- Use python to calculate a x b = c

## Lesson 1 - Just make something happen

- Make something super simple
- Address #1 mental hurdle right away
- 'a' isn't equal to 5, 'b' isn't equal to 7 !
- Code the *graph* then run the *graph*

## Lesson 2 - Same thing but a little better

- placeholder instead of constants
- reuse the same *graph*

## Lesson 3 - Same thing but a little better

- Variable vs Constant vs Placeholder
- Change variable

## Lesson 4 - In Reverse

- Tensorflow is used to *learn*
- for *ab = c*, what if we know *b* and *c*?
- use Tensorflow to estimate *a*
- what happens if the learning rate is too high or too low?

## Lesson 5 - In Reverse with Gradient

- Same as before but let's look at the gradient

## Lesson 6 - Estimation

- Estimate an imperfect line *y = ax + b* given *y* and *x*.
- *y = ax + b* is called our *model*

## Lesson 7 - Same thing but a lot better

- Use the matrix

## Lesson 8 - Same thing with a curve

- *y = ax^2 + bx + c*

## Lesson 9 - Be a good programmer

- move model to its own file
- move data to its own file
- for bonus points we graph

Try running lesson 9 using the following options
```
python lesson_09_tidy.py --model poly_model --plot
python lesson_09_tidy.py --model linear_model --plot
python lesson_09_tidy.py --model abs_model --plot
```

The default data set is *poly_data*.  Try using the *linear_data* for this example like so
```
python lesson_09_tidy.py --data linear_data --model poly_model --plot
python lesson_09_tidy.py --data linear_data --model linear_model --plot
python lesson_09_tidy.py --data linear_data --model abs_model --plot
```


## Lesson 10 - Beyond Linear

In this code we will use matrix multiplication.  There are lots of good resources online to explain matrix math if you are new to matrix operations or need a refresher.

- A lot of problems are not linear
- First multi-layered neural net

First try running this lesson using a linear model.
```python lesson_10_non_linear.py --model linear_model```

You will notice that **fx** doesn't predict **y** well.  That is because this is the xor problem which is not solvable with a linear equation.  No problem, if we stack layers in Tensorflow we can solve such problems.

Next, try running this lesson using a multi layered (stacked) model
```python lesson_10_non_linear.py --model stacked_model```

You will notice that **fx** predicts **y** perfectly.  Yeah.


## Challenge

- add data and model files to lesson 9 and 10


# Review

We explored the following concepts:

- executing the *Code* vs the *Graph*
- reversing the *Graph*
- a definition of *Model*, *Learning Rate*, and *Loss*
- a better way to structure code
- how tensorflow can be used to *Learn* variables in a *Graph*

We have scratched the surface of the capabilities of Tensorflow and you
have gain valuable insight in the basics of the *code*, *graph*, and *model*
that trip up most new students.

As a bonus we even saw an example of how we can solve non-linear problems.

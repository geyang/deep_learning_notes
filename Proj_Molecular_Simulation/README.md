# Tensorflow Implementation of Electrons on Helium Simulation

This project is a Electrons on Helium 2D simulation implemented with Tensorflow. 
As of now, it correctly finds the 2D electron configuration corresponding to the
lowest energy. However some more work needs to be done to tweak the optimizers,
so that the final configuration can be annealed even better.

## Motivations

When writing fast, GPU accelerated scientific code, lots of time are spent on:

1. writing CUDA code in c++ that are mission-specific.  

   This means that if I write code for a GPU then I wouldn't be able to run
   the same code on my working laptop, and vise versa. The code is simply not
   transportable.

2. For an optimizer, writing the Jacobian of the system of equations(SOE) as well
as the SOE itself.  

   This, again, is very model and mission specific, resulting in lots of 
   hard-to-debug, non-reusable code.

Tensorflow solves both of these two problems. Tensorflow provides a useful 
set of operators mostly modeled after the proven API of `numpy`. Each of 
 these operators comes with both CPU and GPU run time, and can be switch 
 by simply passing in a scope parameter, making your code a lot more 
 transportable.
 
 In addition, these operators provides a basis for a good level of abstraction, 
 and they each come with a gradient definition. The tensorflow optimizers
 are able to take advantage of these gradients and the tensor flow, which 
 eliminates the need for you to manually write and validate the Jacobian
 of you SOE. 

**Transparent Math**: One of the goals of this project is to expose the parts
that you need to write a simple particle simulation with tensorflow. So the 
code is done in such way that the logic between the moving parts is relatively 
simple. You can take the code to run as-it-is, or you can look through the 
small code base and adapt it to your own.

## Tricky things

### Inverse Pairwise Distance Matrix has a Faulty Gradient

When we calculate the pairwise interaction between the charges, the following
matrix pops up:

```python
            [ 
                [ 0,           1/ r_{0, 0}, 1/ r_{0, 1}, ... ],
                [ 1/ r_{1, 0}, 0,           1/ r_{1, 2}, ... ],
                [ 1/ r_{3, 0}, 1/ r_{3, 2}, 0,           ... ],
                [ 1/ r_{4, 0}, 1/ r_{4, 2}, 1/ r_{4, 3}, ... ],
                ... 
            ]
```

A naive implementation of the inverse pairwise distance matrix has a automatically
generated gradient that goes to `nan`. This is a result of finite precision 
artifacts of the quantity inside the square root operator, the Distance^2 matrix. 
So in this working version, we postpone taking the square root till after 
adding the interactive term and the static term. 

This is a nice and simple work around that allows us to avoid writing our
own tensor operator in c++ with definition of the gradient. Doing so would've 
defeated the purpose of this exercise, which is to use tensorflow's automatic 
gradient calculation as stated in the [motiations](#Motivations) session.

```python
# Here we define the trap potential function for each xy pair (is a tensor).
def static2(xy):
    return trap_constant * tf.reduce_sum(
        tf.square(xy),
        reduction_index=[0]
    )
    
xys = tf.Variable(some_init, name="electron_locations")
total_energy = energies.total(xys, static2)
```

Now this total energy tensor has a automatically defined gradient that 
the tensorflow optimizers can use. To find the lowest energy, you 
can just simply gradient descent.

## TODOs
- [ ] **Sample and Figures** add more figures showing the annealed results.

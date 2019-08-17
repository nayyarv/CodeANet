# CodeANet

TL;DR: If Ray Hettinger wrote a Neural Net, what would it look like?

Neural Nets are one of the simplest machine learning algorithms out there.
At it's core, it's just a series of matrix multiplcations with non linearities
on the forward pass, while the backward passes are just chain rules and more matrix multiplications.

In this talk, we build up a simple neural net codebase from scratch, adding features 
and changing the design as necessary as requirements change. 

Through this exercise, we motivate and implement

- Forward Propagation of a few simple layers
- Backward Propagation for those layers
- Custom Optimisers

Each step of which upends previous assumptions and requires code changes. 
We finish with a discussion on the existing design as compared to 
torch and the importance of testing in ML codebases.
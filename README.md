# Feature-wise transformations: FiLM for visual reasoning

## References :

- https://distill.pub/2018/feature-wise-transformations/
- https://arxiv.org/pdf/1709.07871.pdf
- https://github.com/ethanjperez/film

## Project: 

This is a reproduction of the FiLM architecture for conditional neural networks described in this article: https://arxiv.org/pdf/1709.07871.pdf.
The aim is to build an architecture capable of making predictions $Y$ conditional on a main input $X$ and a given context on the data $Z$ by
adding an affine transformation through a main chain of resNet blocks ($FiLM(X) = a(Z).X + b(Z)$ where $a(Z)$ and $b(Z)$ are outputs of a contextual neural network processor). 
Basically, the architecture allows different data structures to be taken as input $X$ and context $Z$.

We train the model on a visual reasoning response classification task (with a finite number of possible responses): we pass images as input and questions about the images in the context of a question.
The data is generated artificially by a deterministic algorithm (DataGen class) to experiment with the architecture: 
- The images are a set of 2D shapes in a 3*3 grid.
- The questions are in French. We ask questions about position/color/presence.
- The answers are a list of possible answers depending on the question: yes, no, red, ...



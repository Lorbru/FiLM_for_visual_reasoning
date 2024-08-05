# Feature wise transformations : FiLM for visual reasoning

## Link :
- https://distill.pub/2018/feature-wise-transformations/
- https://arxiv.org/pdf/1709.07871.pdf
- https://github.com/ethanjperez/film

## Project : 

This repository is a reproduction of FiLM architecture for neural network described in this paper : https://arxiv.org/pdf/1709.07871.pdf

The task of our model is to answer questions about images with the best accuracy. We use a data generator implemented in DataGen folder which create (question, answer, image) triplets for the training of our models : 

- Images are a set of 2D shapes in a 3*3 grid
- Questions are in french language speaking. We ask questions about position/color/presence.
- Answers are a list of possible answers depending on the question : yes, no, red, ...

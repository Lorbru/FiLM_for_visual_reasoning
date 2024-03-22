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

The model training and architecture is implemented in Model folder.
Running main will train a model on a little set of data. You can change parameters in config.txt.
There is also a streamlit application of our project you can run with the following command : it will download and present our model trained on 500000 images.

```
python -m streamlit run src/streamlit_demo.py
```

Check also you have all the required libraries : 

```
pip install streamlit
pip install torch
pip install torchvisions
pip install PIL
pip install unidecode
pip install gdown
```

![image](https://github.com/Lorbru/FiLM_for_visual_reasoning/assets/135026945/ef214441-fa4b-477b-a610-973038bb9bc3)

![image](https://github.com/Lorbru/FiLM_for_visual_reasoning/assets/135026945/83c96f40-901d-4fbb-bad0-3991b9aea8e0)


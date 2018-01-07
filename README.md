# Cosine-Similarity-embedded-with-Contrastive-loss function


The two feature vectors x1 and x2 are learned from siamese architecture of convolutonal neural networks, 
c is label of training data which is 1 if two images in a pair belong to the same class and -1 vice versa.

The formula for the cosine similarity between two vectors x1 and x2 is given as:

$$cosSim(x_1,x_2)=\frac{x_1'x_2}{x_1'x_1+x_2'x_2}$$

if c is equal to 1 (i.e., x1 and x2 belongs to a same category)

$$\frac{\partial f} {\partial x_{1i}}= \frac{x_{2i}}{x_1'x_1+x_2'x_2} - \frac{x_{1i}}{x_1'x_1}*cosSim(x_1,x_2) $$

if c is equal to -1 (i.e., x1 and x2 belongs to a different category)

$$\frac{\partial f} {\partial x_{1i}}= - \frac{x_{2i}}{x_1'x_1+x_2'x_2}  + \frac{x_{1i}}{x_1'x_1}*cosSim(x_1,x_2) $$

Similarly the derivative of loss function with respect to x2 is calculated as:

if c is equal to 1 (i.e., x1 and x2 belongs to a same category)

$$\frac{\partial f} {\partial x_{2i}}= \frac{x_{1i}}{x_1'x_1+x_2'x_2} - \frac{x_{2i}}{x_2'x_2}*cosSim(x_1,x_2) $$

if c is equal to -1 (i.e., x1 and x2 belongs to a different category)

$$\frac{\partial f} {\partial x_{2i}}= - \frac{x_{2i}}{x_1'x_1+x_2'x_2}  + \frac{x_{2i}}{x_2'x_2}*cosSim(x_1,x_2) $$

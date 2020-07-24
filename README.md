# gene-prediction

Written by: Vignesh Ravindranath, Shawn Huang

This project was based on the 2008 paper 'Gene prediction in metagenomic fragments: A large scale machine learning approach' by Hoff et at. The purpose of the project is to create a machine learning algorithm to predict potentially novel and known genes in microbial genomes. The independent project also served to gain a deeper understanding of machine learning approaches and bioinformatics concepts.

The project consists of three main steps:
1) Feature engineering/extraction
2) Linear Discriminant training on high-dimentional features
3) Binary classification neural network for coding/noncoding gene prediction

In the first step, features such as monocodon (tricodon) usage, dicodon (hexcodon) usage, translation initiation sites (TIS), and GC content must be extracted from prokaryotic genomes. Monocodon and dicodon usage refer to the frequence of codons and dicodons (6 base pairs) in coding and noncoding regions. Coding proteins often have a TIS upstream of the start codon. These TIS patterns are extracted by comparing up and downstream regions for positive TIS candidates (true start codons in coding sequences) to negative TIS candidates (in-frame start codons within coding sequences). Lastly, it is well known that the GC content between coding and noncoding regions vary. 
In the second step, linear discriminants are derived to reduce the dimensionality of the extracted features. The individual features (excluding GC content) are taken as multivariate linear regression problems and the Normal Equation is utilized to compute the weights (coefficient) matrix for each feature. 
In the last step, a neural network is trained on fragmented data.


Summary of features:
- x1 - tricodon       - (n,64)   --reduced to a weights matrix of (64,1)
- x2 - hexcodon       - (n,4096) --reduced to a weights matrix of (4096,1)
- x3 - positive TIS   - (n,58,64 == n,3712) --reduced to a weights matrix of (3712,1)
- x4 - negative TIS   - (m,58,64 == m,3712) --reduced to a weights matrix of (3712,1)
- x5 - complete seq   - 1 if fragment contains a complete gene, else 0
- x6 - incomplete seq - 0 if fragment contains a complete gene, else 1
- x7 - GC content     - (n,1)

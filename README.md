# Document-Classification

The focus was on feature engineering, that is representing the text as a feature vector of numeric descriptions of the input.
Stanford coreNLP was used to pre-process the text in order to get syntactic features.
Pre-computed word2vec word representations were used to expand the standard lexical representations.
The features used for training the model were Lexical features, Lexical features with Expansion, Dependency Features and
Syntactic Productions.
I trained classiﬁers using these features and then used them to perform document classiﬁcation task.

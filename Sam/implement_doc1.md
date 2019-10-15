To summarize text you have 2 main approaches:

Extractive method: which is choosing specific main words from the input to generate the output , this model tends to work , but won’t output a correctly structured sentences , as it just selects words from input and copy them to the output , without actually understanding the sentences , think of it as a highlighter .

2. Abstractive method: which is building a neural network to truly workout the relation between the input and the output , not merely copying words , this series would go though this method , think of it like a pen.


building a deep network that is capable of:

*analyzing sequences of input
*understanding text
*outputting sequences of output in form of summarizes
*hence the name of seq2seq , sequence of inputs to sequence of outputs , which is the main algorithm that is used here .

<b>(*)Implementations using a seq2seq encoder(bi directional lstm ) decoder (with attention)</b>

![Encoder decoder](https://hackernoon.com/hn-images/1*1BwMlWYa5ewAt96Z-gJ8Yg.png)

<b> (*)Other implementation that i have found truly interesting is a combination of creating new sentences for summarization , with copying from source input , this method is called pointer generator.</b>
<br><br><b> (*)Other implementations that i am currently still researching , is the usage of reinforcement learning with deep learning.</b>

since our task is a nlp task we would need a way to represent words ,this have 2 main approaches that we would discuses ,

either providing the network with a representation for each word , this is called word embedding , which is simply representing a certain word by a an array of numbers , There are multiple already trained word embedding available online , one of them is Glove vectors
or letting the network understand the representations by itslef

So.... now we have to use either process or both....

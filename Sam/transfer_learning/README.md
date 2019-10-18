<h3>Abstractive summarization using BERT(Bidirectional Encoder Representations from Transformers) as encoder and transformer decoder </h3>

After searching for a long f**king time I decided to use a text generation library called "Texar" , Its a beautiful library with a lot of abstractions, it is similar to 
scikit learn, but for text generation problems.

The main idea behind this architecture is to use the transfer learning from pretrained BERT a masked language model ,
I have replaced the Encoder part with BERT Encoder and the deocder is trained from the scratch.

One of the advantages of using Transfomer Networks is training is much faster than LSTM based models as we elimanate sequential behaviour in Transformer models.

Transformer based models generate more gramatically correct  and coherent sentences.<br>
Sayan da please check this out!!!! 


<h3>To run the model</h3>
<pre>
Follow the steps

decide the dataset and all other shit


Step1:
Run Preprocessing
<b>python preprocess.py</b>

This creates two tfrecord files under the data folder.

Step 2:
<b>python main.py</b>

Configurations for the model can be changes from config.py file

Step 3:
Inference 
Run the command <b>python inference.py</b>
This code runs a flask server 




</pre>


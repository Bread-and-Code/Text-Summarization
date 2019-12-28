# THE TASKS AT HAND
1. Third party implementation of Attention mechanism because I just found out there is a huge limitation in the encoder/decoder model
and that is as follows:<br><i>
A potential issue with this encoder-decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences. The performance of a basic encoder-decoder deteriorates rapidly as the length of an input sentence increases.</i>
<br>
So, instead of using simple LSTM model we use Attention mechanism:<br>
<h3>Advantage:</h3>
instead of looking at all the words in the source sequence, we can increase the importance of specific parts of the source sequence that result in the target sequence.
Basically we are trying to find the relation between different word in the source and the target sentences.<br>
<h1>NEXT</h1>
Before going forward with the actual dataset, i.e, our research papers in the drive. I have found a nice dataset in kaggle on Customer Reviews
so lets just build our summarizer on that scaling as a working model and then we go forward with the <b>BIG PLAN</b>.<br>
<h1>FILES TO BE MADE:</h1>
<li>attention.py X</li>
<li>libraries.py X</li>
<li>read.py X</li>
<li>duplicates.py X</li>
<li>preprocess.py X</li>
<li>data_clean.py X</li>
<li>read_sum.py X</li>
<li>sum_clean.py X</li>
<li>append.py X</li>
<li>display.py X</li>
<li>vizal_distrb.py X</li>
<li>maxlen.py X</li>
<li>split.py X</li> 
<li>txt_tokenizer.py X</li>
<li>sum_tokenizer.py X</li>
<li>DNN_model.py X</li>
<li>callback.py X</li>
<li>fit.py X</li>
<li>viz_plot.py X</li>
<li>rev_dict.py X</li>
<li>infer_setup.py X</li>
<li>infer_process.py X</li>
<li>int2txt.py X</li>
<li>predictions.py X</li>
So this phase is done
Go Check <b>Notebook</b> Folder

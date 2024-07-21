# Music-Generation-using-RNN
This is a machine learning based project which uses **Character RNN** to generate music based on strings. This model is then sampled to create never before heard piano tunes. 

This model learn the patterns in the ABC strings, and tries to predict the next characters for the provided string. RNNs maintain an internal state that depends on previously seen elements, so information about all characters seen up until a given moment will be taken into account in generating the prediction.

For vectorization(numerical representation of the string), two lookup tables have been used to convert characters to numbers and vice versa. 
The next step is to actually divide the text into example sequences that will be used during training. Each input sequence fed into the RNN will contain seq_length characters from the text. A target sequence for each input sequence also need to be defined, which will be used in training the RNN to predict the next character.
Then using the Batch Method, the stream of character indices can be converted into sequences of desired size.
The model is based off the LSTM architecture, where a state vector is used to maintain information about the temporal relationships between consecutive characters.

![alt text](https://github.com/Mehulgoyal353/Music-Generation-using-RNN/blob/main/RNN.png)

The final output of the LSTM is then fed into a fully connected Dense layer where we'll output a softmax over each character in the vocabulary, and then sample from this distribution to predict the next character.
To get actual predictions from the model, output distribution is to be sampled from, which is defined by a softmax over the character vocabulary. This means a categorical distribution is to be used to sample over the example prediction. This gives a prediction of the next character (specifically its index) at each timestep.

![alt text](https://github.com/Mehulgoyal353/Music-Generation-using-RNN/blob/main/Softmax.png)

To train the model on this classification task, a form of the crossentropy loss (negative log likelihood loss) can be used. Specifically, the sparse_categorical_crossentropy loss is to be used. It utilizes integer targets for categorical classification tasks.

## The Prediction Procedure

+ Initialize a "seed" start string and the RNN state, and set the number of characters we want to generate.
+ Use the start string and the RNN state to obtain the probability distribution over the next predicted character.
+ Sample from multinomial distribution to calculate the index of the predicted character. This predicted character is then used as the next input to the model.
+ At each time step, the updated RNN state is fed back into the model, so that it now has more context in making the next prediction. After predicting the next character, the updated RNN states are again fed back into the model, which is how it learns sequence dependencies in the data, as it gets more information from the previous predictions.





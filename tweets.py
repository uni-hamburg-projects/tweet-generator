import gensim
from tensorflow.keras import Sequential
from pathlib import Path
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from functools import reduce
from collections import Counter
from trainGenerator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import os


def data_prepare():
    with open("tweetsTrump.txt", errors="ignore") as f:
        content = f.readlines()
        tweets = []
        for tweet in content:
            if ((tweet.find("http") == -1) and (tweet.find("https") == -1) and (tweet.find("www")== -1)):
                tweet = tweet.replace('“',' ').replace('”',' ').replace('. ',' ').replace(',',' ').replace('!',' ').replace('?',' ').replace('.@','@').replace('&amp;','and').replace(': ',' ').replace('– ',' ').replace('- ',' ').replace('— ',' ').replace('--',' ').replace('....',' ').replace('...',' ').replace('"','').replace(':…','').replace(':','').replace("'","").replace(" )",")").replace(").",")").replace('.','').replace('#',' #').replace('-','').replace('‘','').replace('-','').replace('()','').replace(';','')
                tweet = gensim.parsing.preprocessing.strip_multiple_whitespaces(tweet)
                tweet = tweet.strip() #remove initial and lingering white spaces
                if not tweet.isspace() and tweet:
                    tweets.append(tweet.lower())
            else:
                if tweet.find("http") != -1:
                    tweet = ' '.join(word for word in tweet.split(' ') if not word.__contains__('http'))
                if tweet.find("https") != -1:
                    tweet = ' '.join(word for word in tweet.split(' ') if not word.__contains__('https'))
                if tweet.find("www") != -1:
                    tweet = ' '.join(word for word in tweet.split(' ') if not word.__contains__('www'))
                tweet = tweet.replace('“',' ').replace('”',' ').replace('. ',' ').replace(',',' ').replace('!',' ').replace('?',' ').replace('.@','@').replace('&amp;','and').replace(': ',' ').replace('– ',' ').replace('- ',' ').replace('— ',' ').replace('--',' ').replace('....',' ').replace('...',' ').replace('"','').replace(':…','').replace(':','').replace("'","").replace(" )",")").replace(").",")").replace('.','').replace('#',' #').replace('-','').replace('‘','').replace('-','').replace('()','').replace(';','')
                tweet = gensim.parsing.preprocessing.strip_multiple_whitespaces(tweet)
                tweet = tweet.strip() #remove initial and lingering white spaces
                if not tweet.isspace() and tweet:
                    tweets.append(tweet.lower())

    with open('tweets.txt', "w") as file:
        for t in tweets:
            file.write(t)
            file.write("\n")


def word2vec_train():
    documents = []
    with open("tweets.txt", errors="ignore") as f:
        content = f.readlines()
        for tweet in content:
            # add preprocess stuff
            tweet = tweet.split()
            documents.append(tweet)

    for s in documents:
        s.append('endchar') #add indicator for end of tweet
    maxlen = 0
    for i in range(0, len(documents)):
        if len(documents[i])> maxlen:
            maxlen = len(documents[i])
    
    #Write most common start words to a file
    start_comb = [None] * len(documents)
    for i in range(0,len(documents)):
        if(len(documents[i])>1):
            start_comb[i] = documents[i][0] + " " + documents[i][1]

    counts = Counter(start_comb)
    new_list = sorted(start_comb, key=lambda x: -counts[x])
    unique = reduce(lambda l, x: l + [x] if x not in l else l, new_list, [])

    with open('start_words.txt', "w") as file:
        for t in unique:
            if(not t==None):
                file.write(t)
                file.write("\n")

    model = gensim.models.Word2Vec(
        documents,
        size=100,
        window=10,
        min_count=1)

    model.train(documents, total_examples=len(documents), epochs=10)
    model.save("w2v_trump.model")
    return maxlen, documents


def lstm_model(emdedding_size,maxlen):
    model = Sequential()
    model.add(LSTM(units=emdedding_size, return_sequences=True, input_shape=[maxlen, embedding_size]))
    model.add(LSTM(units=emdedding_size))
    model.add(Dense(units=embedding_size, activation = 'linear'))
    model.compile(optimizer='adam', loss='mean_squared_error') #square-error loss
    return model


def sample(model, word_vector, mode):
    wV = word_vector.reshape(100)
    if(mode == 'greedy'):
        candidate = model.wv.similar_by_vector(wV, topn=5)
        return candidate[0][0]
    else:
        #randomly select one of the 5 most similar words
        candidates = model.wv.similar_by_vector(wV, topn=3)
        ind = np.random.randint(3,size=1)
        #selected = np.random.choice(candidates[0], 1,  p=[0.3, 0.25, 0.2, 0.15, 0.1])
        selected = candidates[int(ind)][0]
        return selected


def generate_next(model, text, num_generated=30):
    generatedText = text.split()
    for i in range(num_generated):
        pad = np.repeat("padchar", 60 - len(generatedText)).tolist()
        curText = pad + generatedText
        wordVecs = []
        for word in curText: 
            if word in model.wv.vocab:
                wordVec = model.wv[word]
            else:
                wordVec = np.zeros(100)
            wordVecs.append(wordVec)
        x = np.array(wordVecs).reshape(1,len(curText),100)
        prediction = nn_model.predict(x)  # prediction will be a vector of 100 numbers
        pred_word = sample(model, prediction, mode='greedy')
        generatedText.append(pred_word)
        if (pred_word == 'endchar'):
            break
    return ' '.join(vec for vec in generatedText)


def on_epoch_end(epoch, _):
    # do weight checkpoint
    if epoch%5 == 0:
        '''
        model_json = nn_model.to_json()
        with open("Checkpoints/model-epoch:" + str(epoch) + ".json", "w") as json_file:
            json_file.write(model_json)
        '''

        path = os.path.join("Checkpoints", "model-epoch-" + str(epoch) + "." + "h5")
        nn_model.save_weights(path)
        print("Saved model to disk")

    texts = [] 
    max_seeds = 6
    cur_seeds = 0
    with open("start_words.txt", errors="ignore") as f:
        content = f.readlines()
        for start_words in content:
            texts.append(start_words)     
            cur_seeds += 1
            if(cur_seeds >= max_seeds):
                break
    print('\nGenerating text after epoch: %d' % epoch)
    with open('epoch_end.txt', "a") as file:
        file.write('Generating text after epoch: %d' % epoch)
        file.write("\n")
        for text in texts:
            sample = generate_next(model, text)
            print('%s... -> %s' % (text, sample))
            file.write('%s... -> %s' % (text, sample))
            file.write("\n")
        file.write("\n")


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.05, x_coords.max()+0.05)
    plt.ylim(y_coords.min()+0.05, y_coords.max()+0.05)
    plt.savefig('tsne_'+str(word)+'.png')
    plt.show()


def generateTrainAndValidationSet(sentences, maxlen):
    train_x_file = Path("train_x.txt")
    train_y_file = Path("train_y.txt")
    val_x_file = Path("val_x.txt")
    val_y_file = Path("val_y.txt")
    
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    
    if not (train_x_file.is_file() & train_y_file.is_file() & val_x_file.is_file() & val_y_file.is_file()):
        data_x = []
        data_y= []
        for sentence in sentences:
            max = len(sentence)-1
            for i in range(0, max):
                if i != max-1:
                    a = np.repeat("padchar", max-1-i).tolist()
                    b = sentence[0:(i+1)]
                    a += b
                    pad = np.repeat("padchar", maxlen - max).tolist()
                    pad += a
                    data_x.append(pad)
                    data_y.append(sentence[i+1])
                else:
                    pad = np.repeat("padchar", maxlen - max).tolist()
                    pad += sentence[0:(i+1)]
                    data_x.append(pad)
                    data_y.append(sentence[i+1]) 
             

        #split into training and validation set
        train_X,  val_X, train_Y, val_Y = train_test_split(data_x, data_y, test_size=0.10, random_state=42)
        #write training and validation set to file
        with open('train_x.txt', "a") as file:
            for sentence in train_X:
                file.write(' '.join(word for word in sentence))
                file.write("\n")

        with open('train_y.txt', "a") as file:
            for word in train_Y:
                file.write(word)
                file.write("\n")
                
        with open('val_x.txt', "a") as file:
            for sentence in val_X:
                file.write(' '.join(word for word in sentence))
                file.write("\n")

        with open('val_y.txt', "a") as file:
            for word in val_Y:
                file.write(word)
                file.write("\n")
    else:
        with open("train_x.txt", errors="ignore") as f:
            content = f.readlines()
            for tweet in content:
                tweet = tweet.replace("\n", "").split()
                train_X.append(tweet)

        with open("train_y.txt", errors="ignore") as f:
            content = f.readlines()
            for word in content:
                train_Y.append(word.replace("\n", ""))
        
        with open("val_x.txt", errors="ignore") as f:
            content = f.readlines()
            for tweet in content:
                tweet = tweet.replace("\n", "").split()
                val_X.append(tweet)

        with open("val_y.txt", errors="ignore") as f:
            content = f.readlines()
            for word in content:
                val_Y.append(word.replace("\n", ""))
    return train_X, train_Y, val_X, val_Y


if __name__== "__main__":
    data_prepare()
    maxlen, sentences = word2vec_train()
    model = gensim.models.Word2Vec.load("w2v_trump.model")
    print(model)
    display_closestwords_tsnescatterplot(model, "great")
    display_closestwords_tsnescatterplot(model, "fake")
    pretrained_weights = model.wv.vectors
    vocab_size, embedding_size = pretrained_weights.shape

    sentLens = []
    nn_model = lstm_model(embedding_size,maxlen)

    initial_epoch = 0
    #if directory not empty
    if os.listdir('Checkpoints'):
        checkpoints = os.listdir('Checkpoints')
        last_weights = checkpoints[-1]
        initial_epoch = last_weights.replace('model-epoch-', '').replace('.h5', '')
        nn_model.load_weights("Checkpoints/"+last_weights)
        print("Load weights from epoch " + str(initial_epoch))
        print("Continue training from epoch " + str(initial_epoch))

    print(nn_model.summary())
    
    train_X, train_Y, val_X, val_Y = generateTrainAndValidationSet(sentences, maxlen)

    BATCH_SIZE = 1000

    training_batch_generator = DataGenerator(train_X, train_Y, BATCH_SIZE, maxlen, model)
    validation_batch_generator = DataGenerator(val_X, val_Y, BATCH_SIZE, maxlen, model)

    print("Training on", str(len(train_X)), "examples, validating on", str(len(val_X)),"examples.\n")

    if not int(initial_epoch) == 0:
        initial_epoch = int(initial_epoch)+1

    # launch TensorBoard (data won't show up until after the first epoch)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    nn_model.fit_generator(generator=training_batch_generator, initial_epoch= int(initial_epoch), epochs=600, verbose=1,
                           callbacks=[LambdaCallback(on_epoch_end=on_epoch_end), tensorboard],
                           validation_data=validation_batch_generator)
    

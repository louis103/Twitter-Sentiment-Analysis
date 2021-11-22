from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import textblob
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model_ = load_model("SENTIMENTAl_TEXT_CLASSIFIER-3CAT-5000X50.h5")
def get_subjectivity(text):
    return textblob.TextBlob(text).sentiment.subjectivity
def get_polarity(text):
    return textblob.TextBlob(text).sentiment.polarity
def prepare(text):

    '''Function to predict sentiment class of the passed text'''

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model_.predict(xt).argmax(axis=1)
    # Print the predicted sentiment,text,subjectivity and polarity

    print(f"[TEXT] : {text}")
    print('[Prediction] : The sentiment is : ',sentiment_classes[yt[0]])
    print("[Subjectivity of text] : ",get_subjectivity("Yesterday was the worst day in all my bad days"))
    print("[Polarity of text] : ",get_polarity("Yesterday was the worst day in all my bad days"))



prepare(["Yesterday was the worst day in all my bad days"])


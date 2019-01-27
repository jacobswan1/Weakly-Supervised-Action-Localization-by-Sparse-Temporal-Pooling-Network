# Load arguments
import re
import torch
import pickle
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# Build up the language model and LSTM
class Language_encoder(nn.Module):
    """Args:
        Natural Language Text.
    """

    def __init__(self, input_dim, hidden_dim, batch_size, time_steps, args, output_dim=100,
                 num_layers=2, lstm_input_size=300):
        super(Language_encoder, self).__init__()
        """
            Load GLOVE pre-trained model first.
        """
        with open(args.path_to_glove, "rb") as input_file:
            self.glove = pickle.load(input_file)
        self.wordnet_lemmatizer = WordNetLemmatizer()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.time_steps = time_steps
        self.lstm_input_size = lstm_input_size

        # Define the LSTM/fc module
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.args = args

        # Load Glove Embedding Dictionary
        with open(args.path_to_glove, "rb") as input_file:
            self.glove_model = pickle.load(input_file)

    def language_preprocess(self, input_str):
        # convert to lowercase
        input_str = input_str.lower()
        # remove numbers
        input_str = re.sub(r'\d+', '', input_str)
        # remove punctuation
        input_str = re.sub(r'[^\w\s]', '', input_str)
        # remove whitespaces
        input_str = input_str.strip()
        # remove stop words
        stop_words = set(ENGLISH_STOP_WORDS)
        tokens = word_tokenize(input_str)
        words = [i for i in tokens if not i in stop_words]
        # stemming the words
        words = [self.wordnet_lemmatizer.lemmatize(word) for word in words]
        return words

    def zero_pad_feature(self, word_dict):
        # transform to tensor input, and zero padding the extra feature
        input_seq = torch.zeros(self.time_steps, len(word_dict), self.lstm_input_size).to(self.args.device)

        for i in range(len(word_dict)):
            for j in range(self.time_steps):
                if j < len(word_dict[i]):
                    # word2vec extracting
                    input_seq[j, i, :] = torch.from_numpy(self.glove_model[word_dict[i][j]]).to(self.args.device)
        return input_seq

    def forward(self, text):
        """
            Text will be pre-processed first, and zero-padded for LSTM input.
        """
        word_dict = []
        for lines in text:
            word_dict.append(self.language_preprocess(lines))
        input_seq = self.zero_pad_feature(word_dict)

        lstm_out, _ = self.lstm(input_seq)
        last_output = lstm_out[-1]
        output = self.linear(last_output)

        return output
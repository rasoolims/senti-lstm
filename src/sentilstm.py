from dynet import *
import random, sys, os, codecs, pickle, csv
from optparse import OptionParser
import numpy as np


class SentiLSTM:
    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train', dest='train_data', help='train data', metavar='FILE', default='')
        parser.add_option('--dev', dest='dev_data', help='dev data', metavar='FILE', default='')
        parser.add_option('--input', dest='input_data', help='input data', metavar='FILE', default='')
        parser.add_option('--output', dest='output_data', help='output data', metavar='FILE', default='')
        parser.add_option('--params', dest='params', help='Parameters file', metavar='FILE', default='params.pickle')
        parser.add_option('--embed', dest='embed', help='Word embeddings', metavar='FILE')
        parser.add_option('--model', dest='model', help='Load/Save model file', metavar='FILE', default='model.model')
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
        parser.add_option('--batch', type='int', dest='batchsize', default=128)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--outdir', type='string', dest='output', default='')
        return parser.parse_args()

    def __init__(self, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        self.lstm_dims = options.lstm_dims
        self.num_labels = 2

        if options.train_data != None:
            labels = set()
            tf = codecs.open(options.train_data, 'r')
            for row in tf:
                labels.add(row.split('\t')[1])

            self.rev_labels = list(labels)
            self.label_dict = {label:i for i,label in enumerate(self.rev_labels)}
            self.num_labels = len(self.rev_labels)
            print 'loaded labels#:',self.num_labels

            to_save_params = []
            to_save_params.append(self.rev_labels)
            to_save_params.append(self.label_dict)
            to_save_params.append(self.num_labels)
            fp = codecs.open(options.embed, 'r')
            fp.readline()
            self.embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in fp}
            fp.close()
            self.dim = len(self.embed.values()[0])
            self.word_dict = {word: i+1 for i, word in enumerate(self.embed)}
            self.embed_lookup = self.model.add_lookup_parameters((len(self.word_dict), self.dim))
            self.embed_lookup.set_updated(False)
            for word, i in self.word_dict.iteritems():
                self.embed_lookup.init_row(i, self.embed[word])
            to_save_params.append(self.word_dict)
            to_save_params.append(self.dim)
            print 'Loaded word embeddings. Vector dimensions:', self.dim

            inp_dim = self.dim
            self.builders = [LSTMBuilder(1, inp_dim, self.lstm_dims, self.model),
                             LSTMBuilder(1, inp_dim, self.lstm_dims, self.model)]
            self.hid_dim = options.hidden_units
            self.hid2_dim = options.hidden2_units
            self.hid_inp_dim = options.lstm_dims * 2
            self.H1 = self.model.add_parameters((self.hid_dim, self.hid_inp_dim))
            self.H2 = None if self.hid2_dim == 0 else self.model.add_parameters((self.hid2_dim, self.hid_dim))
            last_hid_dims = self.hid2_dim if self.hid2_dim > 0 else self.hid_dim
            self.O = self.model.add_parameters((self.num_labels, last_hid_dims))
            to_save_params.append(self.hid_dim)
            to_save_params.append(self.hid2_dim)
            to_save_params.append(self.hid_inp_dim)
            print 'writing params'
            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump(to_save_params, paramsfp)
            print 'wrote params'
        else:
            self.read_params(options.params)
            print 'loaded params'
            self.model.load(options.model)

    def read_params(self, f):
        with open(f, 'r') as paramsfp:
            saved_params = pickle.load(paramsfp)
        self.hid_inp_dim = saved_params.pop()
        self.hid2_dim = saved_params.pop()
        self.hid_dim = saved_params.pop()
        self.dim = saved_params.pop()
        self.word_dict = saved_params.pop()
        self.num_labels = saved_params.pop()
        self.label_dict = saved_params.pop()
        self.rev_labels = saved_params.pop()
        self.embed_lookup = self.model.add_lookup_parameters((len(self.word_dict), self.dim))
        inp_dim = self.dim
        self.builders = [LSTMBuilder(1, inp_dim, self.lstm_dims, self.model),
                         LSTMBuilder(1, inp_dim, self.lstm_dims, self.model)]
        self.H1 = self.model.add_parameters((self.hid_dim, self.hid_inp_dim))
        self.H2 = None if self.hid2_dim == 0 else self.model.add_parameters((self.hid2_dim, self.hid_dim))
        last_hid_dims = self.hid2_dim if self.hid2_dim > 0 else self.hid_dim
        self.O = self.model.add_parameters((self.num_labels, last_hid_dims))


if __name__ == '__main__':
    (options, args) = SentiLSTM.parse_options()
    senti_lstm = SentiLSTM(options)
    print senti_lstm.num_labels

from dynet import *
import os, codecs, pickle,time
from optparse import OptionParser
import numpy as np


class SentiLSTM:
    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train', dest='train_data', help='train data', metavar='FILE')
        parser.add_option('--dev', dest='dev_data', help='dev data', metavar='FILE')
        parser.add_option('--input', dest='input_data', help='input data', metavar='FILE')
        parser.add_option('--output', dest='output_data', help='output data', metavar='FILE')
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
        self.batchsize = options.batchsize
        self.trainer = AdamTrainer(self.model)
        self.lstm_dims = options.lstm_dims
        self.num_labels = 2

        if options.train_data != None:
            labels = set()
            tf = codecs.open(os.path.abspath(options.train_data), 'r')
            for row in tf:
                labels.add(row.strip().split('\t')[1])
            tf.close()

            self.rev_labels = list(labels)
            self.label_dict = {label:i for i,label in enumerate(self.rev_labels)}
            self.num_labels = len(self.rev_labels)
            print 'loaded labels#:',self.num_labels

            to_save_params = []
            to_save_params.append(self.rev_labels)
            to_save_params.append(self.label_dict)
            to_save_params.append(self.num_labels)
            fp = codecs.open(os.path.abspath(options.embed), 'r')
            fp.readline()
            embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in fp}
            fp.close()
            self.dim = len(embed.values()[0])
            self.word_dict = {word: i+1 for i, word in enumerate(embed)}
            self.embed_lookup = self.model.add_lookup_parameters((len(self.word_dict)+1, self.dim))
            self.embed_lookup.set_updated(False)
            for word, i in self.word_dict.iteritems():
                self.embed_lookup.init_row(i, embed[word])
            self.embed_lookup.init_row(0, [0]*self.dim)
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

    def build_graph(self, train_lines):
        errors = []
        H1 = parameter(self.H1)
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)

        for train_line in train_lines:
            words,label = train_line.strip().split('\t')
            label = self.label_dict[label]
            tokens = words.split()
            words = []
            for w in tokens:
                orig,trans = w,''
                if '|||' in w:
                    orig = w[:w.rfind('|||')]
                    trans = w[w.rfind('|||')+4:]

                if trans in self.word_dict:
                    words.append(self.word_dict[trans])
                elif orig in self.word_dict:
                    words.append(self.word_dict[orig])
                else:
                    words.append(0) # unknown translation
            word_embeddings = [self.embed_lookup[i] for i in words]
            f_init, b_init = [b.initial_state() for b in self.builders]
            fw = [x.output() for x in f_init.add_inputs(word_embeddings)]
            bw = [x.output() for x in b_init.add_inputs(reversed(word_embeddings))]

            input = concatenate([fw[-1],bw[-1]])
            if H2:
                r_t = O * rectify(dropout(H2 * (rectify(dropout(H1 * input,0.5))),0.5))
            else:
                r_t = O * (rectify(dropout(H1 * input,0.5)))
            err = pickneglogsoftmax(r_t, label)
            errors.append(err)
        return errors

    def train(self, options, best_acc):
        tf = codecs.open(options.train_data, 'r')
        instances = []
        sz = 0
        i = 0
        loss = 0
        start = time.time()
        for row in tf:
            instances.append(row)
            if len(instances)>=self.batchsize:
                errs = self.build_graph(instances)
                sum_errs = esum(errs)
                squared = -sum_errs  # * sum_errs
                loss += sum_errs.scalar_value()
                sum_errs.backward()
                self.trainer.update()
                sz += len(instances)
                i+= 1
                if i%1 == 0:
                    self.trainer.status()
                    print 'loss:',loss / sz,'time:',time.time()-start
                    start = time.time()
                    sz = 0
                    loss = 0

                    if options.dev_data != None:
                        correct = 0
                        all_dev_num = 0
                        fp = codecs.open(options.dev_data,'r')
                        for line in fp:
                            all_dev_num += 1
                            sentence,label = line.strip().split('\t')
                            predicted = self.predict(sentence.strip())
                            if predicted == label:
                                correct += 1
                        acc = float(correct)/all_dev_num
                        print 'acc', acc
                        if acc>best_acc:
                            best_acc = acc
                            print 'saving best accurary', best_acc
                            self.model.save(os.path.join(options.output, options.model))
                errs = []
                instances = []
                renew_cg()
        if len(instances)>=0:
            errs = self.build_graph(instances)
            sum_errs = esum(errs)
            squared = -sum_errs  # * sum_errs
            loss = sum_errs.scalar_value()
            sum_errs.backward()
            self.trainer.update()
            self.trainer.status()
            print 'loss:', loss / len(instances), 'time:', time.time() - start
            start = time.time()
            instances = []
            errs = []
            renew_cg()
            if options.dev_data != None:
                correct = 0
                all_dev_num = 0
                fp = codecs.open(options.dev_data, 'r')
                for line in fp:
                    all_dev_num += 1
                    sentence, label = line.strip().split('\t')
                    predicted = self.predict(sentence.strip())
                    if predicted == label:
                        correct += 1
                acc = float(correct) / all_dev_num
                print 'acc', acc
                if acc > best_acc:
                    best_acc = acc
                    print 'saving best accurary',best_acc
                    self.model.save(os.path.join(options.output, options.model))
        return best_acc

    def predict(self, sentence):
        renew_cg()
        H1 = parameter(self.H1)
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)

        tokens = sentence.split()
        words = []
        for w in tokens:
            if w in self.word_dict:
                words.append(self.word_dict[w])
            else:
                words.append(0)  # unknown translation
        word_embeddings = [self.embed_lookup[i] for i in words]
        f_init, b_init = [b.initial_state() for b in self.builders]
        fw = [x.output() for x in f_init.add_inputs(word_embeddings)]
        bw = [x.output() for x in b_init.add_inputs(reversed(word_embeddings))]

        input = concatenate([fw[-1], bw[-1]])
        if H2:
            r_t = O * rectify(H2 * (rectify(H1 * input)))
        else:
            r_t = O * (rectify(H1 * input))
        label = np.argmax(r_t.npvalue())
        return self.rev_labels[label]


if __name__ == '__main__':
    (options, args) = SentiLSTM.parse_options()
    senti_lstm = SentiLSTM(options)

    if options.train_data!=None:
        best_acc = float('-inf')
        for i in xrange(options.epochs):
            best_acc = senti_lstm.train(options, best_acc)
            print 'saving for iteration',i
            senti_lstm.model.save(os.path.join(options.output, options.model+'_iter_'+str(i)))
    if options.input_data != None:
        fp = codecs.open(os.path.abspath(options.input_data), 'r')
        fw = codecs.open(os.path.abspath(options.output_data), 'w')
        i = 0
        start = time.time()
        for line in fp:
           sen = line.strip().split('\t')[0]
           fw.write(sen+'\t'+senti_lstm.predict(sen)+'\n')
           i += 1
           if i%100==0: sys.stdout.write(str(i)+'...')
        fw.close()
        fp.close()
        sys.stdout.write('done in '+str(time.time()-start)+'\n')
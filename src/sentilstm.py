from dynet import *
import os, codecs, pickle,time
from optparse import OptionParser
import numpy as np
import random

class SentiLSTM:
    @staticmethod
    def parse_options():
        parser = OptionParser()
        parser.add_option('--train', dest='train_data', help='train data', metavar='FILE')
        parser.add_option('--dev', dest='dev_data', help='dev data', metavar='FILE')
        parser.add_option('--input', dest='input_data', help='input data', metavar='FILE')
        parser.add_option('--output', dest='output_data', help='output data', metavar='FILE')
        parser.add_option('--params', dest='params', help='Parameters file', metavar='FILE', default='params.pickle')
        parser.add_option('--embed', dest='embed', help='Word embeddings for fixed embeddings', metavar='FILE')
        parser.add_option('--senti', dest='sentiwn', help='Sentiwordnet file (word\tpos\tneg in each line)', metavar='FILE')
        parser.add_option('--cluster', dest='cluster', help='Word cluster file (cluster\tword\tfreq in each line)',
                          metavar='FILE')
        parser.add_option('--init', dest='embed_init', help='Word embeddings initialization for updateable embeddings', metavar='FILE')
        parser.add_option('--model', dest='model', help='Load/Save model file', metavar='FILE', default='model.model')
        parser.add_option('--epochs', type='int', dest='epochs', default=5)
        parser.add_option('--batch', type='int', dest='batchsize', default=128)
        parser.add_option('--lstmdims', type='int', dest='lstm_dims', default=200)
        parser.add_option('--hidden', type='int', dest='hidden_units', default=200)
        parser.add_option('--embed_dim', type='int', dest='embed_dim', help='learnable word embedding dimension', default=100)
        parser.add_option('--hidden2', type='int', dest='hidden2_units', default=0)
        parser.add_option('--pos_dim', type='int', dest='pos_dim', default=30)
        parser.add_option('--cluster_dim', type='int', dest='cluster_dim', default=50)
        parser.add_option('--dropout', type='float', dest='dropout', help='dropout probability', default=0.0)
        parser.add_option('--outdir', type='string', dest='output', default='')
        parser.add_option("--learn_embed", action="store_false", dest="learnEmbed", default=True,
                          help='Have additional word embedding input that is updatable; default true.')
        parser.add_option("--use_pos", action="store_true", dest="usepos", default=False,
                          help='Use pos tag information.')
        parser.add_option("--pool", action="store_true", dest="usepool", default=False,
                          help='Use average pool as input feature.')
        parser.add_option("--save_iters", action="store_true", dest="save_iters", default=False,
                          help='Save all iterations.')
        parser.add_option("--save_best", action="store_false", dest="save_best", default=True,
                          help='Save all iterations.')
        parser.add_option('--word_drop', type='float', dest='word_drop', default=0, help = 'Word dropout probability (good for fully supervised)')
        parser.add_option("--activation", type="string", dest="activation", default="relu")
        parser.add_option("--trainer", type="string", dest="trainer", default="adam",help='adam,sgd,momentum,adadelta,adagrad')
        return parser.parse_args()

    def __init__(self, options):
        self.model = Model() # Dynet's model.
        self.batchsize = options.batchsize # The number of training instances to be processed at a time.
        trainers = {'adam':AdamTrainer, 'sgd':SimpleSGDTrainer, 'momentum':MomentumSGDTrainer,'adadelta':AdadeltaTrainer,'adagrad':AdagradTrainer}
        self.trainer = trainers[options.trainer](self.model) # The updater (could be MomentumSGDTrainer or SimpleSGDTrainer as well).
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.dropout = options.dropout
        self.activation = self.activations[options.activation]
        self.lstm_dims = options.lstm_dims # The dimension of the LSTM output layer.
        self.num_labels = 2 # Default number of labels.
        self.use_u_embedds = options.learnEmbed # Use updatable word embeddings (default false).
        self.word_drop = options.word_drop
        self.pos_dim = options.pos_dim
        self.max_len = 0
        self.pad_id = 1
        self.save_best = options.save_best
        if options.train_data != None:
            if not os.path.isdir(options.output): os.mkdir(options.output)
            self.usepos = options.usepos
            self.pooling = options.usepool
            labels = set()
            tf = codecs.open(os.path.abspath(options.train_data), 'r')
            seen_words = set() # If we need to learn embeddings, we have to build seen_words.
            seen_pos_tags = set()
            for row in tf:
                spl = row.strip().split('\t')
                if len(spl[0].strip().split())>self.max_len:
                    self.max_len = len(spl[0].strip().split())
                for f in spl[0].strip().split():
                    if '|||' in f:
                        if not self.usepos:
                            assert f.count('|||') == 1
                            seen_words.add(f[:f.rfind('|||')])
                            seen_words.add(f[f.rfind('|||') + 3:])
                        else:
                            assert f.count('|||') == 2
                            seen_words.add(f[:f.rfind('|||')])
                            seen_words.add(f[f.find('|||') + 3:f.rfind('|||')])
                            seen_pos_tags.add(f[f.rfind('|||') + 3:])
                    else:
                        seen_words.add(f)

                labels.add(spl[1]) # The label is separated by tab at the end of line.
            tf.close()
            if options.learnEmbed: print 'number of seen words', len(seen_words)
            if self.usepos: print 'number of seen tags', len(seen_pos_tags)

            self.rev_labels = list(labels)
            self.label_dict = {label:i for i,label in enumerate(self.rev_labels)} # Lookup dictionary for label string values.
            self.num_labels = len(self.rev_labels) # Now changing to the number of actual labels.
            print 'loaded labels#:',self.num_labels

            to_save_params = [] # Bookkeeping the parameters to be saved.
            to_save_params.append(self.pooling)
            to_save_params.append(options.activation)
            to_save_params.append(self.pos_dim)
            to_save_params.append(self.usepos)
            to_save_params.append(self.rev_labels)
            to_save_params.append(self.label_dict)
            to_save_params.append(self.num_labels)
            self.embed_dim = options.embed_dim
            self.embed_updatable_lookup = self.model.add_lookup_parameters(
                (len(seen_words) + 2, self.embed_dim)) if options.learnEmbed else None  # Updatable word embeddings.
            self.word_updatable_dict = {word: i + 2 for i, word in enumerate(seen_words)} # 0th index represent the OOV.
            if options.embed_init!=None:
                fp = codecs.open(os.path.abspath(options.embed_init), 'r')
                embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in fp}
                self.embed_dim = len(embed.values()[0])
                for word in embed.keys():
                    if self.word_updatable_dict.has_key(word):
                        self.embed_updatable_lookup.init_row(self.word_updatable_dict[word], embed[word])
            self.dim = 0
            self.word_dict = None
            self.use_fixed_embed = False
            if options.embed != None:
                self.use_fixed_embed = True
                fp = codecs.open(os.path.abspath(options.embed), 'r') # Reading the embedding vectors from file.
                fp.readline()
                embed = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in fp}
                fp.close()
                self.dim = len(embed.values()[0]) # Word embedding dimension.
                self.word_dict = {word: i+2 for i, word in enumerate(embed)}
                self.embed_lookup = self.model.add_lookup_parameters((len(self.word_dict) + 2, self.dim))
                self.embed_lookup.set_updated(False)  # This means that word embeddings cannot change over time.
                self.embed_lookup.init_row(0, [0] * self.dim)
                for word, i in self.word_dict.iteritems():
                    self.embed_lookup.init_row(i, embed[word])
            self.use_clusters = False
            self.cluster_dict = None
            self.word2cluster = dict()
            self.cluster_dim = options.cluster_dim
            if options.cluster != None:
                self.use_clusters = True
                for line in codecs.open(os.path.abspath(options.cluster), 'r'):
                    cluster,word,freq = line.split()
                    self.word2cluster[word] = cluster

                seen_clusters = set()
                for word in seen_words:
                    if word in self.word2cluster:
                        seen_clusters.add(self.word2cluster[word])

                for word in self.word2cluster.keys():
                    if not self.word2cluster[word] in seen_clusters:
                        del self.word2cluster[word]

                self.cluster_dict = {cluster: i+2 for i, cluster in enumerate(seen_clusters)}
                self.cluster_lookup = self.model.add_lookup_parameters((len(self.cluster_dict) + 2, self.cluster_dim))
                self.cluster_lookup.init_row(0, [0] * self.cluster_dim)
                print 'num of clusters',len(self.cluster_dict),', num of words',len(self.word2cluster)

            self.use_sentiwn = False
            self.sentiwn_dict = None
            if options.sentiwn != None:
                self.use_sentiwn = True
                fp = codecs.open(os.path.abspath(options.sentiwn), 'r')
                entries = {line.split()[0]: [float(f) for f in line.strip().split()[1:]] for line in fp}
                self.sentiwn_dict = {word: i + 2 for i, word in enumerate(entries)}
                self.senti_embed_lookup = self.model.add_lookup_parameters((len(self.sentiwn_dict) + 2, 2))
                self.senti_embed_lookup.set_updated(False)
                self.senti_embed_lookup.init_row(0, [0,0])
                for word, i in self.sentiwn_dict.iteritems():
                    self.senti_embed_lookup.init_row(i, entries[word])
                print 'loaded',len(entries),'sentiwordnet entries.'

            self.pos_dict = {pos:i+2 for i,pos in enumerate(seen_pos_tags)} if self.usepos else None
            if self.usepos:
                self.pos_embed_lookup = self.model.add_lookup_parameters((len(self.pos_dict)+2, self.pos_dim))
                self.pos_embed_lookup.set_updated(True)
            if options.learnEmbed: self.embed_updatable_lookup.set_updated(True)

            to_save_params.append(self.cluster_dim)
            to_save_params.append(self.use_clusters)
            to_save_params.append(self.cluster_dict)
            to_save_params.append(self.word2cluster)
            to_save_params.append(self.pos_dict)
            to_save_params.append(self.word_dict)
            to_save_params.append(self.sentiwn_dict)
            to_save_params.append(self.use_sentiwn)
            to_save_params.append(self.word_updatable_dict)
            to_save_params.append(self.dim)
            to_save_params.append(self.embed_dim)
            to_save_params.append(self.use_fixed_embed)
            print 'Loaded word embeddings. Vector dimensions:', self.dim

            inp_dim = self.dim + (self.embed_dim if options.learnEmbed else 0) + (self.pos_dim if self.usepos else 0) \
                      + (2 if self.use_sentiwn else 0) + (self.cluster_dim if self.use_clusters else 0)
            self.builders = [LSTMBuilder(1, inp_dim, self.lstm_dims, self.model),
                             LSTMBuilder(1, inp_dim, self.lstm_dims, self.model)] # Creating two lstms (forward and backward).
            self.hid_dim = options.hidden_units
            self.hid2_dim = options.hidden2_units
            self.hid_inp_dim = options.lstm_dims * 2 + (inp_dim if self.pooling else 0)
            self.H1 = self.model.add_parameters((self.hid_dim, self.hid_inp_dim))
            self.H2 = None if self.hid2_dim == 0 else self.model.add_parameters((self.hid2_dim, self.hid_dim))
            last_hid_dims = self.hid2_dim if self.hid2_dim > 0 else self.hid_dim
            self.O = self.model.add_parameters((self.num_labels, last_hid_dims)) # Output layer.
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
        self.use_fixed_embed = saved_params.pop()
        self.embed_dim = saved_params.pop()
        self.dim = saved_params.pop()
        self.word_updatable_dict = saved_params.pop()
        self.use_sentiwn = saved_params.pop()
        self.sentiwn_dict = saved_params.pop()
        self.word_dict = saved_params.pop()
        self.pos_dict = saved_params.pop()
        self.word2cluster = saved_params.pop()
        self.cluster_dict = saved_params.pop()
        self.use_clusters = saved_params.pop()
        self.cluster_dim = saved_params.pop()
        self.num_labels = saved_params.pop()
        self.label_dict = saved_params.pop()
        self.rev_labels = saved_params.pop()
        self.usepos = saved_params.pop()
        self.pos_dim = saved_params.pop()
        self.activation = self.activations[saved_params.pop()]
        self.pooling = saved_params.pop()
        self.use_u_embedds = True if len(self.word_updatable_dict)>1 else False
        self.embed_updatable_lookup = self.model.add_lookup_parameters(
            (len(self.word_updatable_dict) + 1, self.embed_dim)) if self.use_u_embedds else None
        self.embed_lookup = self.model.add_lookup_parameters((len(self.word_dict) + 1, self.dim)) if self.use_fixed_embed else None
        self.cluster_lookup = self.model.add_lookup_parameters((len(self.cluster_dict) + 1, self.cluster_dim)) if self.use_clusters else None
        self.senti_embed_lookup = self.model.add_lookup_parameters((len(self.sentiwn_dict) + 1, 2)) if self.use_sentiwn else None
        self.pos_embed_lookup = self.model.add_lookup_parameters((len(self.pos_dict), self.pos_dim)) if self.usepos else None
        inp_dim = self.dim + (self.embed_dim if self.use_u_embedds else 0) + (self.pos_dim if self.usepos else 0) \
                  + (2 if self.use_sentiwn else 0) + (self.cluster_dim if self.use_clusters else 0)
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
            senti_word_ids = []
            wordsu = [] # For adding updatable word indices.
            pos_tags = []
            clusters = []
            for w in tokens:
                orig,trans,pos = w,'',''
                wordu = orig
                if '|||' in w:
                    if not self.usepos:
                        orig = w[:w.rfind('|||')]
                        trans = w[w.rfind('|||')+3:]
                    else:
                        orig = w[:w.find('|||')]
                        trans = w[w.find('|||')+3:w.rfind('|||') + 3:]
                        pos = w[w.rfind('|||')+3:]
                    wordu = trans
                    pos_tags.append(self.pos_dict[pos]) if self.usepos else pos_tags.append(0)
                else:
                    pos_tags.append(0)
                    wordu = w

                if self.word_updatable_dict.has_key(wordu) and random.uniform(0,1)>=self.word_drop: # If in-vocabulary and no need to drop it out.
                    wordsu.append(self.word_updatable_dict[wordu])
                else:
                    wordsu.append(0)

                if self.use_fixed_embed:
                    if trans in self.word_dict:
                        words.append(self.word_dict[trans])
                    elif orig in self.word_dict:
                        words.append(self.word_dict[orig])
                    else:
                        words.append(0) # unknown translation
                else:
                    words.append(0)

                if self.use_sentiwn:
                    if orig.lower() in self.sentiwn_dict:
                        senti_word_ids.append(self.sentiwn_dict[orig.lower()])
                    else:
                        senti_word_ids.append(0)
                else:
                    senti_word_ids.append(0)

                if self.use_clusters:
                    if trans in self.word2cluster:
                        clusters.append(self.cluster_dict[self.word2cluster[trans]])
                    elif orig in self.word2cluster:
                        clusters.append(self.cluster_dict[self.word2cluster[orig]])
                    else: clusters.append(0)
                else: clusters.append(0)

            '''
             # padding
            if len(words)<self.max_len:
                for i in range(len(words),self.max_len):
                    words.append(self.pad_id)
                    clusters.append(self.pad_id)
                    senti_word_ids.append(self.pad_id)
                    wordsu.append(self.pad_id)
                    pos_tags.append(self.pad_id)
            '''
            word_embeddings = [self.embed_lookup[i] if self.use_fixed_embed else None for i in words]
            cluster_embeddings = [self.cluster_lookup[i] if self.use_clusters else None for i in clusters]
            senti_embeddings = [self.senti_embed_lookup[i] if self.use_sentiwn else None for i in senti_word_ids]
            updatable_embeddings = [self.embed_updatable_lookup[wordsu[i]]  if self.use_u_embedds else None for i in xrange(len(wordsu))]
            tag_embeddings = [self.pos_embed_lookup[pos_tags[i]] if self.usepos else None for i in xrange(len(pos_tags))]
            seq_input = [concatenate(filter(None, [word_embeddings[i],updatable_embeddings[i],tag_embeddings[i],senti_embeddings[i],cluster_embeddings[i]])) for i in xrange(len(wordsu))]
            if not self.pooling: pool_input = None
            else:
                pool_input = seq_input[0]
                for i in range(1,len(seq_input)):
                    pool_input += seq_input[i]
                pool_input /= len(seq_input)
            f_init, b_init = [b.initial_state() for b in self.builders]
            fw = [x.output() for x in f_init.add_inputs(seq_input)]
            bw = [x.output() for x in b_init.add_inputs(reversed(seq_input))]

            input = concatenate(filter(None,[fw[-1],bw[-1],pool_input]))
            # I assumed that the activation function is ReLU; it is worth trying tanh as well.
            if H2:
                r_t = O * self.activation(dropout(H2 * (self.activation(dropout(H1 * input,self.dropout))),self.dropout))
            else:
                r_t = O * (self.activation(dropout(H1 * input,self.dropout)))
            err = pickneglogsoftmax(r_t, label) # Getting the softmax loss function value to backprop later.
            errors.append(err)
        return errors

    def train(self, options, best_acc):
        rows = codecs.open(options.train_data, 'r').read().strip().split('\n')
        random.shuffle(rows)
        instances = []
        sz = 0
        i = 0
        loss = 0
        start = time.time()
        self.max_len = 0
        for row in rows:
            sen_len = len(row.split('\t')[0].split())
            if sen_len>self.max_len: self.max_len = sen_len
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
                if i%1 == 0: # You can change this to report (and save model if required) less frequently.
                    self.trainer.status()
                    print 'loss:',loss / sz,'time:',time.time()-start,'max_len',self.max_len
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
                            if self.save_best:
                                print 'saving best accurary', best_acc
                                self.model.save(os.path.join(options.output, options.model))
                            else:
                                print 'best accurary', best_acc
                errs = []
                instances = []
                self.max_len = 0
                renew_cg()
        '''
        skipping the rest because it does have the same batch size
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
                    if self.save_best:
                        print 'saving best accurary',best_acc
                        self.model.save(os.path.join(options.output, options.model))
                    else:
                        print 'best accurary', best_acc
        '''
        return best_acc

    def predict(self, sentence):
        renew_cg()
        H1 = parameter(self.H1)
        H2 = parameter(self.H2) if self.H2 != None else None
        O = parameter(self.O)

        tokens = sentence.split()
        words = []
        wordsu = []
        pos_tags = []
        senti_word_ids = []
        clusters = []
        for w in tokens:
            word = w if not self.usepos else w[:w.rfind('|||')]
            tag = w[w.rfind('|||')+3:] if self.usepos else None
            pos_tags.append(self.pos_dict[tag]) if self.usepos else pos_tags.append(0)
            if self.use_fixed_embed:
                if word in self.word_dict:
                    words.append(self.word_dict[word])
                else:
                    words.append(0)  # unknown translation
            else:
                words.append(0)

            if word in self.word_updatable_dict:
                wordsu.append(self.word_updatable_dict[word])
            else:
                wordsu.append(0)

            if self.use_sentiwn:
                if word.lower() in self.sentiwn_dict:
                    senti_word_ids.append(self.sentiwn_dict[word.lower()])
                else:
                    senti_word_ids.append(0)
            else:
                senti_word_ids.append(0)

            if self.use_clusters and word in self.word2cluster:
                clusters.append(self.cluster_dict[self.word2cluster[word]])
            else:
                clusters.append(0)

        word_embeddings = [self.embed_lookup[i] if self.use_fixed_embed else None for i in words]
        cluster_embeddings = [self.cluster_lookup[i] if self.use_clusters else None for i in clusters]
        senti_embeddings = [self.senti_embed_lookup[i] if self.use_sentiwn else None for i in senti_word_ids]
        updatable_embeddings = [self.embed_updatable_lookup[wordsu[i]] if self.use_u_embedds else None for i in
                                xrange(len(wordsu))]
        tag_embeddings = [self.pos_embed_lookup[pos_tags[i]] if self.usepos else None for i in xrange(len(pos_tags))]
        seq_input = [concatenate(filter(None, [word_embeddings[i], updatable_embeddings[i], tag_embeddings[i],
                                               senti_embeddings[i], cluster_embeddings[i]])) for i in xrange(len(wordsu))]
        if not self.pooling:  pool_input = None
        else:
            pool_input = seq_input[0]
            for i in range(1, len(seq_input)):
                pool_input += seq_input[i]
            pool_input /= len(seq_input)
        f_init, b_init = [b.initial_state() for b in self.builders]
        fw = [x.output() for x in f_init.add_inputs(seq_input)]
        bw = [x.output() for x in b_init.add_inputs(reversed(seq_input))]

        input = concatenate(filter(None,[fw[-1], bw[-1],pool_input]))
        if H2:
            r_t = O * self.activation(H2 * (self.activation(H1 * input)))
        else:
            r_t = O * (self.activation(H1 * input))
        label = np.argmax(r_t.npvalue())
        return self.rev_labels[label]

if __name__ == '__main__':
    (options, args) = SentiLSTM.parse_options()
    print options
    senti_lstm = SentiLSTM(options)

    if options.train_data!=None:
        best_acc = float('-inf')
        for i in xrange(options.epochs):
            best_acc = senti_lstm.train(options, best_acc)
            if options.save_iters:
                print 'saving for iteration',i
                senti_lstm.model.save(os.path.join(options.output, options.model+'_iter_'+str(i)))
            else:
                print 'end of iteration', i
                print 'end of iteration', i
        senti_lstm.model.save(os.path.join(options.output, options.model + '.final'))
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
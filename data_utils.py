# Author: Jan Buys
# Code credit: Tensorflow seq2seq; BIST parser; pytorch master source

from collections import Counter
from collections import defaultdict
from pathlib import Path
import re

import torch

_EOS = 0

_LIN = 0
_RELU = 1
_TANH = 2
_SIG = 3

class ConllEntry:
  def __init__(self, id, form):
    self.id = id
    self.form = form
    self.norm = form 

class Sentence:
  """Container class for single example."""
  def __init__(self, conll, tokens):
    self.conll = conll
    self.word_tensor = torch.LongTensor(tokens).view(-1, 1)

  def __len__(self):
    return len(self.conll)

  def text_line(self):
    return ' '.join([entry.norm for entry in self.conll[1:]])

  @classmethod
  def from_vocab_conll(cls, conll, word_vocab, max_length=-1):
    tokens = [word_vocab.get_id(entry.norm) for entry in conll] + [_EOS]
    if max_length > 0 and len(tokens) > max_length:
      return cls(conll[:max_length], tokens[:max_length])
    return cls(conll, tokens)

class Vocab:
  def __init__(self, word_list, counts=None):
    self.words = word_list
    self.dic = {word: i for i, word in enumerate(word_list)}
    self.counts = counts

  def __len__(self):
    return len(self.words)

  def get_word(self, i):
    return self.words[i]

  def get_id(self, word):
    return self.dic[word]

  def form_vocab(self):
    return set(filter(lambda w: not w.startswith('UNK'), 
                      self.words))

  def write_vocab(self, fn):
    with open(fn, 'w') as fh:
      for word in self.words:
        fh.write(word + '\n')

  def write_count_vocab(self, fn, add_eos):
    assert self.counts is not None
    with open(fn, 'w') as fh:
      for i, word in enumerate(self.words):
        if i == 0 and add_eos:
          fh.write(word + '\t0\n')
        else:  
          fh.write(word + '\t' + str(self.counts[word]) + '\n')

  @classmethod
  def from_counter(cls, counter, add_eos=False):
    if add_eos:
      word_list = ['_EOS']
    else:
      word_list = []
    word_list.extend([entry[0] for entry in counter.most_common()])
    return cls(word_list, counter)

  @classmethod
  def read_vocab(cls, fn):
    with open(fn, 'r') as fh:
      word_list = []
      for line in fh:
        entry = line.rstrip('\n').split('\t')
        word_list.append(entry[0])
    return cls(word_list)

  @classmethod
  def read_count_vocab(cls, fn):
    with open(fn, 'r') as fh:
      word_list = []
      dic = {}
      for line in fh:
        entry = line[:-1].rstrip('\n').split('\t')
        if len(entry) < 2:
          entry = line[:-1].strip().split()
        assert len(entry) >= 2, line
        word_list.append(entry[0])
        dic[entry[0]] = int(entry[1])
    return cls(word_list, Counter(dic))


def create_length_histogram(sentences, working_path):
  token_count = 0
  missing_token_count = 0

  sent_length = defaultdict(int)
  for sent in sentences:
    sent_length[len(sent)] += 1
    token_count += len(sent)
    missing_token_count += min(len(sent), 50)
  lengths = list(sent_length.keys())
  lengths.sort()
  print('Num Tokens: %d. Num <= 50: %d (%.2f percent).'
        % (token_count, missing_token_count,
            missing_token_count*100/token_count))

  cum_count = 0
  with open(working_path + 'train.histogram', 'w') as fh:
    for length in lengths:
      cum_count += sent_length[length]
      fh.write((str(length) + '\t' + str(sent_length[length]) + '\t' 
                + str(cum_count) + '\n'))
  print('Created histogram')   


# Stanford/Berkeley parser UNK processing case 5 (English specific).
# Source class: edu.berkeley.nlp.PCFGLA.SimpleLexicon
def map_unk_class(word, is_sent_start, vocab, replicate_rnng=False):
  unk_class = 'UNK'
  num_caps = 0
  has_digit = False
  has_dash = False
  has_lower = False

  if replicate_rnng:
    # Replicating RNNG bug
    for ch in word:
      has_digit = ch.isdigit()
      has_dash = ch == '-'
      if ch.isalpha():
        has_lower = ch.islower()
        if not ch.islower():
          num_caps += 1
  else:
    for ch in word:
      has_digit = has_digit or ch.isdigit()
      has_dash = has_dash or ch == '-'
      if ch.isalpha():
        has_lower = has_lower or ch.islower() or ch.istitle() 
        if not ch.islower():
          num_caps += 1

  lowered = word.lower()
  if word[0].isupper() or word[0].istitle():
    if is_sent_start and num_caps == 1:
      unk_class += '-INITC'
      if lowered in vocab:
        unk_class += '-KNOWNLC'
    else:
      unk_class += '-CAPS'
  elif not word[0].isalpha() and num_caps > 0:
    unk_class += '-CAPS'
  elif has_lower:
    unk_class += '-LC'

  if has_digit:
    unk_class += '-NUM'
  if has_dash:
    unk_class += '-DASH'

  if len(word) >= 3 and lowered[-1] == 's':
    ch2 = lowered[-2]
    if ch2 != 's' and ch2 != 'i' and ch2 != 'u':
      unk_class += '-s'
  elif len(word) >= 5 and not has_dash and not (has_digit and num_caps > 0):
    # common discriminating suffixes
    suffixes = ['ed', 'ing', 'ion', 'er', 'est', 'ly', 'ity', 'y', 'al']
    for suf in suffixes:
      if lowered.endswith(suf):
        unk_class += '-' + suf
        break

  return unk_class


def read_sentences_given_fixed_vocab(txt_path, txt_name, working_path):
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')

  print('reading')
  sentences = []
  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*')
      tokens = [root]
      for word in line.split():
        tokens.append(ConllEntry(len(tokens), word))
      for j, node in enumerate(tokens):
        assert node.form in word_vocab
        tokens[j].word_id = word_vocab.get_id(node.form)   
      sentences.append(Sentence.from_vocab_conll(tokens, word_vocab))

  print('%d sentences read' % len(sentences))
  return (sentences, word_vocab)


def read_sentences_fixed_vocab(txt_path, txt_name, working_path):
  wordsCount = Counter()

  conll_sentences = []
  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*')
      tokens = [root]
      for word in line.split():
        tokens.append(ConllEntry(len(tokens), word))
      wordsCount.update([node.form for node in tokens])
      conll_sentences.append(tokens)
  print('%d sentences read' % len(conll_sentences))
  word_vocab = Vocab.from_counter(wordsCount, add_eos=True)
  word_vocab.write_count_vocab(working_path + 'vocab', add_eos=True)

  parse_sentences = []
  for sent in conll_sentences:
    for j, node in enumerate(sent): 
      sent[j].word_id = word_vocab.get_id(node.norm) 
    parse_sentences.append(Sentence.from_vocab_conll(sent, word_vocab))
  
  return (parse_sentences, word_vocab)


def read_sentences_create_vocab(txt_path, txt_name, working_path,
    use_unk_classes=True, replicate_rnng=False, max_length=-1): 
  wordsCount = Counter()

  conll_sentences = []
  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*')
      tokens = [root]
      for word in line.split():
        tokens.append(ConllEntry(len(tokens), word))
      wordsCount.update([node.form for node in tokens])
      conll_sentences.append(tokens)

  # For words, replace singletons with Berkeley UNK classes
  singletons = set(filter(lambda w: wordsCount[w] == 1, wordsCount.keys()))
  form_vocab = set(filter(lambda w: wordsCount[w] > 1, wordsCount.keys()))   

  wordsNormCount = Counter()
  for i, sentence in enumerate(conll_sentences):
    for j, node in enumerate(sentence):
      if node.form in singletons:
        if use_unk_classes:
          conll_sentences[i][j].norm = map_unk_class(node.form, j==1, 
              form_vocab, replicate_rnng)
        else:
          conll_sentences[i][j].norm = 'UNK'
    wordsNormCount.update([node.norm for node in conll_sentences[i]])
                             
  word_vocab = Vocab.from_counter(wordsNormCount, add_eos=True)

  print(str(len(singletons)) + ' singletons')
  print('Word vocab size %d' % len(word_vocab))

  word_vocab.write_count_vocab(working_path + 'vocab', add_eos=True)

  parse_sentences = []
  for sent in conll_sentences:
    for j, node in enumerate(sent): 
      sent[j].word_id = word_vocab.get_id(node.norm) 
    parse_sentences.append(Sentence.from_vocab_conll(sent, word_vocab,
        max_length))

  write_text(working_path + txt_name + '.txt', parse_sentences)

  return (parse_sentences,
          word_vocab)

def read_sentences_given_vocab(txt_path, txt_name, working_path, 
    use_unk_classes=True, replicate_rnng=False, max_length=-1): 
  word_vocab = Vocab.read_count_vocab(working_path + 'vocab')
  form_vocab = word_vocab.form_vocab()

  print('reading')
  sentences = []
  conll_sentences = []

  with open(txt_path + txt_name + '.txt', 'r') as txtFP:
    for line in txtFP:
      root = ConllEntry(0, '*root*')
      sentence = [root]
      for word in line.split():
        sentence.append(ConllEntry(len(sentence), word))
      conll_sentences.append(sentence) 

      for j, node in enumerate(sentence):
        if node.form not in form_vocab: 
          if use_unk_classes:
            sentence[j].norm = map_unk_class(node.form, j==1, form_vocab,
                                             replicate_rnng)
          else:
            sentence[j].norm = 'UNK'
        if sentence[j].norm in word_vocab.dic:
          sentence[j].word_id = word_vocab.get_id(sentence[j].norm)
        else: # back off to least frequent word
          sentence[j].word_id = len(word_vocab) - 1
          sentence[j].norm = word_vocab.get_word(sentence[j].word_id)
      sentences.append(Sentence.from_vocab_conll(sentence, word_vocab,
        max_length))

  write_text(working_path + txt_name + '.txt', sentences)

  return (sentences,
          word_vocab)


def write_text(fn, sentences):
  with open(fn, 'w') as fh:
    for sentence in sentences:
      fh.write(sentence.text_line() + '\n') 



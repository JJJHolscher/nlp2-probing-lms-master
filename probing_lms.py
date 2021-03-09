# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# __Probing Language Models__
#
# This notebook serves as a start for your NLP2 assignment on probing Language Models. This notebook will become part of the contents that you will submit at the end, so make sure to keep your code (somewhat) clean :-)
#
# __note__: This is only the second time anyone is doing this assignment. That's exciting! But it might well be the case that certain aspects are too unclear. Do not hesitate at all to reach to me once you get stuck, I'd be grateful to help you out.
#
# __note 2__: This assignment is not dependent on big fancy GPUs. I run all this stuff on my own 3 year old CPU, without any Colab hassle. So it's up to you to decide how you want to run it.
# %% [markdown]
# # Models
#
# For the Transformer models you are advised to make use of the `transformers` library of Huggingface: https://github.com/huggingface/transformers
# Their library is well documented, and they provide great tools to easily load in pre-trained models.

# %%
#
## Your code for initializing the transformer model(s)
#
# Note that most transformer models use their own `tokenizer`, that should be loaded in as well.
#
from transformers import GPT2Tokenizer, GPT2Model



model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# Note that some models don't return the hidden states by default.
# This can be configured by passing `output_hidden_states=True` to the `from_pretrained` method.


# %%
#
## Your code for initializing the rnn model(s)
#
# The Gulordava LSTM model can be found here:
# https://drive.google.com/file/d/19Lp3AM4NEPycp_IBgoHfLc_V456pmUom/view?usp=sharing
#
# N.B: I have altered the RNNModel code to only output the hidden states that you are interested in.
# If you want to do more experiments with this model you could have a look at the original code here:
# https://github.com/facebookresearch/colorlessgreenRNNs/blob/master/src/language_models/model.py
#
from collections import defaultdict
from lstm.model import RNNModel
import torch


model_location = 'gulordava/state_dict.pt'  # <- point this to the location of the Gulordava .pt file
lstm = RNNModel('LSTM', 50001, 650, 650, 2)
lstm.load_state_dict(torch.load(model_location))


# This LSTM does not use a Tokenizer like the Transformers, but a Vocab dictionary that maps a token to an id.
with open('lstm/vocab.txt', encoding='ISO-8859-1') as f:
    w2i = {w.strip(): i for i, w in enumerate(f)}

vocab = defaultdict(lambda: w2i["<unk>"])
vocab.update(w2i)

# %% [markdown]
# It is a good idea that before you move on, you try to feed some text to your LMs; and check if everything works accordingly.
# %% [markdown]
# # Data
#
# For this assignment you will train your probes on __treebank__ corpora. A treebank is a corpus that has been *parsed*, and stored in a representation that allows the parse tree to be recovered. Next to a parse tree, treebanks also often contain information about part-of-speech tags, which is exactly what we are after now.
#
# The treebank you will use for now is part of the Universal Dependencies project. I provide a sample of this treebank as well, so you can test your setup on that before moving on to larger amounts of data.
#
# Make sure you accustom yourself to the format that is created by the `conllu` library that parses the treebank files before moving on. For example, make sure you understand how you can access the pos tag of a token, or how to cope with the tree structure that is formed using the `to_tree()` functionality.

# %%
# READ DATA
from typing import List
from conllu import parse_incr, TokenList


# If stuff like `: str` and `-> ..` seems scary, fear not!
# These are type hints that help you to understand what kind of argument and output is expected.
def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))

    return ud_parses

# %% [markdown]
# # Generating Representations
#
# We now have our data all set, our models are running and we are good to go!
#
# The next step is now to create the model representations for the sentences in our corpora. Once we have generated these representations we can store them, and train additional diagnostic (/probing) classifiers on top of the representations.
#
# There are a few things you should keep in mind here. Read these carefully, as these tips will save you a lot of time in your implementation.
# 1. Transformer models make use of Byte-Pair Encodings (BPE), that chunk up a piece of next in subword pieces. For example, a word such as "largely" could be chunked up into "large" and "ly". We are interested in probing linguistic information on the __word__-level. Therefore, we will follow the suggestion of Hewitt et al. (2019a, footnote 4), and create the representation of a word by averaging over the representations of its subwords. So the representation of "largely" becomes the average of that of "large" and "ly".
#
#
# 2. Subword chunks never overlap multiple tokens. In other words, say we have a phrase like "None of the", then the tokenizer might chunk that into "No"+"ne"+" of"+" the", but __not__ into "No"+"ne o"+"f the", as those chunks overlap multiple tokens. This is great for our setup! Otherwise it would have been quite challenging to distribute the representation of a subword over the 2 tokens it belongs to.
#
#
# 3. **Important**: If you closely examine the provided treebank, you will notice that some tokens are split up into multiple pieces, that each have their own POS-tag. For example, in the first sentence the word "Al-Zaman" is split into "Al", "-", and "Zaman". In such cases, the conllu `TokenList` format will add the following attribute: `('misc', OrderedDict([('SpaceAfter', 'No')]))` to these tokens. Your model's tokenizer does not need to adhere to the same tokenization. E.g., "Al-Zaman" could be split into "Al-"+"Za"+"man", making it hard to match the representations with their correct pos-tag. Therefore I recommend you to not tokenize your entire sentence at once, but to do this based on the chunking of the treebank. <br /><br />
# Make sure to still incoporate the spaces in a sentence though, as these are part of the BPE of the tokenizer. That is, the tokenizer uses a different token id for `"man"`, than it does for `" man"`: the former could be part of `" woman"`=`" wo`"+`"man"`, whereas the latter would be the used in case *man* occurs at the start of a word. The tokenizer for GPT-2 adds spaces at the start of a token (represented as a `Ġ` symbol). This means that you should keep track whether the previous token had the `SpaceAfter` attribute set to `'No'`: in case it did not, you should manually prepend a `" "` ahead of the token.
#
#
# 4. The LSTM LM does not have the issues related to subwords, but is far more restricted in its vocabulary. Make sure you keep the above points in mind though, when creating the LSTM representations. You might want to write separate functions for the LSTM, but that is up to you.
#
#
# 5. **N.B.**: Make sure that when you run a sentence through your model, you do so within a `with torch.no_grad():` block, and that you have run `model.eval()` beforehand as well (to disable dropout).
#
#
# 6. **N.B.**: Make sure to use a token's ``["form"]`` attribute, and not the ``["lemma"]``, as the latter will stem any relevant morphological information from the token. We don't want this, because we want to feed well-formed, grammatical sentences to our model.
#
#
# I would like to stress that if you feel hindered in any way by the simple code structure that is presented here, you are free to modify it :-) Just make sure it is clear to an outsider what you're doing, some helpful comments never hurt.

# %%
# FETCH SENTENCE REPRESENTATIONS
from torch import Tensor
import pickle


# Should return a tensor of shape (num_tokens_in_corpus, representation_size)
# Make sure you correctly average the subword representations that belong to 1 token!

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer) -> Tensor:
    # check which function to use
    if isinstance(model, GPT2Model):
        representations = fetch_sen_reps_transformer(ud_parses, model, tokenizer)
    else:
        representations = fetch_sen_reps_lstm(ud_parses, model, tokenizer)

    # return the representations
    return representations

def fetch_sen_reps_transformer(ud_parses: List[TokenList], model, tokenizer) -> Tensor:
    full_representation = []

    # loop over the parsed sentences
    for sentence in ud_parses:
        space_after = 'No'
        sentence_ids = []
        subword_dict = {}

        # loop over the words in the parsed sentence
        for word_index, word in enumerate(sentence):
            # check if the previous word had a space after
            if (space_after == 'No'):
                token = word['form']
            else:
                token = " " + word['form']
            if (word['misc'] is not None):
                space_after = word['misc']['SpaceAfter']
            else:
                space_after = 'Yes'

            # tokenize the word
            tokenized_word = tokenizer(token, return_tensors='pt')
            subword_ids = tokenized_word['input_ids']

            # check if multiple subwords
            if (subword_ids.shape[1] > 1):
                chunk_index_list = []
                chunks = torch.chunk(subword_ids, subword_ids.shape[1], dim=1)
                for chunk_index, chunk in enumerate(chunks):
                    sentence_ids += chunk
                    chunk_index_list.append(word_index + chunk_index)
                subword_dict[word_index] = chunk_index_list
            else:
                sentence_ids += subword_ids

            # DEBUG: print the tokens
            #if (subword_ids.shape[1] > 1):
                #chunks = torch.chunk(subword_ids, subword_ids.shape[1], dim=1)
                #for chunk in chunks:
                    #print(chunk)
                    #back_converted = tokenizer.convert_ids_to_tokens(chunk)
                    #print(back_converted)
            #else:
                #print(subword_ids)
                #back_converted = tokenizer.convert_ids_to_tokens(subword_ids)
                #print(back_converted)

        # pass the sentence through the model
        sentence_ids = torch.cat(sentence_ids, dim=0)
        sentence_ids = sentence_ids.unsqueeze(dim=0)
        print(sentence_ids.shape)
        model.eval()
        with torch.no_grad():
            sentence_representation = model(sentence_ids)
            sentence_representation = sentence_representation.last_hidden_state

        # mean the subwords into one word representation
        new_sentence_representation = []
        current_word_index = 0
        sentence_representation = sentence_representation.squeeze()
        for word_index, word_rep in enumerate(sentence_representation):
            if word_index != current_word_index:
                continue
            elif word_index in subword_dict:
                average_word_representation = []
                for chunk_index in subword_dict[word_index]:
                    average_word_representation.append(sentence_representation[chunk_index])
                average_word_representation = torch.mean(torch.stack(average_word_representation), dim=0)
                new_sentence_representation.append(average_word_representation)
                current_word_index += len(subword_dict[word_index])
            else:
                new_sentence_representation.append(word_rep)
                current_word_index += 1

        # add the sentence representation to the list
        sentence_representation = torch.stack(new_sentence_representation, dim=0)
        full_representation.append(sentence_representation)

    # stack all sentence representations
    full_representation = torch.cat(full_representation, dim=0)

    # return the full representation
    return full_representation

def fetch_sen_reps_lstm(ud_parses: List[TokenList], model, tokenizer) -> Tensor:
    full_representation = []

    # loop over the parsed sentences
    for sentence in ud_parses:
        sentence_ids = []

        # loop over the words in the parsed sentence
        for word in sentence:
            # tokenize the word
            token_id = tokenizer[word['form']]
            token_id = torch.tensor([[token_id]])
            sentence_ids.append(token_id)

        # pass the sentence through the model
        sentence_ids_cat = torch.cat(sentence_ids, dim=1)
        model.eval()
        with torch.no_grad():
            # generate a hidden state
            hidden_state = model.init_hidden(sentence_ids_cat.shape[0])
            # take the representation of the sentence
            sentence_representation = model(sentence_ids_cat, hidden_state)

        # add the sentence representation to the list
        sentence_representation = sentence_representation.squeeze()
        full_representation.append(sentence_representation)

    # stack all sentence representations
    full_representation = torch.cat(full_representation, dim=0)

    # return the full representation
    return full_representation

# %% [markdown]
# To validate your activation extraction procedure I have set up the following assertion function as a sanity check. It compares your representation against a pickled version of mine.
#
# For this I used `distilgpt2`.

# %%
def error_msg(model_name, gold_embs, embs, i2w):
    with open(f'{model_name}_tokens1.pickle', 'rb') as f:
        sen_tokens = pickle.load(f)

    diff = torch.abs(embs - gold_embs)
    max_diff = torch.max(diff)
    avg_diff = torch.mean(diff)

    print(f"{model_name} embeddings don't match!")
    print(f"Max diff.: {max_diff:.4f}\nMean diff. {avg_diff:.4f}")

    print("\nCheck if your tokenization matches with the original tokenization:")
    for idx in sen_tokens.squeeze():
        if isinstance(i2w, list):
            token = i2w[idx]
        else:
            token = i2w.convert_ids_to_tokens(idx.item())
        print(f"{idx:<6} {token}")


def assert_sen_reps(model, tokenizer, lstm, vocab):
    with open('distilgpt2_emb1.pickle', 'rb') as f:
        distilgpt2_emb1 = pickle.load(f)

    with open('lstm_emb1.pickle', 'rb') as f:
        lstm_emb1 = pickle.load(f)

    corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')[:1]

    own_distilgpt2_emb1 = fetch_sen_reps(corpus, model, tokenizer)
    own_lstm_emb1 = fetch_sen_reps(corpus, lstm, vocab)

    assert distilgpt2_emb1.shape == own_distilgpt2_emb1.shape,         f"Distilgpt2 shape mismatch: {distilgpt2_emb1.shape} (gold) vs. {own_distilgpt2_emb1.shape} (yours)"
    assert lstm_emb1.shape == own_lstm_emb1.shape,         f"LSTM shape mismatch: {lstm_emb1.shape} (gold) vs. {own_lstm_emb1.shape} (yours)"

    if not torch.allclose(distilgpt2_emb1, own_distilgpt2_emb1, rtol=1e-3, atol=1e-3):
        error_msg("distilgpt2", distilgpt2_emb1, own_distilgpt2_emb1, tokenizer)
    if not torch.allclose(lstm_emb1, own_lstm_emb1, rtol=1e-3, atol=1e-3):
        error_msg("lstm", lstm_emb1, own_lstm_emb1, list(vocab.keys()))


assert_sen_reps(model, tokenizer, lstm, vocab)

# %% [markdown]
# Next, we should define a function that extracts the corresponding POS labels for each activation, which we do based on the **``"upostag"``** attribute of a token (so not the ``xpostag`` attribute). These labels will be transformed to a tensor containing the label index for each item.

# %%
# FETCH POS LABELS


# Should return a tensor of shape (num_tokens_in_corpus,)
# Make sure that when fetching these pos tags for your train/dev/test corpora you share the label vocabulary.
def fetch_pos_tags(ud_parses: List[TokenList], pos_vocab=None) -> Tensor:
    # check if the vocabulary is None
    if pos_vocab is None:
        # create a new vocab
        pos_vocab = defaultdict()

    # create a list of tags from the input
    all_tags = []

    # TODO: random nummer als value. Dus {upostag: random nummer}
    # loop over the parsed sentences
    word_index = 0
    for sentence in ud_parses:
        # loop over the words in the parsed sentence
        for word in sentence:
            # get the tag
            tag = word['upostag']

            # check if the tag does not exist in the vocabulary
            if tag not in pos_vocab:
                pos_vocab[tag] = word_index
                all_tags.append(torch.tensor(word_index))
            else:
                all_tags.append(torch.tensor(pos_vocab[tag]))

        # add to the word index
        word_index += 1

    # stack the tags into a tensor
    all_tags = torch.stack(all_tags, dim=0)

    # return the tags and vocabulary
    return all_tags, pos_vocab


# %%
import os

# Function that combines the previous functions, and creates 2 tensors for a .conllu file:
# 1 containing the token representations, and 1 containing the (tokenized) pos_tags.

def create_data(filename: str, lm, w2i, pos_vocab=None):
    ud_parses = parse_corpus(filename)

    sen_reps = fetch_sen_reps(ud_parses, lm, w2i)
    pos_tags, pos_vocab = fetch_pos_tags(ud_parses, pos_vocab=pos_vocab)

    return sen_reps, pos_tags, pos_vocab


lm = model  # or `lstm`
w2i = tokenizer  # or `vocab`
use_sample = True

train_x, train_y, train_vocab = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-train.conllu'),
    lm,
    w2i
)

dev_x, dev_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-dev.conllu'),
    lm,
    w2i,
    pos_vocab=train_vocab
)

test_x, test_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-test.conllu'),
    lm,
    w2i,
    pos_vocab=train_vocab
)

# %% [markdown]
# # Diagnostic Classification
#
# We now have our models, our data, _and_ our representations all set! Hurray, well done. We can finally move onto the cool stuff, i.e. training the diagnostic classifiers (DCs).
#
# DCs are simple in their complexity on purpose. To read more about why this is the case you could already have a look at the "Designing and Interpreting Probes with Control Tasks" by Hewitt and Liang (esp. Sec. 3.2).
#
# A simple linear classifier will suffice for now, don't bother with adding fancy non-linearities to it.
#
# I am personally a fan of the `skorch` library, that provides `sklearn`-like functionalities for training `torch` models, but you are free to train your dc using whatever method you prefer.
#
# As this is an Artificial Intelligence master and you have all done ML1 + DL, I expect you to use your train/dev/test splits correctly ;-)

# %%
# DIAGNOSTIC CLASSIFIER

# %% [markdown]
# # Trees
#
# For our gold labels, we need to recover the node distances from our parse tree. For this we will use the functionality provided by `ete3`, that allows us to compute that directly. I have provided code that transforms a `TokenTree` to a `Tree` in `ete3` format.

# %%
# In case you want to transform your conllu tree to an nltk.Tree, for better visualisation

def rec_tokentree_to_nltk(tokentree):
    token = tokentree.token["form"]
    tree_str = f"({token} {' '.join(rec_tokentree_to_nltk(t) for t in tokentree.children)})"

    return tree_str


def tokentree_to_nltk(tokentree):
    from nltk import Tree as NLTKTree

    tree_str = rec_tokentree_to_nltk(tokentree)

    return NLTKTree.fromstring(tree_str)


# %%
# !pip install ete3
from ete3 import Tree as EteTree


class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)

    def __str__(self):
        return self.get_ascii(show_internal=True)

    def __repr__(self):
        return str(self)


def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx

def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)

    return FancyTree(f"{newick_str};")


# %%
# Let's check if it works!
# We can read in a corpus using the code that was already provided, and convert it to an ete3 Tree.

def parse_corpus(filename):
    from conllu import parse_incr

    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))

    return ud_parses

corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')
item = corpus[0]
tokentree = item.to_tree()
ete3_tree = tokentree_to_ete(tokentree)
print(ete3_tree)

# %% [markdown]
# As you can see we label a token by its token id (converted to a string). Based on these id's we are going to retrieve the node distances.
#
# To create the true distances of a parse tree in our treebank, we are going to use the `.get_distance` method that is provided by `ete3`: http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html#working-with-branch-distances
#
# We will store all these distances in a `torch.Tensor`.
#
# Please fill in the gap in the following method. I recommend you to have a good look at Hewitt's blog post  about these node distances.

# %%
def create_gold_distances(corpus):
    all_distances = []

    for item in (corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        # Your code for computing all the distances comes here.

        all_distances.append(distances)

    return all_distances

# %% [markdown]
# The next step is now to do the previous step the other way around. After all, we are mainly interested in predicting the node distances of a sentence, in order to recreate the corresponding parse tree.
#
# Hewitt et al. reconstruct a parse tree based on a _minimum spanning tree_ (MST, https://en.wikipedia.org/wiki/Minimum_spanning_tree). Fortunately for us, we can simply import a method from `scipy` that retrieves this MST.

# %%
from scipy.sparse.csgraph import minimum_spanning_tree
import torch


def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()

    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.

    return mst

# %% [markdown]
# Let's have a look at what this looks like, by looking at a relatively short sentence in the sample corpus.
#
# If your addition to the `create_gold_distances` method has been correct, you should be able to run the following snippet. This then shows you the original parse tree, the distances between the nodes, and the MST that is retrieved from these distances. Can you spot the edges in the MST matrix that correspond to the edges in the parse tree?

# %%
item = corpus[5]
tokentree = item.to_tree()
ete3_tree = tokentree_to_ete(tokentree)
print(ete3_tree, '\n')

gold_distance = create_gold_distances(corpus[5:6])[0]
print(gold_distance, '\n')

mst = create_mst(gold_distance)
print(mst)

# %% [markdown]
# Now that we are able to map edge distances back to parse trees, we can create code for our quantitative evaluation. For this we will use the Undirected Unlabeled Attachment Score (UUAS), which is expressed as:
#
# $$\frac{\text{number of predicted edges that are an edge in the gold parse tree}}{\text{number of edges in the gold parse tree}}$$
#
# To do this, we will need to obtain all the edges from our MST matrix. Note that, since we are using undirected trees, that an edge can be expressed in 2 ways: an edge between node $i$ and node $j$ is denoted by both `mst[i,j] = 1`, or `mst[j,i] = 1`.
#
# You will write code that computes the UUAS score for a matrix of predicted distances, and the corresponding gold distances. I recommend you to split this up into 2 methods: 1 that retrieves the edges that are present in an MST matrix, and one general method that computes the UUAS score.

# %%
def edges(mst):
    edges = set()

    # Your code for retrieving the edges from the MST matrix

    return edges

def calc_uuas(pred_distances, gold_distances):
    uuas = None

    # Your code for computing the UUAS score

    return uuas

# %% [markdown]
# # Structural Probes
#
# We now have everything in place to start doing the actual exciting stuff: training our structural probe!
#
# To make life easier for you, we will simply take the `torch` code for this probe from John Hewitt's repository. This allows you to focus on the training regime from now on.

# %%
import torch.nn as nn
import torch


class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim

        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))

        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)

        batchlen, seqlen, rank = transformed.size()

        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)

        diffs = transformed - transposed

        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)

        return squared_distances


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()

        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=(1,2))
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents

        else:
            batch_loss = torch.tensor(0.0)

        return batch_loss, total_sents

# %% [markdown]
# I have provided a rough outline for the training regime that you can use. Note that the hyper parameters that I provide here only serve as an indication, but should be (briefly) explored by yourself.
#
# As can be seen in Hewitt's code above, there exists functionality in the probe to deal with batched input. It is up to you to use that: a (less efficient) method can still incorporate batches by doing multiple forward passes for a batch and computing the backward pass only once for the summed losses of all these forward passes. (_I know, this is not the way to go, but in the interest of time that is allowed ;-), the purpose of the assignment is writing a good paper after all_).

# %%
from torch import optim

'''
Similar to the `create_data` method of the previous notebook, I recommend you to use a method
that initialises all the data of a corpus. Note that for your embeddings you can use the
`fetch_sen_reps` method again. However, for the POS probe you concatenated all these representations into
1 big tensor of shape (num_tokens_in_corpus, model_dim).

The StructuralProbe expects its input to contain all the representations of 1 sentence, so I recommend you
to update your `fetch_sen_reps` method in a way that it is easy to retrieve all the representations that
correspond to a single sentence.
'''

def init_corpus(path, concat=False, cutoff=None):
    """ Initialises the data of a corpus.

    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]

    embs = fetch_sen_reps(corpus, model, tokenizer, concat=concat)
    gold_distances = create_gold_distances(corpus)

    return gold_distances, embs


# I recommend you to write a method that can evaluate the UUAS & loss score for the dev (& test) corpus.
# Feel free to alter the signature of this method.
def evaluate_probe(probe, _data):
    # YOUR CODE HERE

    return loss_score, uuas_score


# Feel free to alter the signature of this method.
def train(_data):
    emb_dim = 768
    rank = 64
    lr = 10e-4
    batch_size = 24

    probe = StructuralProbe(emb_dim, rank)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()

    for epoch in range(epochs):

        for i in range(0, len(corpus), batch_size):
            optimizer.zero_grad()

            # YOUR CODE FOR DOING A PROBE FORWARD PASS

            batch_loss.backward()
            optimizer.step()

        dev_loss, dev_uuas = evaluate_probe(probe, _dev_data)

        # Using a scheduler is up to you, and might require some hyper param fine-tuning
        scheduler.step(dev_loss)

    test_loss, test_uuas = evaluate_probe(probe, _test_data)

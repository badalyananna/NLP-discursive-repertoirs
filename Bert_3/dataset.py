import re

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

LABELS = {
    'anticipazione': 0,
    'causa': 1,
    'commento': 2,
    'conferma': 3,
    'considerazione': 4,
    'contrapposizione': 5,
    'deresponsabilizzazione': 6,
    'descrizione': 7,
    'dichiarazione di intenti':8,
    'generalizzazione': 9,
    'giudizio': 10,
    'giustificazione': 11,
    'implicazione': 12,
    'non risposta': 13,
    'opinione': 14,
    'possibilità': 15,
    'prescrizione': 16,
    'previsione': 17,
    'proposta': 18,
    'ridimensionamento': 19,
    'sancire': 20,
    'specificazione': 21,
    'valutazione': 22
}

A = 0
B = 1
C = 2

rep = dict()
rep['sancire'] = A
rep['descrizione'] = B
rep['specificazione'] = C
rep['possibilità'] = C
rep['opinione'] = A
# rep['RO'] = G + H 
rep['causa'] = A
rep['conferma'] = C
rep['non risposta'] = A
rep['contrapposizione'] = A
rep['implicazione'] = C
rep['giudizio'] = A
rep['previsione'] = A
rep['giustificazione'] = A
rep['commento'] = A
rep['generalizzazione'] = A
rep['valutazione'] = C
rep['dichiarazione di intenti'] = C
rep['proposta'] = B
rep['deresponsabilizzazione'] = A
rep['prescrizione'] = C
#rep['ridimensionamento'] = G + F + L + D + E + G + N
rep['ridimensionamento'] = C
rep['considerazione'] = B
rep['anticipazione'] = B

class HyperionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name):
        #fill_null_features(df)
        df = filter_empty_labels(df)
        #df = twitter_preprocess(df)
        # df = to_lower_case(df)
        df['Stralcio'] = df['Stralcio'].str.lower()
        df['Repertorio'] = df['Repertorio'].str.lower()
        uniform_labels(df)          
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = tokenizer(
        df['Stralcio'].tolist(),
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
        self.vectors = df['Repertorio'].apply(lambda x: rep[x]).to_numpy()
        self.labels = encode_labels(df['Repertorio'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        
        item['vectors'] = self.vectors[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
    def labels_list(self):
        return LABELS

# Dataset loading and preprocessing
def fill_null_features(df):
    """
    For each column in the list ['Domanda','Testo'], if the value of the cell is null, then replace it
    with the value of the previous cell to obtain a full dataset 
    
    :param df: the dataframe
    """
    for c in ['Domanda','Testo']:
        for i in range(0,len(df.index)):  
            if not df[c][i]:
                j=i
                while j>0: 
                    j-=1
                    if df[c][j]:
                        df[c][i] = df[c][j]
                        break


#Delete examples with empty label
def filter_empty_labels(df):
    filter = df["Repertorio"] != ""
    return df[filter]

#Convert to lower case
def to_lower_case(df):
    return df.applymap(str.lower)


#Lables uniformation uncased
def uniform_labels(df):
    df['Repertorio'].replace('implicazioni','implicazione', inplace=True)
    df['Repertorio'].replace('previsioni','previsione', inplace=True)


def encode_labels(repertori):
    #le = preprocessing.LabelEncoder()
    #le.fit(LABELS)
    #return le.transform(df['Repertorio'])
    encoded = []
    for i in repertori:
        encoded.append(LABELS[i])
    return np.array(encoded)


def encode_str_label(rep:str):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.transform([rep])

def decode_labels(encoded_labels):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.inverse_transform(encoded_labels)

def twitter_preprocess(text:str) -> str:
    """
    - It takes a string as input
    - It returns a string as output
    - It does the following:
        - Normalizes terms like url, email, percent, money, phone, user, time, date, number
        - Annotates hashtags
        - Fixes HTML tokens
        - Performs word segmentation on hashtags
        - Tokenizes the string
        - Replaces tokens extracted from the text with other expressions
        - Removes non-alphanumeric characters
        - Removes extra whitespaces
        - Removes repeated characters
        - Removes leading and trailing whitespaces
    
    :param text: The text to be processed
    :type text: str
    :return: A string with the preprocessed text.
    """
    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag"},
    fix_html=True,  # fix HTML tokens
    
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )

    text = str(" ".join(text_processor.pre_process_doc(text)))
    text = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r'(\w)\1{2,}',r'\1\1', text)
    text = re.sub ( r'^\s' , '' , text )
    text = re.sub ( r'\s$' , '' , text )

    return text
    

def train_val_split(df, tok_name,  val_perc=0.2, subsample = False):
    """
    It takes a dataframe, a tokenizer name, a validation percentage and a subsample flag. It then splits
    the dataframe into a training and validation set, and returns a HyperionDataset object for each
    
    :param df: the dataframe containing the data
    :param tok_name: the name of the tokenizer to use
    :param val_perc: the percentage of the data that will be used for validation
    :param subsample: if True, subsample the dataset to 50 samples per class, defaults to False
    (optional)
    :return: A tuple of two datasets, one for training and one for validation.
    """
    gb = df.groupby('Repertorio')
    train_list = []
    val_list = []
    for x in gb.groups:
        if subsample:
            class_df = gb.get_group(x).head(50)
        else:
            class_df = gb.get_group(x)

        # Validation set creation
        val = class_df.sample(frac=val_perc)
        train = pd.concat([class_df,val]).drop_duplicates(keep=False)

        #train_list.append(train.head(500))
        train_list.append(train)
        val_list.append(val)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    return HyperionDataset(train_df, tok_name), HyperionDataset(val_df, tok_name)

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def make_predictions(logits, reps=rep):
    logits = torch.nn.Sigmoid()(logits)
    results = []
    for i in range(len(logits)):
        logit = logits[i]
        max_dif = 0
        best = ''
        for r in reps:
            similarity = cos_sim(logit, reps[r])
            if similarity > max_dif:
                best = r
                max_dif = similarity
        results.append(best)

    return torch.from_numpy(encode_labels(results))
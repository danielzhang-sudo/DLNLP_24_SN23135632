import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length=64):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X.iloc[index].values[0]
        y = self.y.iloc[index].values

        # Tokenize the text
        encoded_input = self.tokenizer.encode_plus(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()
        token_type_ids = encoded_input['token_type_ids'].squeeze()

        text = {
            'ids':input_ids,
            'mask':attention_mask,
            'token_type_ids':token_type_ids
        }

        return input_ids, attention_mask, token_type_ids, torch.tensor(y, dtype=torch.float)

def data_preprocessing(args, task):

    path = args.path

    tree = ET.parse(path)
    sentences = tree.getroot()

    dict = {}
    counter = 0
    for _, sentence in enumerate(sentences):
        texts = []
        aspect_terms = []
        aspect_categories  = []
        term_polarities = []
        category_polarities = []

        text = sentence.find('text').text
        texts.append(text)

        aspect_term_elements = sentence.find('aspectTerms')
        if aspect_term_elements is not None:
            for aspect_term in aspect_term_elements.findall('aspectTerm'):
                
                term = aspect_term.get('term')
                term_polarity = aspect_term.get('polarity')
                aspect_terms.append(term)
                term_polarities.append(term_polarity)
        else:
            aspect_terms.append(None)
            term_polarities.append(None)
        
        aspect_category_elements = sentence.find('aspectCategories')
        if aspect_category_elements is not None:
            for aspect_category in aspect_category_elements.findall('aspectCategory'):
                category = aspect_category.get('category')
                category_polarity = aspect_category.get('polarity')
                aspect_categories.append(category)
                category_polarities.append(category_polarity)
        else:
            aspect_categories.append(None)
            category_polarities.append(None)

        for i in range(len(aspect_terms)):
            for j in range(len(aspect_categories)):
                dict[counter] = {
                    'text':texts[0],
                    'aspect_term':aspect_terms[i],
                    'term_polarity':term_polarities[i],
                    'aspect_category':aspect_categories[j],
                    'category_polarity':category_polarities[j]
                }
                counter += 1
            counter += 1

    df = pd.DataFrame.from_dict(dict).T.reset_index().drop(columns=['index'])

    if task == 'a':
        return get_data_A(df)
    elif task == 'b':
        return get_data_B(df)
    else:
        raise ValueError("Argument(s) 'task' must be 'a' or 'b'")


def get_data_A(df):

    df_one_hot = pd.get_dummies(df['aspect_category'], drop_first=False, dtype=int)
    df_one_hot = pd.concat([df, df_one_hot], axis=1)
    df_one_hot = df_one_hot.groupby(df_one_hot['text']).aggregate({'aspect_term':'first', 'term_polarity':'first', 'category_polarity':'first', 'ambience':'max', 'anecdotes/miscellaneous':'max', 'food':'max', 'price':'max', 'service':'max'})
    df_one_hot = df_one_hot.reset_index()

    print(df_one_hot.head())

    return split_dataset(df_one_hot)

def get_data_B(df):

    df_catsent_hot = pd.get_dummies(df['aspect_category'], drop_first=False, dtype=int)
    df_catsent_hot = pd.concat([df, df_catsent_hot], axis=1)
    for i, row in df_catsent_hot.iterrows():
        if row['category_polarity'] == 'negative':
            df_catsent_hot.loc[i, row['aspect_category']] = 1
        elif row['category_polarity'] == 'neutral':
            df_catsent_hot.loc[i, row['aspect_category']] = 2
        elif row['category_polarity'] == 'conflict':
            df_catsent_hot.loc[i, row['aspect_category']] = 3
        elif row['category_polarity'] == 'positive':
            df_catsent_hot.loc[i, row['aspect_category']] = 4
    
    df_catsent_hot = df_catsent_hot.groupby(df_catsent_hot['text']).aggregate({'aspect_term':'first', 'term_polarity':'first', 'category_polarity':'first', 'ambience':'max', 'anecdotes/miscellaneous':'max', 'food':'max', 'price':'max', 'service':'max'})
    df_catsent_hot.reset_index(inplace=True)

    return split_dataset(df_catsent_hot)


def split_dataset(df_one_hot):
    # Set the seed for reproducibility
    seed = 42

    # Calculate the sizes of the datasets
    total_size = len(df_one_hot)
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    # Calculate the number of samples for each dataset
    train_samples = int(total_size * train_size)
    val_samples = int(total_size * val_size)
    test_samples = total_size - train_samples - val_samples

    # Create the training dataset
    train_dataset = df_one_hot.sample(n=train_samples, random_state=seed)

    # Create the validation dataset
    remaining_indices = df_one_hot.index.difference(train_dataset.index)
    val_dataset = df_one_hot.loc[remaining_indices].sample(n=val_samples, random_state=seed)

    # Create the test dataset
    test_indices = remaining_indices.difference(val_dataset.index)
    test_dataset = df_one_hot.loc[test_indices]

    # Reset the indices for the datasets
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    return train_dataset, val_dataset, test_dataset

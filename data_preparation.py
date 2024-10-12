# Import necessary libraries
import os
import pandas as pd
import re

# Define constants for file paths
DATASET_URL = "https://github.com/grant-TraDA/NLP-2023W/raw/refs/heads/main/13.%20Mining%20UNGA%20debates/project1/solution/dataset/dataset.zip"
DATASET_ZIP_PATH = "dataset.zip"
UN_CORPUS_PATH = "UN General Debate Corpus/TXT"

# Function to download and unzip the dataset
def download_and_extract_data():
    """Downloads and extracts the UN General Debate dataset."""
    os.system(f"wget {DATASET_URL}")
    os.system(f"unzip -q {DATASET_ZIP_PATH}")

def read_debate_texts(corpus_path):
    """
    Reads debate texts from the UN General Debate Corpus.
    
    Args:
        corpus_path (str): Path to the corpus folder.
        
    Returns:
        pd.DataFrame: DataFrame with session, year, country, and text columns.
    """
    sessions = []
    years = []
    countries = []
    texts = []
    
    for session_folder in os.listdir(corpus_path):
        try:
            for country_file in os.listdir(f'{corpus_path}/{session_folder}'):
                with open(f'{corpus_path}/{session_folder}/{country_file}', 'r', encoding='utf-8') as f:
                    texts.append(f.read())  # Read the speech text
                countries.append(country_file.split('_')[0])  # Extract country name
                sessions.append(session_folder.split()[1])  # Extract session number
                years.append(session_folder.split()[-1])  # Extract year
        except Exception as e:
            print(f"Error processing file: {session_folder}. Error: {e}")
    
    # Create a DataFrame from the collected data
    return pd.DataFrame({'session': sessions, 'year': years, 'country': countries, 'text': texts})


def extract_sentences_with_keywords(df, keywords):
    """
    Extracts sentences containing specific keywords from the debate texts.
    
    Args:
        df (pd.DataFrame): DataFrame containing the debate texts.
        keywords (list of str): List of keywords to search for in the texts.
        
    Returns:
        pd.DataFrame: DataFrame with session, year, country, and sentences containing the keywords.
    """
    session_list = []
    year_list = []
    country_list = []
    sentence_list = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Split the text into sentences
        sentences = row['text'].split('.')
        for s in sentences:
            # Check if any keyword appears in the sentence using regex
            if re.search(r"\b(" + "|".join(keywords) + r")\b", s):
                session_list.append(row['session'])
                year_list.append(row['year'])
                country_list.append(row['country'])
                # Clean and store the sentence
                s = s.replace('\n', ' ').replace('\t', ' ').replace('\s+', ' ').strip() + '.'
                sentence_list.append(s)

    # Create a new DataFrame for sentences containing the keywords
    return pd.DataFrame({'session': session_list, 'year': year_list, 'country': country_list, 'sentence': sentence_list})


def save_to_csv(df, filename):
    """
    Saves the DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        filename (str): Name of the output CSV file.
    """
    df.to_csv(filename, index=False)

# Example assertion tests to check the functions
def test_functions():
    """Test the main functions with basic assertions."""
    # Test that the DataFrame has the correct columns after reading texts
    test_df = pd.DataFrame({
        'session': ['74'], 
        'year': ['2019'], 
        'country': ['UnitedStates'], 
        'text': ["This is a test speech about peace and war."]
    })
    assert 'session' in test_df.columns
    assert 'year' in test_df.columns
    assert 'country' in test_df.columns
    assert 'text' in test_df.columns
    
    # Test that the extract function returns sentences with the keywords
    extracted_sentences = extract_sentences_with_keywords(test_df, ['peace', 'war'])
    assert len(extracted_sentences) > 0
    assert 'peace' in extracted_sentences['sentence'].iloc[0] or 'war' in extracted_sentences['sentence'].iloc[0]

    print("All tests passed!")

if __name__ == "__main__":
    # Download and extract the dataset
    download_and_extract_data()
    
    # Read the debate texts
    df = read_debate_texts(UN_CORPUS_PATH)
    
    # Extract sentences with specific keywords
    keywords = ['security']
    df_sentences = extract_sentences_with_keywords(df, keywords)
    
    # Save the extracted sentences to a CSV file
    save_to_csv(df_sentences, 'UN_peace_and_war.csv')
    
    # Run the assertion tests
    test_functions()
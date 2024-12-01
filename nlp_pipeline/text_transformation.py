import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from wordfreq import zipf_frequency
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import re
from PyPDF2 import PdfReader, PdfWriter
from unidecode import unidecode

# key between langdetect language ISO code and NLTK's names for snowball, stopwords, and entity detection
nltk_langdetect_dict = {
    'ar':'arabic',
    'az':'azerbaijani',
    'eu':'basque',
    'bn':'bengali',
    'ca':'catalan',
    'zh':'chinese',
    'da':'danish',
    'nl':'dutch',
    'en':'english',
    'fi':'finnish',
    'fr':'french',
    'de':'german',
    'el':'greek',
    'he':'hebrew',
    'hu':'hungarian',
    'id':'indonesian',
    'it':'italian',
    'kk':'kazakh',
    'ne':'nepali',
    'no':'norwegian',
    'pt':'portuguese',
    'ro':'romanian',
    'ru':'russian',
    'sl':'slovene',
    'es':'spanish',
    'sv':'swedish',
    'tg':'tajik',
    'tr':'turkish'
}
snowball_languages = [
    'arabic',
    'danish',
    'dutch',
    'english',
    'finnish',
    'french',
    'german',
    'hungarian',
    'italian',
    'norwegian',
    'portuguese',
    'romanian',
    'russian',
    'spanish',
    'swedish'
]

spacy_entity_lang_dict = {
    'ca': 'ca_core_news_lg',
    'zh': 'zh_core_web_lg',
    'hr': 'hr_core_news_lg',
    'nl': 'nl_core_news_lg',
    'en': 'en_core_web_lg',
    'fi': 'fi_core_news_lg',
    'fr': 'fr_core_news_lg',
    'de': 'de_core_news_lg',
    'el': 'el_core_news_lg',
    'it': 'it_core_news_lg',
    'ja': 'ja_core_news_lg',
    'ko': 'ko_core_news_lg',
    'lt': 'lt_core_news_lg',
    'mk': 'mk_core_news_lg',
    'no': 'nb_core_news_lg',
    'pl': 'pl_core_news_lg',
    'pt': 'pt_core_news_lg',
    'ro': 'ro_core_news_lg',
    'ru': 'ru_core_news_lg',
    'es': 'es_core_news_lg',
    'sv': 'sv_core_news_lg',
    'uk': 'uk_core_news_lg'
}

def gen_nltk_lang_dict(dictionary, lang):
    try:
        return dictionary[lang]
    except:
        return "english"
    
def gen_spacy_entity_lang_dict(dictionary, lang):
    try:
        return dictionary[lang]
    except:
        return "en_core_web_lg"

def lower(stringx):
    "lower case the text"
    return stringx.lower()
  

def replace_accents(stringr):
    "remove unusual characters, currency symbols, and replace accented characters"
    return unidecode(stringr)


def remove_headers_and_footers(stringx):
    # Split the content by [newpage]
    sections = stringx.split('[newpage]')
    
    # Check if there are enough sections, (first section always empty)
    if len(sections) < 3:
        return stringx  # No headers/footers to remove if there's less than 2 sections

    # Identify header (first line of the first section) and footer (first line of the last section)
    header = None
    for section in sections[1:]:
        if section.strip():
            header = re.sub(r'\d+', '', section.strip().split('\n')[0])
            break
        
    # Identify footer (first line of the last section if not empty)
    footer = re.sub(r'\d+', '', sections[-1].strip().split('\n')[-1]) if sections[-1].strip() else None

    # Process each section
    cleaned_sections = []
    for index, section in enumerate(sections):
        lines = section.strip().split('\n')  # Split section into lines
        
        if index == 0:
            continue
        # Keep the first header (in the first section)
        elif index == 1 and lines and re.sub(r'\d+', '', lines[0]) == header:
            # remove footer
            if lines and re.sub(r'\d+', '', lines[-1]) == footer:
                lines.pop() #removes last element
        
        # in middle sections remove all 
        elif index < len(sections) - 1:
            # For other sections, remove the header if it matches
            if lines and re.sub(r'\d+', '', lines[0]) == header:
                lines.pop(0)  # Remove the header
            # remove footer
            if lines and re.sub(r'\d+', '', lines[-1]) == footer:
                lines.pop() #removes last element
        
        #if final section, then do not remove the footer
        else:
            # For other sections, remove the header if it matches
            if lines and re.sub(r'\d+', '', lines[0]) == header:
                lines.pop(0)  # Remove the header
        
        
        # Join the cleaned lines back together, preserving line breaks
        cleaned_sections.append('\n'.join(lines))

    stringx = '[newpage]' + '\n[newpage]'.join(cleaned_sections).strip()
    # Join cleaned sections back together with '[newpage]'
    return stringx


def remove_urls(stringx):
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9./]+"
    stringx = re.sub(url_pattern, "", stringx)
    return stringx


ligature_map = {
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
    'ﬀ': 'ff',
    'ﬆ': 'st'
}

def replace_ligatures(stringx):
    "Function to replace ligatures"
    for ligature, replacement in ligature_map.items():
        stringx = stringx.replace(ligature, replacement)
    return stringx


def replace_newline_period(stringx):
    "replace new line characters, new page characters, and periods with |. Also remove multiple white spaces"
    
    # replace newlines with |s
    stringx = stringx.replace("\n", " | ")
    
    # replace [newpage]
    stringx = stringx.replace("[newpage]", "")

    # remove multiple whitespace
    stringx = " ".join(stringx.split())

    # this is a list of dots that we do not want to remove
    abbreviations = ['h.e.', 'h. e.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'p.m.', 'a.m.', 'dr.', 'mr.', 'mrs.', 'ms.', 
                    'jr.', 'sr.', 'st.', 'no.', 'co.', 'ltd.', 'inc.', 'prof.', 'gen.', 'col.', 'lt.', 'capt.', 
                    'maj.', 'rev.', 'u.s.', 'cf.', 'et al.', 'ibid.', 'op.', 'cit.', 'vol.', 'fig.', 'ch.', 'ed.', 
                    'esq.', 'sec.']

    for abbr in abbreviations:
        # Escape periods for regex, and replace with a placeholder
        replacement = abbr.replace(".", "#pl#")
        stringx = re.sub(re.escape(abbr), replacement, stringx)

    # replace all periods with |, including a space so the words can be found independently of the period
    stringx = stringx.replace(".", " | ")
    stringx = stringx.replace("?", " | ")
    stringx = stringx.replace("!", " | ")

    # append a vertical line to beginning and end so all sentences are enclosed by two |s
    stringx = "| " + stringx + " |"
    
    return stringx

def replace_period(stringx):
    "replace new line characters, new page characters with space ' ', and periods with |. Also remove multiple white spaces"
    # replace newlines with |s
    stringx = stringx.replace("\n", " ")
    # replace [newpage]
    stringx = stringx.replace("[newpage]", "")
    # remove multiple whitespace
    stringx = " ".join(stringx.split())
    
    # this is a list of periods that we want to replace with nothing
    abbreviations = ['h.e.', 'h. e.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'p.m.', 'a.m.', 'dr.', 'mr.', 'mrs.', 'ms.', 
                    'jr.', 'sr.', 'st.', 'no.', 'co.', 'ltd.', 'inc.', 'prof.', 'gen.', 'col.', 'lt.', 'capt.', 
                    'maj.', 'rev.', 'u.s.', 'cf.', 'et al.', 'ibid.', 'op.', 'cit.', 'vol.', 'fig.', 'ch.', 'ed.', 
                    'esq.', 'sec.']

    for abbr in abbreviations:
        # Escape periods for regex, and replace with a placeholder
        replacement = abbr.replace(".", "")
        stringx = re.sub(re.escape(abbr), replacement, stringx)

    # replace all periods with |, including a space so the words can be found independently of the period
    stringx = stringx.replace(".", " | ")
    stringx = stringx.replace("?", " | ")
    stringx = stringx.replace("!", " | ")
    
    # append a vertical line to beginning and end so all sentences are enclosed by two |s
    stringx = "| " + stringx + " |"
    
    return stringx

def replace_with_dict(stringx, replace_map):
    "replace dict names with dict values in stringx"
    for find, replace in replace_map.items():
        stringx = stringx.replace(find, replace)
    return stringx


    
def remove_punctuation(stringx):
    "remove punctuation, except |s, replace quotation marks apostophes brackets, commas, colons, and semicolons with nothing. dashes and slashes with spaces"
    stringx = stringx.replace(" & ", " and ").replace("&", " and")
    #delete these punctuation
    stringx = stringx.translate(str.maketrans('', '', '"#\'(),/:;?@[]^`{}~”“’;•'))
    replace_with_space = '!.?$%*+-<=>\\_~'
    stringx = stringx.translate(str.maketrans(replace_with_space, ' '*len(replace_with_space)))
    
    return stringx

    
def drop_numbers(stringx):
    "Remove all numbers"
    stringx = re.sub(r'\d+', '', stringx)
    return stringx

def exclude_words(stringx, exclude_words_df):
    "excludes words or phrases using a pandas dataframe with the first column being a list of words to exclude and replace with empty '' "
    exclude_words_df.rename(columns={exclude_words_df.columns[0]: 'word to exclude'}, inplace=True)
    exclude_words = set(exclude_words_df['word to exclude'].dropna().str.lower())  # drop NaN values and convert to a set for faster lookups
    for word in exclude_words:
        stringx = re.sub(rf'(\s){word.lower()}(\s)', r'\1\2', stringx)  # Match the word with spaces before and after
    return stringx
        

def replace_words_text_transform(stringx, replace_words_df):
    "replaces words or phrases using a pandas dataframe with the first column being a list of words to replace and the second column being a list of words to replace with."
    replace_words_df.rename(columns={replace_words_df.columns[0]: 'term', replace_words_df.columns[1]: 'replacement'}, inplace=True)
    replace_dict = dict(zip(replace_words_df['term'].dropna().str.lower(), replace_words_df['replacement'].dropna().str.lower()))
    for term, replacement in replace_dict.items():
        stringx = re.sub(rf'(\s){term.lower()}(\s)', r'\1' + replacement.lower() + r'\2', stringx)  # Match and replace whole words with spaces
    return stringx

def remove_stopwords(stringx, language):
    "remove stopwords"
    eng_stopwords = stopwords.words(gen_nltk_lang_dict(nltk_langdetect_dict, language))

    # remove stopwords, tweet tokenizer because doens't split apostrophes
    tk = TweetTokenizer()
    tokenized_string = tk.tokenize(stringx)
    stringx = [item for item in tokenized_string if item not in eng_stopwords]

    stringx = " ".join(stringx)

    return stringx
    
def stem(stringx, stemmer=None, language=None):
    "stem a string. snowball is less agressive, lancaster only works with english"
    if stemmer == "snowball":
        if not(language in snowball_languages):
            language = "english"
        stemmer = SnowballStemmer(gen_nltk_lang_dict(nltk_langdetect_dict, language))
    elif stemmer == "lancaster":
        stemmer = LancasterStemmer()
    if stemmer != None:
        tk = TweetTokenizer()
        tokenized_string = tk.tokenize(stringx)
        stringx = [stemmer.stem(item) for item in tokenized_string]
        stringx = " ".join(stringx)
    return stringx

def gen_word_count_dict(stringx, exclude_words):
    "create a dictionary of word counts in a string. exclude_words is a list of words to filter out"
    counts = dict()
    words = stringx.split(" ")

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    counts = {word:count for word, count in counts.items() if (len(word) > 1) and not(word in exclude_words) and not(word.isnumeric())}

    return counts

def get_single_sentiment(stringx, sentiment_analyzer = SentimentIntensityAnalyzer):
    "get sentiment of a single string"
    sia = sentiment_analyzer()
    return sia.polarity_scores(stringx)  

def get_word_frequency(word, language):
    "https://pypi.org/project/wordfreq/"
    try:
        freq = zipf_frequency(word, language)
    except:
        freq = zipf_frequency(word, "en")
    return freq

def gen_sentiment_report(stringx, sentiment_analyzer = SentimentIntensityAnalyzer):
    "generate sentiment report of a string"
    stringx = lower(stringx)
    stringx = replace_newline_period(stringx)
    stringx = remove_punctuation(stringx)
    
    string_list = stringx.split("|")
    string_list = [x for x in string_list if (len(x) > 3) & (len(x.split(" ")) > 2) & (not(x.isnumeric()))]
    sentiment_list = [get_single_sentiment(x, sentiment_analyzer)["compound"] for x in string_list]
 
    
    sentiment_report = pd.DataFrame({
        "sentence_number": list(range(1,len(string_list)+1)),
        "sentence": string_list,
        "sentiment": sentiment_list
    })
    
    return sentiment_report

def gen_entity_count_dict(stringx, lang):
    "create a dictionary of entity counts in a string"
    stringx = stringx.replace("\n", " ") # get rid of newline character if present
    stringx = stringx.replace("[newpage]", " ") # get rid of newpage character if present
    
    spacy_model = gen_spacy_entity_lang_dict(spacy_entity_lang_dict, lang)
    ner = spacy.load(spacy_model)
    ner.max_length = len(stringx)
    text = ner(stringx)
    
    tmp = pd.DataFrame({
        "entity": [ent.text for ent in text.ents],
        "label": [ent.label_ for ent in text.ents]
    })
    tmp = tmp.groupby(["entity", "label"])["entity"].count().sort_values(ascending = False)
    
    counts = dict(zip([x[0] + "|" + x[1] for x in tmp.index], tmp.values))
    
    return counts

def doc_split(processor, text_ids, split_by_page = True, split_by_n_words=500):
    "split a document into page-sized or n_words length documents"
    doc_list = pd.DataFrame(columns = ["text_id", "doc"])
    for text_id in text_ids:
        file_path = processor.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]
        
        # read file
        file = open(f"{file_path}", "r", encoding = "UTF-8") 
        stringx = file.read()
        file.close()
        
        # split into documents
        if split_by_page:
            tmp = pd.DataFrame({
                "text_id": text_id,
                "doc": stringx.split("[newpage]")
            }).loc[lambda x: x.doc.str.len() > 3, :].reset_index(drop=True)
        else: 
            split_string = stringx.split()
            doc_string = [" ".join(split_string[i:(i + split_by_n_words)]) for i in range(0, len(split_string), split_by_n_words)]
            
            tmp = pd.DataFrame({
                "text_id": text_id,
                "doc": doc_string
            })
            
        # combine to main list
        doc_list = pd.concat([doc_list, tmp], ignore_index = True)
        
    return doc_list

def train_bertopic_model(processor, text_ids, model_name, notes="", split_by_n_words = None):
    "train a bertopic model based off a set of text_ids"
    # getting stopwords of majority language of documents
    majority_language = list(processor.metadata.loc[lambda x: x.text_id.isin(text_ids), "detected_language"].values)
    majority_language = max(set(majority_language), key=majority_language.count)
    lang_stopwords = stopwords.words(gen_nltk_lang_dict(nltk_langdetect_dict, majority_language))
    vectorizer_model = CountVectorizer(stop_words=lang_stopwords)
    
    # creating doc_list
    if split_by_n_words == None:
        doc_list = doc_split(processor, text_ids)
    else:
        doc_list = doc_split(processor, text_ids, split_by_page = False, split_by_n_words = split_by_n_words)
        
    # dictionary of text_ids to new shorter documents
    doc_count = {}
    for i in doc_list.text_id:
        doc_count[i] = doc_count.get(i, 0) + 1
    # convert back into a list
    #[item for sublist in [[key] * value for key, value in doc_count.items()] for item in sublist]

    # training the model
    print("training BERTopic model...")
    if majority_language == "en":
        model = BERTopic(vectorizer_model=vectorizer_model)
    else:
        model = BERTopic(language="multilingual", vectorizer_model=vectorizer_model)
    topics, probs = model.fit_transform(doc_list.doc.values)
    
    # create BERTopic directory if it doesn't exist
    if not os.path.exists(f"{processor.data_path}bertopic_models/"):
        os.makedirs(f"{processor.data_path}bertopic_models/")
    if not os.path.exists(f"{processor.data_path}bertopic_models/{model_name}/"):
        os.makedirs(f"{processor.data_path}bertopic_models/{model_name}/")
        
    # model metadata file
    if not os.path.exists(f"{processor.data_path}bertopic_models/model_metadata.csv"):
        metadata = pd.DataFrame(columns = ["model_name", "notes", "text_ids", "document_ids", "split_by_n_words"])
    else:
        metadata = pd.read_csv(f"{processor.data_path}bertopic_models/model_metadata.csv")

    # remove prior metadata if this model name already exists
    metadata = metadata.loc[lambda x: x.model_name != model_name, :].reset_index(drop=True)
    
    tmp_metadata = pd.DataFrame({
        "model_name": model_name,
        "notes": notes,
        "text_ids": str(text_ids),
        "document_ids": str(doc_count),
        "split_by_n_words": str(split_by_n_words)
    }, index = [0])
    metadata = pd.concat([metadata, tmp_metadata], ignore_index = True)
    
    print(f"saving BERTopic model to {processor.data_path}bertopic_models/{model_name}/model...")
    model.save(f"{processor.data_path}bertopic_models/{model_name}/model")
    print("model trained and saved")
    
    metadata.to_csv(f"{processor.data_path}bertopic_models/model_metadata.csv", index = False)
    
def load_bertopic_model(processor, model_name):
    "load a previously trained bertopic model"
    model = BERTopic.load(f"{processor.data_path}bertopic_models/{model_name}/model")
    return model

def bertopic_visualize(processor, model, model_name, method_name, plot_name, timestamps = None, presence_text_ids = None, presence_topic_ids = None, *args, **kwargs):
    "save visualizations from a bertopic model to html"
    
    plot_df = None
    
    #document info of the model
    metadata = pd.read_csv(f"{processor.data_path}bertopic_models/model_metadata.csv")
    text_ids = eval(metadata.loc[lambda x: x.model_name == model_name, "text_ids"].values[0])
    split_by_n_words = metadata.loc[lambda x: x.model_name == model_name, "split_by_n_words"].values[0]
    if split_by_n_words == None or pd.isna(split_by_n_words):
        split_by_page = True
    else:
        split_by_page = False
    docs = doc_split(processor, text_ids, split_by_page, split_by_n_words).doc.values
    doc_count = eval(metadata.loc[lambda x: x.model_name == model_name, "document_ids"].values[0])
    doc_ids = [item for sublist in [[key] * value for key, value in doc_count.items()] for item in sublist]
    
    # cluster plot needs the list of documents
    if method_name == "visualize_documents":
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("getting embeddings...")
        embeddings = sentence_model.encode(docs, show_progress_bar=False)
        
        fig = model.visualize_documents(docs, embeddings=embeddings)
    elif method_name == "visualize_topics_over_time":
        timestamp_dict = dict(zip(text_ids, timestamps))
        full_timestamps = [timestamp_dict[x] for x in doc_ids]
        topics_over_time = model.topics_over_time(docs, full_timestamps)

        func = getattr(model, method_name)
        fig = func(topics_over_time, **kwargs)
    elif method_name == "visualize_topics_presence":
        # getting relative attribution to each topic
        doc_info = model.get_document_info(docs)
        doc_info["text_id"] = [item for sublist in [[key] * value for key, value in doc_count.items()] for item in sublist]
        df = doc_info.loc[:, ["Topic", "text_id", "Probability"]].groupby(by = ["Topic", "text_id"]).sum().reset_index()
        df["n_docs"] = df.text_id
        df = df.replace({"n_docs": doc_count})
        df["topic_share"] = df.Probability / df.n_docs
        df = df.loc[:, ["Topic", "text_id", "topic_share"]]
        df = df.rename(columns={"Topic": "topic"})
        df["topic_desc"] = df.topic
        df = df.replace({"topic_desc": dict(zip(doc_info.Topic, doc_info.Name))})
        
        # plotting
        plot_df = df.drop(["topic"], axis = 1).pivot("topic_desc", "text_id").transpose().reset_index()
        plot_df = plot_df.fillna(0)
        plot_df = plot_df.iloc[:, 1:].set_index("text_id").transpose()
        
        if presence_text_ids != None:
            plot_df = plot_df.loc[:, presence_text_ids]
        if presence_topic_ids != None:
            plot_df = plot_df.loc[[True if int(x.split("_")[0]) in presence_topic_ids else False for x in list(plot_df.index)], :]
        
        plt.figure(figsize = (plot_df.shape[1] * 2, plot_df.shape[0] * 2))
        sns.heatmap(plot_df, annot = True, linewidth = 0.5, cmap = sns.color_palette("Blues", as_cmap=True))
        plt.ylabel("Topic")
        plt.xlabel("Text ID")
        plt.tight_layout()
        plt.savefig(f"{processor.data_path}bertopic_models/{model_name}/{plot_name}.png", dpi = 99)
    else:
        func = getattr(model, method_name)
        fig = func(*args, **kwargs)
    
    try:
        fig.write_html(f"{processor.data_path}bertopic_models/{model_name}/{plot_name}.html")
    except:
        pass
    
    return plot_df

def convert_utf8(processor, text_ids, which_file = "local_raw_filepath"):
    "convert text(s) to UTF-8 by ID #"
    if (isinstance(text_ids, int)):
        text_ids = [text_ids]
    for text_id in text_ids:
        file_path = processor.metadata.loc[lambda x: x.text_id == text_id, which_file].values[0]
        file = open(file_path, "r", encoding = "latin1")
        stringx = file.read()
        file.close()
        
        file = open(file_path, "w")
        file.write(str(stringx.encode("UTF-8")))
        file.close()
        
def replace_words(processor, text_ids, replacement_list, path_prefix = ""):
    "replace words/phrases in text files"
    if (isinstance(text_ids, int)):
        text_ids = [text_ids]
    
    # if no prefix, replace the terms in the "txt_files" directory
    if path_prefix == "":
        file_path_prefix = f"{processor.data_path}/txt_files/"
    else:
        file_path_prefix = f"{processor.data_path}/transformed_txt_files/{path_prefix}_"
        
    for text_id in text_ids:
        file_path = f"{file_path_prefix}{text_id}.txt"
        file = open(file_path, "r", encoding = "latin1")
        stringx = file.read()
        file.close()
        
        stringx = re.sub(re.escape("("), " ", stringx)
        stringx = re.sub(re.escape(")"), " ", stringx)
        
        for punc in [x for x in ".,:;?!-"]:#"#[x for x in string.punctuation if x not in ["(", ")", "[", "]", "\\"]] + ["\n"] + ["\t"]:
            term = f"\{punc}"
            replacement = " " + punc
            stringx = re.sub(term, replacement, stringx)
        
        # do replacement
        stringx = re.sub("  ", " ", stringx)
        stringx = re.sub("   ", " ", stringx)
        stringx = re.sub("    ", " ", stringx)
        stringx = re.sub("     ", " ", stringx)
        stringx = re.sub("\n", " ", stringx)
        for i in range(len(replacement_list)):
            term = " " + replacement_list.iloc[i, 0] + " "
            replacement = " " + replacement_list.iloc[i, 1] + " "
            stringx = re.sub(term, replacement, stringx)
            
        file = open(file_path, "w", encoding = "latin1")
        file.write(stringx)
        file.close()
        
        
def filter_pdf_pages(processor, page_num_column):
    "edit PDF files to select only certain pages, in a metadata column in format e.g. '2:10,12:20,23'"
    # select out pages
    for text_id in list(processor.metadata.text_id.values):
        page_nums = processor.metadata.loc[lambda x: x.text_id == text_id, page_num_column].values[0]
        
        if (page_nums != "") & (not(pd.isnull(page_nums))):
            final_pages = []
            
            for pages in page_nums.split(","):
                if ":" in pages:
                    start_i = int(pages.split(":")[0]) - 1
                    end_i = int(pages.split(":")[1])
                    final_pages += list(range(start_i, end_i))
                else:
                    final_pages.append(int(pages)-1)
            # cut down PDF and replace
            pdf_file_path = processor.metadata.loc[lambda x: x.text_id == text_id, "local_raw_filepath"].values[0]
            file_base_name = pdf_file_path.replace(".pdf", "")
        
            pdf = PdfReader(pdf_file_path)
            pages = final_pages
            pdfWriter = PdfWriter()
        
            for page_num in pages:
                pdfWriter.add_page(pdf.pages[page_num])
        
            with open(pdf_file_path.format(file_base_name), 'wb') as f:
                pdfWriter.write(f)
                f.close()
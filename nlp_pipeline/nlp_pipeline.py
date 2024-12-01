from importlib import import_module
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

class nlp_processor:
    """Primary class of the library
    parameters:
        :data_path: str: filepath where all files will be created and stored. Will be created if doesn't already exist. Pass absolute directory.
        :metadata_addt_column_names: list: list of additional file names relevant for the analysis, e.g., publication date, title
        :windows_tesseract_path: str: tesseract .exe file path in windows for OCR conversion
        :windows_poppler_path: str: poppler /bin file path in windows for OCR conversion
    """
    
    def __init__(
        self,
        data_path,
        metadata_addt_column_names = [],
        windows_tesseract_path = None, 
        windows_poppler_path = None
    ):
        # initializing parameters
        if data_path[-1:] != "/":
            data_path += "/"
        self.data_path = data_path
        self.metadata_addt_column_names = metadata_addt_column_names
        self.windows_tesseract_path = windows_tesseract_path
        self.windows_poppler_path = windows_poppler_path
        
        # setting up directory structure
        self.files_setup = import_module("nlp_pipeline.files_setup")
        self.files_setup.setup_directories(self.data_path)
        
        # generating metadata file
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names)
        self.metadata = pd.read_csv(f"{data_path}metadata.csv")
        
        # making text transformations available
        self.text_transformation = import_module("nlp_pipeline.text_transformation")
        
        # making search terms available
        self.search_terms = import_module("nlp_pipeline.search_terms")
        
        # making visualizations available
        self.visualizations = import_module("nlp_pipeline.visualizations")
        
    def refresh_object_metadata(self):
        "update the metadata of the processor in case changes are made to the file outside of the object"
        self.metadata = pd.read_csv(f"{self.data_path}metadata.csv")
        self.metadata = self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names) # make sure text_id added
    
    def sync_local_metadata(self):
        "update the metadata local file to reflect actual state of files in the data directory"
        self.metadata = self.files_setup.refresh_local_metadata(self.metadata, self.data_path)
        self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)
        
    def download_text_id(self, text_ids):
        "download a file from a URL and update the metadata file. Pass either single text id or list of them. Can also point to a local .txt file"
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"downloading file {counter}/{len(text_ids)}")
            counter += 1
            
            web_filepath = self.metadata.loc[lambda x: x.text_id == text_id, "web_filepath"].values[0]
            tmp_metadata = self.files_setup.download_document(self.metadata, self.data_path, text_id, web_filepath)
            
            # update the object's metadata
            if not(tmp_metadata is None):
                self.metadata = tmp_metadata
                self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)
    
    def clear_directories(self, raw_files = True, txt_files = True, transformed_txt_files = True, csv_outputs = True, visual_outputs = True):
        "clear the data in different directories"
        self.files_setup.clear_directories(self, raw_files, txt_files, transformed_txt_files, csv_outputs, visual_outputs)
        
    def filter_pdf_pages(self, page_num_column):
        "select certain pages of a pdf. Will edit the PDF file in place. Have a column denoting the page numbers to keep in the metadata file in format like '1:10,24:25,30,44,50:54'. Leave the entry blank to keep all pages."
        self.text_transformation.filter_pdf_pages(self, page_num_column)        
        
    def convert_utf8(self, text_ids, which_file = "local_raw_filepath"):
        "convert text(s) to UTF-8 by ID #"
        self.text_transformation.convert_utf8(self, text_ids, which_file)

    def convert_to_text(self, text_ids, force_ocr = False):
        "convert a pdf or html file to text. Pass either single text id or list of them. 'force_ocr' is whether or not to force ocr conversion of a pdf"
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"converting to text: file {counter}/{len(text_ids)}")
            counter += 1
            
            tmp_metadata = self.files_setup.convert_to_text(self.metadata, self.data_path, text_id, self.windows_tesseract_path, self.windows_poppler_path, force_ocr)
            
            # update the object's metadata
            if not(tmp_metadata is None):
                self.metadata = tmp_metadata
                self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)

    def transform_text(
            self, 
            text_ids, 
            path_prefix,
            perform_lower = False,
            replace_accented_and_unusual_characters = False,
            perform_remove_multiple_header_and_footers = False,
            perform_remove_urls = False,
            perform_replace_period=False,
            perform_replace_newline_period = False,
            replace_words_with_punctuation_df = None,
            perform_remove_punctuation = False,
            drop_numbers = False,
            replace_words_df = None,
            exclude_words_df = None,
            perform_remove_stopwords = False,
            perform_stemming = False,
            stemmer = "snowball"
    ):
        """Transforms texts in various ways and writes new text files to the transformed_txt_files/ directory
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what to prefix the resulting .txt files with to label which transformations have been done. A "_" will be appended to the end of it.
            :perform_lower: boolean: whether or not to lower case the text
            :replace_accented_and_unusual_characters: boolean: whether to remove accented characters, currency symbols, and unusual characters. 
            :perform_remove_multiple_header_and_footers: boolean: whether to remove all headers after the first one, and same for footers. First header and first footer will be kept. 
            :perfrom_remove_urls: boolean: whether to remove urls and replace with nothing
            :perform_replace_period: boolean: whether or not to replace periods with | so words and sentences can be identified in isolation            
            :perform_replace_newline_period: boolean: whether or not to replace new lines and periods with | so words and sentences can be identified in isolation
            :replace_words_with_punctuation_df: boolean: the same as replace_words_df but it is run before punctuation is removed in the case that punctuation is removed 
            :perform_remove_punctuation: boolean: whether or not to remove punctuation, except for |'s    "remove punctuation, except |s, replace quotation marks apostophes brackets, commas, colons, and semicolons with nothing. dashes and slashes with spaces"
            :exclude_words_df: pandas dataframe with the first column being a list of words to exclude and replace with ""
            :replace_words_df: pandas dataframe with the first column being a list of words to replace and the second column being a list of words to replace with.
            :perform_remove_stopwords: boolean: whether or not to remove stopwords
            :perform_stemming: boolean: whether or not to perform stemming
            :stemmer: if choosing to stem, nltk stemmer. E.g., nltk.stem.snowball.SnowballStemmer("english"), or string of "snowball" or "lancaster" for one of these. Lancaster only works in english
        """
        path_prefix += "_"
        
        print(len(text_ids))
        
        if type(text_ids) != list:
            text_ids = [text_ids]
            
        print(len(text_ids))
            
        counter = 1
        for text_id in text_ids:
            
            
            text_path = self.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]
            language = self.metadata.loc[lambda x: x.text_id == text_id, "detected_language"].values[0]
            
            print("transforming " + str(text_path) + f" {counter}/{len(text_ids)}")
            counter += 1
            
            
            if (str(language) == "") | (str(language) == "nan"): # if no language, default to english
                language = "en"
            
            # only perform if text file exists and transformed file doesn't already exist
            if (".txt" in str(text_path)) & (not(os.path.exists(f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"))):
                # reading original text
                file = open(f"{text_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                
                if perform_lower:
                    stringx = self.text_transformation.lower(stringx)
                stringx = self.text_transformation.replace_ligatures(stringx)
                if replace_accented_and_unusual_characters:
                    stringx = self.text_transformation.replace_accents(stringx)
                if perform_remove_urls:
                    stringx = self.text_transformation.remove_urls(stringx)
                if perform_remove_multiple_header_and_footers:
                    stringx = self.text_transformation.remove_headers_and_footers(stringx)
                if perform_replace_period:
                    stringx = self.text_transformation.replace_period(stringx)
                if perform_replace_newline_period:
                    stringx = self.text_transformation.replace_newline_period(stringx)
                if replace_words_with_punctuation_df is not None:
                    stringx = self.text_transformation.replace_words_text_transform(stringx, replace_words_with_punctuation_df)
                if perform_remove_punctuation:
                    stringx = self.text_transformation.remove_punctuation(stringx)
                if drop_numbers:
                    stringx = self.text_transformation.drop_numbers(stringx)
                if replace_words_df is not None:
                    stringx = self.text_transformation.replace_words_text_transform(stringx, replace_words_df)
                if exclude_words_df is not None:
                    stringx = self.text_transformation.exclude_words(stringx, exclude_words_df)
                if perform_remove_stopwords:
                    stringx = self.text_transformation.remove_stopwords(stringx, language)
                if perform_stemming:
                    stringx = self.text_transformation.stem(stringx, stemmer, language)
                
                # write text file
                file = open(f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt", "wb+")
                file.write(stringx.encode())
                file.close()
                    
    def gen_word_count_csv(self, text_ids, path_prefix, exclude_words):
        """Gets word counts from files in transformed_txt_files/ directory and writes them to CSV
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "word_count_dict": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"creating word count dictionary: {counter}/{len(text_ids)}")
            counter += 1
            
            text_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            
            # only perform if text file exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "word_count_dict"].values[0]
            if os.path.exists(text_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading original text
                file = open(f"{text_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                word_dict = self.text_transformation.gen_word_count_dict(stringx, exclude_words)
                
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "word_count_dict"] = str(word_dict)
                csv.to_csv(csv_path, index = False)
            
    def bar_plot_word_count(self, text_ids, path_prefix, n_words=10, title=""):
        """bar plot of top words occurring in text(s)
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is. Pass "entity" to do a chart based on entity counts instead of word counts.
            :n_words: int: top n words to show in the plot
            :title: str: title of the plot if desired
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # only do if dataframe exists
        if path_prefix != "entity_":
            csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        else:
            csv_path = f"{self.data_path}csv_outputs/entity_counts.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path).loc[lambda x: x.text_id.isin(text_ids), :].reset_index(drop=True)
            
            if path_prefix == "entity_":
                csv = csv.rename(columns={"entity_count_dict": "word_count_dict"})
            
            df = self.visualizations.convert_word_count_dict_to_df(csv)
            if path_prefix == "entity_":
                df["word"] = [x[0] for x in df.word.str.split("|")]
            
            p, plot_data = self.visualizations.bar_plot_word_count(df, n_words, title)
            
            return (p, plot_data)
        
    def word_cloud(self, text_ids, path_prefix, n_words=10):
        """word cloud of top words occurring in text(s)
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is. Pass "entity" to do a chart based on entity counts instead of word counts.
            :n_words: int: top n words to show in the plot
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # only do if dataframe exists
        if path_prefix != "entity_":
            csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        else:
            csv_path = f"{self.data_path}csv_outputs/entity_counts.csv"
            
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path).loc[lambda x: x.text_id.isin(text_ids), :].reset_index(drop=True)
            
            if path_prefix == "entity_":
                csv = csv.rename(columns={"entity_count_dict": "word_count_dict"})
            
            df = self.visualizations.convert_word_count_dict_to_df(csv)
            
            if path_prefix == "entity_":
                df["word"] = [x[0] for x in df.word.str.split("|")]
                
            p, plot_data = self.visualizations.word_cloud(df, n_words)
            
            return (p, plot_data)
        
    def gen_sentiment_report(self, text_id=None, stringx=None, sentiment_analyzer=SentimentIntensityAnalyzer):
        """generate sentiment of phrases in a particular string or document
        parameters:
            :text_id: float: single text_id to generate report for
            :stringx: str: string to generate report for. Either this or text_id should be blank
            :sentiment_analyzer: nltk SentimentIntensityAnalyzer: which analyzer to use
        output:
            :pd.DataFrame: with columns:
                :sentence_number: sentence id in the string
                :sentence: the text of the sentence
                :sentiment: sentiment of the sentence
        """
        if stringx is None:
            txt_path = f"{self.data_path}txt_files/{text_id}.txt"
            file = open(f"{txt_path}", "r", encoding = "UTF-8") 
            stringx = file.read()
            file.close()
            
        return self.text_transformation.gen_sentiment_report(stringx = stringx, sentiment_analyzer = sentiment_analyzer)
        
    def gen_sentiment_csv(self, text_ids, path_prefix, sentiment_analyzer = SentimentIntensityAnalyzer, overwrite = False):
        """generating average sentiment of documents. A higher score is more positive, lower is more negative. Depends on sentences being delimited by |
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
            :sentiment_analyzer: nltk SentimentIntensityAnalyzer: which analyzer to use
            :overwrite: Boolean: whether or not to overwrite the sentiment score if one exists already
        output:
            :pd.DataFrame: with columns:
                :text_id: text ids
                :avg_sentiment_w_neutral: average sentiment score with 0.0 neutral sentences
                :avg_sentiment_wo_neutral: average sentiment score excluding 0.0 neutral sentences
                :neutral_proportion: % of sentences in the document with a 0.0 sentiment score
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
    
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}sentiments.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "avg_sentiment_w_neutral": "",
                "avg_sentiment_wo_neutral": "",
                "neutral_proportion": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"getting sentiments: {counter}/{len(text_ids)}")
            counter += 1
            
            txt_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            
            # only do if txt path exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "avg_sentiment_w_neutral"].values[0]
            if os.path.exists(txt_path) & (((str(prior_value) == "") | (str(prior_value) == "nan")) | overwrite):
                # reading original text
                file = open(f"{txt_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                # calculating sentiments
                sentiments = [self.text_transformation.get_single_sentiment(x, sentiment_analyzer)["compound"] for x in stringx.split("|") if (len(x) > 3) & (len(x.split(" ")) > 2) & (not(x.isnumeric()))] # only do sentiment for sentences with more than 2 words and not numeric
                
                # adding and writing to CSV
                try:
                    csv.loc[csv.text_id == text_id, "avg_sentiment_w_neutral"] = sum(sentiments) / len(sentiments)
                    csv.loc[csv.text_id == text_id, "avg_sentiment_wo_neutral"] = sum([x for x in sentiments if x != 0.0]) / len([x for x in sentiments if x != 0.0])
                    csv.loc[csv.text_id == text_id, "neutral_proportion"] = len([x for x in sentiments if x == 0.0]) / len(sentiments)
                except:
                    csv.loc[csv.text_id == text_id, "avg_sentiment_w_neutral"] = 0
                    csv.loc[csv.text_id == text_id, "avg_sentiment_wo_neutral"] = 0
                    csv.loc[csv.text_id == text_id, "neutral_proportion"] = 0
                csv.to_csv(csv_path, index = False)
                             
    def plot_word_occurrences(self, text_ids_list, word, path_prefix, x_labels = None, title = ""):
        """get plot of occurrences of a particular word over groups of documents. Searches for contains rather than exact matches.
        parameters:
            :text_ids_list: list[int]: either list of list of text_ids, e.g., [[1,2], [3,4]], or if individual documents rather than groups, [1,2,3,4]
            :word: str: which word to look for
            :path_prefix: str: what the prefix of the files in the csv_outputs/ path is. Need to have run the gen_word_count_csv() function. Pass "entity" to do a chart based on entity counts instead of word counts.
            :x_labels: list: what to label the x-axis in the plot, what are the different documents or groups of documents. E.g., decades or years.
            :title: str: additional title to the plot
        """
        path_prefix += "_"
        
        if path_prefix != "entity_":
            csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        else:
            csv_path = f"{self.data_path}csv_outputs/entity_counts.csv"
        
        # only run if file exists
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
            
            if path_prefix == "entity_":
                csv = csv.rename(columns={"entity_count_dict": "word_count_dict"})

            p, plot_df = self.visualizations.plot_word_occurrences(csv, text_ids_list, word, x_labels, title)
            
            return (p, plot_df)
           
    def plot_sentiment(self, text_ids_list, path_prefix, x_labels = None, title = "", sentiment_col = "avg_sentiment_wo_neutral"):
        """get plot of sentiment of a particular word over groups of documents. Searches for contains rather than exact matches.
        parameters:
            :text_ids_list: list[int]: either list of list of text_ids, e.g., [[1,2], [3,4]], or if individual documents rather than groups, [1,2,3,4]
            :path_prefix: str: what the prefix of the files in the csv_outputs/ path is. Need to have run the get_sentiment_csv() function.
            :x_labels: list: what to label the x-axis in the plot, what are the different documents or groups of documents. E.g., decades or years.
            :title: str: additional title to the plot
            :sentiment_col: str: which column in the sentiment df to plot, one of ["avg_sentiment_wo_neutral", "avg_sentiment_w_neutral", "neutral_proportion"]
        """
        path_prefix += "_"
        
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}sentiments.csv"
        
        # only run if file exists
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
            p, plot_df = self.visualizations.plot_sentiment(csv, text_ids_list, x_labels, title, sentiment_col)
            return (p, plot_df)
        
    
    def gen_summary_stats_csv(self, text_ids, path_prefix):
        """generating various summary statistics: number of words, unique words, sentences, pages, etc. per document. Depends on sentences being delimited by |
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
        output:
            :pd.DataFrame: with columns:
                :text_id: text ids
                :n_words: number of total words in the transformed document
                :n_unique_words: number of unique words in the transformed document
                :n_sentences: number of sentences in the transformed document
                :n_pages: number of pages in the original document
                :avg_word_length: average word length (number of letters) in the original document
                :avg_word_incidence: average incidence of words in the language of the document (higher is more frequent/common words, a word with Zipf value 6 appears once per thousand words and a word with Zipf value 3 appears once per million words)
                :num_chars_numeric: number of numeric characters in the document
                :num_chars_alpha: number of alpha characters in the document
                :numeric_proportion: numeric characters / (numeric + alpha characters)
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
    
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}summary_stats.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "n_words": "",
                "n_unique_words": "",
                "n_sentences": "",
                "n_pages": "",
                "avg_word_length": "",
                "avg_word_incidence": "",
                "num_chars_numeric": "",
                "num_chars_alpha": "",
                "numeric_proportion": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"getting word and sentence count: {counter}/{len(text_ids)}")
            counter += 1
            
            txt_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            #orig_txt_path = f"{self.data_path}txt_files/{text_id}.txt" # for n_pages
            orig_txt_path = self.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]

            language = self.metadata.loc[lambda x: x.text_id == text_id, "detected_language"].values[0]
            
            # only do if txt path exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "n_words"].values[0]
            if os.path.exists(txt_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading transformed text
                file = open(f"{txt_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                # reading original text for n_pages
                file = open(f"{orig_txt_path}", "r", encoding = "UTF-8") 
                orig_stringx = file.read()
                file.close()
                
                # calculating n_words
                n_words = len([x for x in stringx.split(" ") if len(x) > 1]) # minimum word length
                
                # calculating n_unique_words
                n_unique_words = len(set([x for x in stringx.split(" ") if len(x) > 1]))
                
                # calculating n_sentences
                n_sentences = len([x for x in stringx.split("|") if len(x) > 2]) # minimum sentence length
                
                # calculating n_pages
                n_pages = orig_stringx.count("[newpage]")
                
                # calculating avg_word_length
                avg_word_length = [len(x) for x in orig_stringx.split(" ") if len(x) > 1]
                try:
                    avg_word_length = sum(avg_word_length) / len(avg_word_length)
                except:
                    avg_word_length = 0
                
                # calculating avg incidence
                word_list = [x for x in orig_stringx.split(" ") if len(x) > 1]
                avg_word_incidence = [self.text_transformation.get_word_frequency(x, language) for x in word_list]
                try:
                    avg_word_incidence = sum(avg_word_incidence) / len(avg_word_incidence)
                except:
                    avg_word_incidence = 0
                    
                # calculating numeric proportion
                num_chars_numeric = sum(c.isdigit() for c in stringx)
                num_chars_alpha = max(sum(c.isalpha() for c in stringx), 1)
                numeric_proportion = num_chars_numeric / (num_chars_numeric + num_chars_alpha)
                
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "n_words"] = n_words
                csv.loc[csv.text_id == text_id, "n_unique_words"] = n_unique_words
                csv.loc[csv.text_id == text_id, "n_sentences"] = n_sentences
                csv.loc[csv.text_id == text_id, "n_pages"] = n_pages
                csv.loc[csv.text_id == text_id, "avg_word_length"] = avg_word_length
                csv.loc[csv.text_id == text_id, "avg_word_incidence"] = avg_word_incidence
                csv.loc[csv.text_id == text_id, "num_chars_numeric"] = num_chars_numeric
                csv.loc[csv.text_id == text_id, "num_chars_alpha"] = num_chars_alpha
                csv.loc[csv.text_id == text_id, "numeric_proportion"] = numeric_proportion
                csv.to_csv(csv_path, index = False)
                             
    def plot_summary_stats(self, text_ids_list, path_prefix, x_labels = None, title = "", summary_stats_col = "n_words"):
        """get plot of various summary statistics over groups of documents.
        parameters:
            :text_ids_list: list[int]: either list of list of text_ids, e.g., [[1,2], [3,4]], or if individual documents rather than groups, [1,2,3,4]
            :path_prefix: str: what the prefix of the files in the csv_outputs/ path is. Need to have run the get_sentiment_csv() function.
            :x_labels: list: what to label the x-axis in the plot, what are the different documents or groups of documents. E.g., decades or years.
            :title: str: additional title to the plot
            :summary_stats_col: str: which column in the summary_stats_col df to plot, one of ["n_words","n_unique_words","n_sentences","n_pages","avg_word_length","avg_word_incidence","num_chars_numeric","num_chars_alpha","numeric_proportion"]
        """
        path_prefix += "_"
        
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}summary_stats.csv"
        
        # only run if file exists
        print(os.path.exists(csv_path))
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
            p, plot_df = self.visualizations.plot_summary_stats(csv, text_ids_list, x_labels, title, summary_stats_col)
            return (p, plot_df)
           
    def plot_text_similarity(self, text_ids, path_prefix = "", label_column = "text_id", figsize = (22,16)):
        """get plot of text similarities using TF-IDF.
        parameters:
            :text_ids_list: list[int]: list of text_ids to compare similarity
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is, leave blank for raw, untransformed text
            :label_column: str: what column from the metadata file to use for labelling on the plot
            :figsize: tuple(int): size of the plot
        output:
            heat map similarity plot
            :pd.DataFrame: with rows and columns showing pairwise text similarity
            :list[str]: axis labels
        """
        return self.visualizations.gen_similarity_plot(self, text_ids, path_prefix, label_column, figsize)
    
    def gen_cluster_df(self, text_id_dict, path_prefix = ""):
        """"given dict of groups + text ids, return two principal components of text similarity via pairwise  TF-IDF
        parameters:
            :text_id_dict: dict{str: list[int]}: keys = grouping of text ids (e.g., publication, year, etc.), values = list of text_ids in group
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is, leave blank for raw, untransformed text
        output:
            :pd.DataFrame: with columns:
                :text_id: text ids
                :group: group of each text id (from the keys of text_id_dict)
                :pc1: first principal component 
                :pc2: second principal component
        """
        return self.visualizations.gen_cluster_df(self, text_id_dict, path_prefix)
    
    def plot_cluster(self, plot_df, color_column = "group"):
        """"given a PCA cluster df, return a scatter plot of text similarity
        parameters:
            :plot_df: pd.DataFrame: output from gen_cluster_df() function
            :color_column: str: column to color scatterplot groups by. Defaults to "group".
        """
        return self.visualizations.plot_cluster(plot_df, color_column)
    
    
    def gen_entity_count_csv(self, text_ids):
        """Gets entity counts from files in txt_files/ directory and writes them to CSV. May have to run "python -m spacy download en_core_web_lg", or equivalent for interested language. See https://spacy.io/models/ for more information. Works from the raw, untransformed text, since capitalization is important for NER.
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s)
        """
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/entity_counts.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "entity_count_dict": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"creating entity count dictionary: {counter}/{len(text_ids)}")
            counter += 1
            
            text_path = f"{self.data_path}txt_files/{text_id}.txt"
            
            # only perform if text file exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "entity_count_dict"].values[0]
            if os.path.exists(text_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading original text
                file = open(f"{text_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                lang = self.metadata.loc[lambda x: x.text_id == text_id, "detected_language"].values[0]
                
                try:
                    entity_dict = self.text_transformation.gen_entity_count_dict(stringx, lang)
                except:
                    print(f"language {lang} not found, performing with English")
                    entity_dict = self.text_transformation.gen_entity_count_dict(stringx, "en")
                    
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "entity_count_dict"] = str(entity_dict)
                csv.to_csv(csv_path, index = False)
                
    def train_bertopic_model(self, text_ids, model_name, notes = "", split_by_n_words = None):
        """"train a BERTopic model from a set of text_ids and saves it to {data_path}/bertopic_models/{model_name}/model and updates a metadatda CSV with information on the model. The document_ids field of that CSV tells the number of smaller documents each larger document was split into, which is necessary for  topic modelling. 
        parameters:
            :text_ids: list[int]: single text_id or list of them to train the topic model
            :model_name: str: descriptive name of the model, will create a directory of this name
            :notes: str: commentary/notes on the model
            :split_by_n_words: int: split longer documents into smaller ones of this word length, leave as None to split by page
        """
        self.text_transformation.train_bertopic_model(self, text_ids, model_name, notes, split_by_n_words)
        
    def load_bertopic_model(self, model_name):
        """"load a trained BERTopic model
        parameters:
            :model_name: str: descriptive name of the model
        """
        return self.text_transformation.load_bertopic_model(self, model_name)
    
    def visualize_bertopic_model(self, model_name, method_name, plot_name, timestamps = None, presence_text_ids = None, presence_topic_ids = None, *args, **kwargs):
        """"save a BERTopic plot to html or png in the model name directory
        parameters:
            :model_name: str: descriptive name of the model
            :method_name: str: visualization function, list of options here: https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html, or the custom one 'visualize_topics_presence'
            :plot_name: str: what to name the saved plot
            :timestamps: list[datetime.datetime]: for the visualize_topics_over_time() function, have to also pass a list of timestamps corresponding to the timestamps for each text_id used to train the original model_name model
            :presence_text_ids: list[int]: which text_ids to include in the plot (only for visualize_topics_presence)
            :presence_topic_ids: list[int]: which topic ids to include in the plot (only for visualize_topics_presence)
            :**kwargs: keyword arguments of the visualization function, e.g., top_n_topics = 10
        """
        model = self.load_bertopic_model(model_name)
        plot_df = self.text_transformation.bertopic_visualize(self, model, model_name, method_name, plot_name, timestamps, presence_text_ids, presence_topic_ids, *args, **kwargs)
        print(f"plot saved to {self.data_path}bertopic_models/{model_name}/{plot_name}.*")
        return plot_df
    
    def replace_words(self, text_ids, replacement_list, path_prefix = ""):
        """"replace words in the text files
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s)            
            :replacement_list: Pandas DataFrame: a dataframe where the first column is the term to be found (can be multi-word), and the second is the term to replace it with
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is, leave blank for raw, untransformed text in the txt_files/ directory
        """
        self.text_transformation.replace_words(self, text_ids, replacement_list, path_prefix)
        
    def gen_search_terms(self, group_name, text_ids, search_terms_df, path_prefix, character_buffer = 100):
        """"get information (context, sentiment, etc.) on occurrences of search terms in a selection of texts
        parameters:
            :group_name: str: what to call this group of texts
            :text_ids: list[int]: list of text ids to analyze
            :search_terms_df: Pandas DataFrame: dataframe with terms to look for. Can have multiple columns ending in most specific terms to search for. For e.g., groupings "grouping">"concept">"permutation". Only the worsd in the rightmost column will actually be searched for, the others are for groupings/aggregations.
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is, leave blank for raw, untransformed text in the txt_files/ directory
            :character_buffer: int: how many characters on either side of the search term to gather for context
        output:
            the following pandas dataframes are written to csv_outputs/
                :search_terms_{group_name}_counts_by_{column_name}.csv: A CSV with counts, count per 1000 words, number and share of texts where the term appears, and the sentiment of the sentence context and character buffer context. The former is the sentence where the term occurs, as well as the sentence prior and after. The latter is the text +- the number of characters of the integer chosen in the character_buffer parameter. This CSV will be duplicated for as many columns as are in the search_terms_df parameter.
                :search_terms_{group_name}_occurrences.csv: A CSV with the context of each individual occurrence of the permutations in the search_terms_df parameter.
        """
        self.search_terms.gen_search_terms(self, group_name, text_ids, search_terms_df, path_prefix, character_buffer = 100)
        
    def gen_aggregated_search_terms(self, group_names, text_ids, search_terms_df, path_prefix, character_buffer = 100):
        """"get information (context, sentiment, etc.) on occurrences of search terms in a selection of texts, aggregated by groups. E.g., for years, etc.
        parameters:
            :group_names: list[str]: a list of the names of the groups of texts, e.g., years, etc.
            :text_ids: list[list[int]]: a list of list of text ids corresponding to the groups
            :search_terms_df: Pandas DataFrame: dataframe with terms to look for. Can have multiple columns ending in most specific terms to search for. For e.g., groupings "grouping">"concept">"permutation"
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is, leave blank for raw, untransformed text in the txt_files/ directory
            :character_buffer: int: how many characters on either side of the search term to gather for context
        output:
            the following pandas dataframe is written to csv_outputs/
                :search_terms_grouped_by_{column_name}.csv: A CSV with counts, count per 1000 words, number and share of texts where the term appears, and the sentiment of the sentence context and character buffer context. The difference to the output from the gen_search_terms() function is that multiple groups can be shown together in the same file, as a new "group" column is added to the output.
        """
        self.search_terms.gen_aggregated_search_terms(self, group_names, text_ids, search_terms_df, path_prefix, character_buffer = 100)        

    def gen_co_occurring_terms(self, group_name, co_occurrence_terms_df, n_words = 50):
        """"get top co-occurring words alongside terms/groups generated from the gen_search_terms function, which must be run first. Context corpus comes from the character_buffer argument of that function.
        parameters:
            :group_name: str: Must correspond to the group name used in the gen_search_terms function
            :co_occurrence_terms_df: Pandas DataFrame: dataframe with terms to look for. Can have multiple columns ending in most specific terms to search for. For e.g., groupings "grouping">"concept">"permutation". Leave a column blank to not consider it. I.e., entering a term for "grouping" and "concept" but not "permutation" will create a search corpus consisting of all permutations present under context.
            :n_words: int: top N co-occurring words to show
        output:
            the following pandas dataframe is written to csv_outputs/
                :search_terms_{group_name}_co_occurrences.csv: A CSV with counts, co-occurrent words and their counts.
        """
        self.search_terms.gen_co_occurring_terms(self, group_name, co_occurrence_terms_df, n_words)
        
    def gen_second_level_search_terms(self, group_name, second_level_search_terms_df):
        """"create sub-corpuses from the character_buffer contextual text from the gen_search_terms function, which must be run first. Then check those sub-corpuses for additional search terms.
        parameters:
            :group_name: str: Must correspond to the group name used in the gen_search_terms function
            :second_level_search_terms_df: Pandas DataFrame: dataframe with terms to look for. Can have multiple columns ending in most specific terms to search for. For e.g., groupings "grouping">"concept">"permutation". Leave a column blank to not consider it. I.e., entering a term for "grouping" and "concept" but not "permutation" will create a search corpus consisting of all permutations present under context. Then with one final column corresponding to the term to search for in this sub-corpus.
        output:
            the following pandas dataframe is written to csv_outputs/
                :search_terms_{group_name}_second_level_counts.csv: A CSV with the second-level search term counts.
        """
        self.search_terms.gen_second_level_search_terms(self, group_name, second_level_search_terms_df)
        
    def gen_top_words(self, group_names, text_ids, path_prefix, per_1000 = True, top_n = 100, exclude_words = []):
        """"create a list of top N words in different documents and groups of documents. If multiple groups are passed, the top_n word list will be constructed from the first group, with subsequent ones left joined to that word list.
        parameters:
            :group_names: list[str]: a list of the names of the groups of texts, e.g., years, etc.
            :text_ids: list[list[int]]: a list of list of text ids corresponding to the groups. A list of lists must be passed.
            :path_prefix: str: 
            :per_1000: bool: whether to get the raw counts or per 1000 words
            :top_n: int: how many top words to show
            :exclude words: list[str]: list of words to exclude from the top words list
        output:
            the following pandas dataframe is written to csv_outputs/
                :top_{top_n}_words.csv: A CSV with the second-level search term counts.
        """
        self.search_terms.gen_top_words(self, group_names, text_ids, path_prefix, per_1000, top_n, exclude_words)
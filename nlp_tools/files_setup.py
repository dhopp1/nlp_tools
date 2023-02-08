import pandas as pd
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import os

def setup_directories(data_path):
    "setup requied and standardized directory structure"
    
    # creating base path
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        
    # path for raw documents, PDFs, HTML, etc.
    if not os.path.isdir(f"{data_path}raw_files/"):
        os.makedirs(f"{data_path}raw_files/")
    
    # path for .txt documents
    if not os.path.isdir(f"{data_path}txt_files/"):
        os.makedirs(f"{data_path}txt_files/")
        
    # path for potential transformed .txt files, e.g., stemmed, stopwords removed, etc.
    if not os.path.isdir(f"{data_path}transformed_txt_files/"):
        os.makedirs(f"{data_path}transformed_txt_files/")
        
    # path for transformed data CSVs
    if not os.path.isdir(f"{data_path}csv_outputs/"):
        os.makedirs(f"{data_path}csv_outputs/")
        
    # path for reports, plots, and outputs
    if not os.path.isdir(f"{data_path}visual_outputs/"):
        os.makedirs(f"{data_path}visual_outputs/")
    

def generate_metadata_file(data_path, metadata_addt_column_names):
    "generate an initial metadata file. metadata_addt_column_names is list of additional columns necessary for the analysis (publication date, etc.)"
    
    # only create the file if it doesn't exist already
    if not(os.path.isfile(f"{data_path}metadata.csv")):
        metadata = pd.DataFrame(
            columns = ["text_id", "web_filepath", "local_raw_filepath", "local_txt_filepath"] + metadata_addt_column_names   
        )
        metadata.to_csv(f"{data_path}metadata.csv", index=False)
    else:
        metadata = pd.read_csv(f"{data_path}metadata.csv")
        metadata.text_id = range(1, len(metadata) + 1) # ensure creation of a unique text id field
    return metadata


def download_document(metadata, data_path, text_id, web_filepath):
    "download a file from a URL and update the metadata file"
    
    # first check if this file already downloaded
    if not(os.path.isfile(f"{data_path}raw_files/{text_id}.html")) and not(os.path.isfile(f"{data_path}raw_files/{text_id}.pdf")):
        response = requests.get(web_filepath)
        content_type = response.headers.get('content-type')
        
        if "application/pdf" in content_type:
            ext = ".pdf"
        elif "text/html" in content_type:
            ext = ".html"
        else:
            ext = ""
        
        if ext != "":
            file = open(f"{data_path}raw_files/{text_id}{ext}", "wb+")
            file.write(response.content)
            file.close()
            
            metadata.loc[metadata.text_id == text_id, "local_raw_filepath"] = f"{data_path}raw_files/{text_id}{ext}"
            return metadata
        else:
            return None
        
        
def parse_pdf(pdf_path):
    "parse a pdf and return text string"
    reader = PdfReader(pdf_path)
    return_text = ""
    for i in range(len(reader.pages)):
        return_text += "[newpage] " + reader.pages[i].extract_text()
        
    return return_text
        
def parse_html(html_path):
    "parse an html file and return text string"
    file = open(html_path, "r", encoding = "UTF-8")
    html = file.read()
    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    
    # get text
    return_text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in return_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    return_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return return_text
    
        
def convert_to_text(metadata, data_path, text_id):
    "convert a PDF or HTML file into raw text. In PDFs, new pages encoded with '[newpage]'"
    raw_path = metadata.loc[lambda x: x.text_id == text_id, "local_raw_filepath"].values[0]
    # path exists (not nan) and is longer than 0
    raw_exists = type(raw_path) == str
    if raw_exists:
         raw_exists = raw_exists & (len(raw_path) > 0)
    
    if raw_exists:
        # first check if this file already converted
        if not(os.path.isfile(f"{data_path}txt_files/{text_id}.txt")):
            # pdf file
            if ".pdf" in raw_path:
                return_text = parse_pdf(raw_path)
            elif ".html" in raw_path:
                return_text = parse_html(raw_path)
            
            # write text file
            file = open(f"{data_path}txt_files/{text_id}.txt", "wb+")
            file.write(return_text.encode())
            file.close()
            
            # update metadata file
            metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"] = f"{data_path}txt_files/{text_id}.txt"
            final_return = metadata
        else:
            final_return = None
    else:
        final_return = None
        
    return final_return
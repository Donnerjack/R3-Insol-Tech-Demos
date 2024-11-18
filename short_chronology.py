#Load libraries
import re
import requests
from langchain_community.document_loaders import TextLoader, BSHTMLLoader
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from langchain_text_splitters import TokenTextSplitter
import pandas as pd
import time
import openai
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup

def scrape_and_process_wiki(wiki_link: str, output_path: str, max_batches: int = 3):
    response = requests.get(wiki_link)
    response.encoding = 'utf-8'
    
    with open("history.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    with open("history.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, 'html.parser')
        document_text = soup.get_text()
    
    document_text = re.sub("\n\n+", "\n", document_text)
    
    class KeyDevelopment(BaseModel):
        date: str = Field(
            ..., description="The date when there was an important historic development (in YYYY-MM-DD format if available, otherwise YYYY-MM or YYYY)."
        )
        description: str = Field(
            ..., description="What happened on this date? What was the development?"
        )
        evidence: str = Field(
            ...,
            description="Repeat in verbatim the sentence(s) from which the date and description information were extracted",
        )
    
    class ExtractionData(BaseModel):
        key_developments: List[KeyDevelopment]
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ])
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    
    extractor = prompt | llm.with_structured_output(
        schema=ExtractionData,
        method="function_calling",
        include_raw=False,
    )
    
    text_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    
    texts = text_splitter.split_text(document_text)
    texts = texts[:(max_batches * 1)]  # Limit to first max_batches * 3 chunks
    
    key_developments = []
    batch_size = 3
    delay_seconds = 1
    max_retries = 5
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        retries = 0
        print(f"Processing batch {(i//batch_size)+1} of {max_batches}")
        
        while retries < max_retries:
            try:
                extractions = extractor.batch(
                    [{"text": text} for text in batch],
                    {"max_concurrency": 5},
                )
                
                for extraction in extractions:
                    key_developments.extend(extraction.key_developments)
                
                break
    
            except openai.RateLimitError as e:
                retries += 1
                print(f"Rate limit error: {e}. Retrying in {delay_seconds * retries} seconds...")
                time.sleep(delay_seconds * retries)
    
            except Exception as e:
                print(f"Error: {e}")
                break
    
        if retries == max_retries:
            print("Max retries reached. Exiting.")
            break
    
        if (i // batch_size + 1) % batch_size == 0:
            time.sleep(delay_seconds)
    
    data = [
        {
            "date": kd.date,
            "description": kd.description,
            "evidence": kd.evidence
        }
        for kd in key_developments
    ]
    
    df = pd.DataFrame(data)
    
    def format_date(date_str):
        for fmt in ['%Y-%m-%d', '%Y-%m', '%Y']:
            try:
                date_obj = pd.to_datetime(date_str, format=fmt)
                if fmt == '%Y-%m-%d':
                    return date_obj.strftime('%d/%m/%Y')
                elif fmt == '%Y-%m':
                    return date_obj.strftime('%m/%Y')
                else:
                    return date_obj.strftime('%Y')
            except ValueError:
                continue
        return date_str
    
    df['formatted_date'] = df['date'].apply(format_date)
    df_sorted = df.sort_values(by="date")
    df_sorted[['formatted_date', 'description', 'evidence']].to_excel(output_path, index=False)
    
    return df_sorted

if __name__ == "__main__":
    wiki_link = input("YOUR_WIKI_URL_HERE:")
    output_path = r"D:\R3-demo\abridged_chronology.xlsx"
    chronology_df = scrape_and_process_wiki(wiki_link, output_path)
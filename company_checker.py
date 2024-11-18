import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests

load_dotenv()

class CompanyAnalyser:
    def __init__(self):
        self.ch_api_key = os.getenv('CH_API_KEY')
        if not self.ch_api_key:
            raise ValueError("Companies House API key not found")
            
        # Fixed base URL to match the working test
        self.base_url = "https://api.companieshouse.gov.uk"
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
    def get_filing_history(self, company_number):
        response = requests.get(
            f"{self.base_url}/company/{company_number}/filing-history",
            auth=(self.ch_api_key, ''),
            params={'category': 'accounts'}
        )
        if response.status_code != 200:
            st.error(f"Filing history API error: {response.text}")
            return {}
        return response.json()
    
    def get_accounts_documents(self, filing_history):
        accounts = []
        for filing in filing_history.get('items', [])[:2]:
            if filing.get('links', {}).get('document_metadata'):
                doc_response = requests.get(
                    filing['links']['document_metadata'],
                    auth=(self.ch_api_key, '')
                )
                if doc_response.status_code == 200:
                    accounts.append(doc_response.json())
        return accounts
    
    def get_officers(self, company_number):
        response = requests.get(
            f"{self.base_url}/company/{company_number}/officers",
            auth=(self.ch_api_key, '')
        )
        if response.status_code != 200:
            st.error(f"Officers API error: {response.text}")
            return {}
        return response.json()
    
    def get_psc(self, company_number):
        response = requests.get(
            f"{self.base_url}/company/{company_number}/persons-with-significant-control",
            auth=(self.ch_api_key, '')
        )
        if response.status_code != 200:
            st.error(f"PSC API error: {response.text}")
            return {}
        return response.json()
    
    def get_company_profile(self, company_number):
        response = requests.get(
            f"{self.base_url}/company/{company_number}",
            auth=(self.ch_api_key, '')
        )
        if response.status_code != 200:
            st.error(f"Company profile API error: {response.text}")
            return {}
        return response.json()
    
    def analyse_company(self, company_number):
        with st.spinner('Analysing company data...'):
            profile = self.get_company_profile(company_number)
            if not profile:
                return "Unable to fetch company profile"
                
            accounts = self.get_accounts_documents(self.get_filing_history(company_number))
            officers = self.get_officers(company_number)
            psc = self.get_psc(company_number)
            
            st.subheader("Company Profile")
            st.json(profile)
            
            st.subheader("Raw Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Filing History")
                st.json(accounts)
            with col2:
                st.write("Officers")
                st.json(officers)
            with col3:
                st.write("PSC")
                st.json(psc)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a financial analyst. Provide a concise summary of the company based on the following data."),
                ("user", """
                Company Profile: {profile}
                Accounts Data: {accounts}
                Current Directors: {officers}
                Persons with Significant Control: {psc}
                
                Please provide:
                1. Company overview (status, incorporation date, address)
                2. Overall level of assets and financial position
                3. List of current directors
                4. Persons with significant control
                5. Any notable financial trends or concerns
                """)
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({
                "profile": profile,
                "accounts": accounts,
                "officers": officers,
                "psc": psc
            })

def main():
    st.title("Companies House Analysis")
    st.write("Enter a company number to analyze its data and structure.")
    
    company_number = st.text_input("Company Number:", value="00048839")
    
    if st.button("Analyse"):
        if company_number:
            analyzer = CompanyAnalyser()
            try:
                analysis = analyzer.analyse_company(company_number)
                if analysis:
                    st.markdown("### AI Analysis")
                    st.write(analysis)
            except Exception as e:
                st.error(f"Error analyzing company: {str(e)}")
        else:
            st.warning("Please enter a company number")

if __name__ == "__main__":
    main()
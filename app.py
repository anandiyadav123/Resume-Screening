import streamlit as st
import numpy as np
import re
import nltk
import pickle


# loading knn
knn =pickle.load(open('knn.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


def clean_resume(text):
    cleantext = re.sub(r'http\S+\S', '', text)
    cleantext1 = re.sub(r'@\s+', '', cleantext)
    cleantext2 = re.sub(r'#\s+', '', cleantext1)
    cleantext3 = re.sub(r'<.*?>', '', cleantext2)  # Remove HTML tags
    cleantext4 = re.sub(r'[\r\n]+', ' ', cleantext3)  # Replace newline characters with space
    cleantext5 = re.sub(r'[^\x00-\x7F]+', ' ', cleantext4)  # Remove non-ASCII characters
    return cleantext5
import string
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    new_text=[]
    for word in text.split():
        if word.lower() in stopwords:
            new_text.append('')
        else:
            new_text.append(word)
    return " ".join(new_text)


# web app
def main():
    st.title("Resume Screening ")
    upload_file=st.file_uploader("Upload your Resume",type=['txt','pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')

        cleaned1=clean_resume(resume_text)
        cleaned2=remove_punctuation(cleaned1)
        cleaned3=remove_stopwords(cleaned2)
        cleaned_resume=tfidf.transform([cleaned3])
        prediction=knn.predict(cleaned_resume)[0]
        st.write(prediction)

        # map category ID to category name
        category_mapping={
            15:"Java Developer",
            23:"Testing",
            8:"DevOps Engineer",
            20:"Python Developer",
            24:"Web Developer",
            12:"HR",
            13:"Hadoop",
            3:"Blockchain",
            10:"ETL Developer",
            18:"Operation Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechanical Engineer",
            1:"Arts",
            7:"Database",
            11:"Electrical Engineer",
            14:"Health and Fitness",
            19:"PMO",
            4:"Business Analyst",
            9:"DotNet Developer",
            2:"Automation Testing",
            17:"Network Engineer",
            21:"SAP Developer",
            5:"Civil Engineer",
            0:"Advocate",
        }

        category_name=category_mapping.get(prediction,'unknown')
        st.write("Predicted Category : ",category_name)

#python
if __name__=="__main__":
    main()
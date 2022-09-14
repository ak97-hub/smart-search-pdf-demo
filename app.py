import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64
import pandas as pd
import numpy as np
import fitz
import pickle
from PIL import Image


class PDFObject:
    # Init variables
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.colors = {'red': [1, 0.8, 0.8],  # light red
                       'aqua': [0.5, 1, 1],  # light aqua
                       'dark_brown': [0.5, 0, 0],  # dark brown
                       'lime': [0, 1, 0],  # Lime green
                       'yellow': None}
        self.pdfFileObj = None
        self.pdfIn = None
        self.pdfIncopy = None
        self.words = None
        self.word_embeddings = None
        self.sentences = None
        self.word_similarities = None

    def read_pdf(self, file):
        self.pdfIn = fitz.open(stream=file, filetype="pdf")  # fitz.open(path)
        self.pdfIncopy = self.pdfIn
        # self.pdfFileObj = open(path, 'rb')
        self._get_texts_()

    def get_word_similarities(self, word):
        input = self._get_word_embeddings_(word)
        word_similarities = self._get_inner_product_(input,
                                                     self.word_embeddings)
        self.word_similarities = word_similarities
        greens = self.words[np.where(word_similarities >= 0.9)[0]]
        yellows = self.words[np.where(
            (word_similarities >= 0.8) & (word_similarities <= 0.8999))[0]]
        reds = self.words[np.where(
            (word_similarities >= 0.50) & (word_similarities <= 0.7999))[0]]

        highlight_dict = {'lime': list(set(greens)), 'red': list(
            set(reds)), 'yellow': list(set(yellows))}

        for color, texts in highlight_dict.items():
            self._highlighter_(texts, color)

    def _get_texts_(self):
        text = ""
        for page in self.pdfIn:
            text += page.get_text() + "\n"

        self.sentences = pd.Series(text.lower().split('.'))
        self.sentences = self.sentences.str.split('\n').str.join(' ')
        self.words = pd.Series(text.lower().split())
        self.word_embeddings = [
            self._get_word_embeddings_(word) for word in self.words]

    def _get_word_embeddings_(self, word):
        try:
            return self.embeddings_dict[word]
        except KeyError:
            return np.zeros(128)

    def _get_inner_product_(self, embedding1, embedding2):
        try:
            return np.inner(embedding1, embedding2)
        except TypeError:
            return 0

    def _highlighter_(self, texts, color):
        for page in self.pdfIn:
            print(page)
            text_instances = [page.search_for(
                " " + text + " ") if len(text.split()) == 1 else
                page.search_for(text) for text in texts]

            # iterate through each instance for highlighting
            for inst in text_instances:
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=self.colors[color])
                annot.update()
        print(type(self.pdfIn))
        self.pdfIn.save("./images/out.pdf")

    def reset_pdf(self):
        for page in self.pdfIn:
            for annot in page.annots():
                page.delete_annot(annot)


@st.cache
def load_embeddings_dict(embeddings_dict_path):
    with open(embeddings_dict_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" \
    width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def show_pdf_obj(input):
    base64_pdf = base64.b64encode(input).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" \
    width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def remove_padding():
    hide_streamlit_style = """
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div
        {padding-top: 2rem;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def show_header(lottie_path):
    f = open(lottie_path)
    data = json.load(f)
    st_lottie(data, height=50, key='coding', speed=1)


def main():
    st.set_page_config(page_title='Smart Scanner',
                       layout='wide',
                       page_icon=':rocket:')
    remove_padding()
    hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 12], gap='small')

    with col1:
        show_header("lottie.json")

    with col2:
        st.image("./images/logo.png")

    st.markdown("""---""")
    st.title('Search Intelligently')

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if not uploaded_file:
        with open("./images/Profile.pdf", "rb") as pdf_file:
            uploaded_file = pdf_file.read()
    if uploaded_file:
        with st.spinner("AI scanning :robot:"):
            if 'emb_dict' not in st.session_state:
                st.session_state.emb_dict = load_embeddings_dict(
                    'word_embeddings3.pickle')
            if 'pdf' not in st.session_state:
                st.session_state.pdf = uploaded_file.getvalue()
                pdfObj = PDFObject(st.session_state.emb_dict)
                pdfObj.read_pdf(st.session_state.pdf)
            else:
                if uploaded_file != st.session_state.pdf:
                    try:
                        st.session_state.pdf = uploaded_file.getvalue()
                    except AttributeError:
                        st.session_state.pdf = uploaded_file
                    pdfObj = PDFObject(st.session_state.emb_dict)
                    pdfObj.read_pdf(st.session_state.pdf)

    pdf_col1, pdf_col2 = st.columns([4, 10])

    with pdf_col1:
        search_value = st.text_input("Search", value="data")
        if search_value:
            pdfObj.reset_pdf()
            if len(search_value.split()) == 1:
                st.write(search_value)
                pdfObj.get_word_similarities(search_value.lower())

        img = Image.open('./images/matches.png')
        img = np.array(img)
        st.image(img)

    with pdf_col2:
        if uploaded_file and not search_value:
            show_pdf_obj(st.session_state.pdf)
        elif uploaded_file and search_value:
            show_pdf_obj(pdfObj.pdfIn.write())
        else:
            show_pdf("./images/profile.pdf")


if __name__ == "__main__":
    main()

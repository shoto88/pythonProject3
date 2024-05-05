import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic

st.set_page_config(page_title="あなたの悩みに合わせた施術を提案します", page_icon=":sparkles:", layout="wide")
# CSSを設定する
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# スタイリングファイルをロード
local_css("style.css")

# Streamlitの設定

# OpenAI APIキーを設定
#openai_api_key= os.getenv("sk-proj-xuIvnkBGjBCUBnG1ZAPqT3BlbkFJKYketmjnPGHRdUCvUpoX")

# Anthropic APIキーを設定
#anthropic_api_key= os.getenv("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")

# 美容施術の情報が記載されたテキストファイルのパス
treatments_file = "treatments.txt"

# 肌の悩みの選択肢を定義
skin_concerns = ["シミ", "シワ", "たるみ", "くすみ", "ニキビ", "乾燥", "毛穴の開き", "ニキビ跡",'そばかす','肌荒れ','肝斑','赤くすみ','小じわ','たるみ']

# 過去の施術歴の選択肢を定義
past_treatments = ["IPL", "ボトックス注射",'レーザーフェイシャル', "ピーリング", "ダーマペン", "イオン導入"]

# テキストファイルからデータをロード
loader = TextLoader(treatments_file)
docs = loader.load()

# テキストを分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embeddingsを作成し、Faissにインデックス化
embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-xuIvnkBGjBCUBnG1ZAPqT3BlbkFJKYketmjnPGHRdUCvUpoX")
db = FAISS.from_documents(splits, embeddings)

# カスタムプロンプトを定義
prompt_template = """
あなたは美容施術の専門家です。
以下の情報に基づいて、最適な美容施術を2つ提案してください。
また、それぞれの施術を提案する理由を、機序や適応の観点から説明してください。
さらに、提案した施術の後におすすめする施術があれば、それも提案してください。
また、それらはいずれも簡潔に述べてください。文章の始まりは、「、と始めてください」
また、回答は以下の形式を参考にして書いてください。
【提示いただいた情報から以下の施術をお勧めします
お勧め施術No.１：（施術名）
理由：
（改行して理由を述べる）
お勧め施術No.2：（施術名）
理由：
（改行して理由を述べる）

また、施術後はフォロー目的に以下の施術を組み合わせることでさらなる改善が見込めます！
（施術名）
理由：（改行して理由を述べる）】
ユーザー情報:
肌タイプ: {skin_type}
肌の悩み: {skin_concerns}
過去の施術歴: {past_treatments}
自由記述の悩み: {free_concerns}

関連する美容施術の情報:
{context}

提案:"""

prompt = PromptTemplate(
    input_variables=["skin_type", "skin_concerns", "past_treatments", "free_concerns", "context"],
    template=prompt_template,
)

# Claude-Sonnet APIを使用してLLMChainを作成
llm = ChatAnthropic(anthropic_api_key="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",model_name="claude-3-opus-20240229", temperature=0.2, max_tokens=1024)
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlitアプリケーション
def main():
    st.title("お勧め施術提案します！")
    st.markdown("<div style='text-align: center; color: pink; font-size: 28px;'>皆様の美容に関するお悩みを解決いたします！</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        skin_type = st.radio("肌タイプを選択してください:", ("普通肌", "敏感肌"), index=1)
        selected_concerns = st.multiselect("肌の悩みを選択してください:", skin_concerns)

        selected_past_treatments = st.multiselect("過去に受けた施術を選択してください:", past_treatments)
    with col2:
        free_concerns = st.text_area("その他の悩みがあれば自由に記述してください:")

    if st.button("おすすめの施術を提案", help="クリックして施術を提案します"):
        # ユーザー情報をクエリとして使用し、関連する施術情報を取得
        query = f"{' '.join(selected_concerns)} {' '.join(selected_past_treatments)} {free_concerns}"
        docs = db.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in docs])

        # LLMChainを使用して施術を提案
        result = chain.invoke({
            "skin_type": skin_type,
            "skin_concerns": ", ".join(selected_concerns),
            "past_treatments": ", ".join(selected_past_treatments),
            "free_concerns": free_concerns,
            "context": context
        })

        st.subheader("おすすめの美容施術")
        st.write(result['text'])

if __name__ == "__main__":
    main()

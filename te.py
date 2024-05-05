import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic

# OpenAI APIキーを設定
#openai_api_key= os.getenv("sk-proj-xuIvnkBGjBCUBnG1ZAPqT3BlbkFJKYketmjnPGHRdUCvUpoX")

# Anthropic APIキーを設定
#anthropic_api_key= os.getenv("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")

# 美容施術の情報が記載されたテキストファイルのパス
treatments_file = "treatments.txt"

# 肌の悩みの選択肢を定義
skin_concerns = ["シミ", "シワ", "くすみ", "ニキビ", "乾燥", "毛穴の開き", "ニキビ跡",'そばかす','肌荒れ','肝斑','赤くすみ','小じわ','たるみ','肌質改善']

# 過去の施術歴の選択肢を定義
past_treatments = ["IPL", "ボトックス注射",'レーザーフェイシャル', "ピーリング", "ダーマペン", "イオン導入"]

# テキストファイルからデータをロード
loader = TextLoader(treatments_file)
docs = loader.load()

# テキストを分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
splits = text_splitter.split_documents(docs)

# Embeddingsを作成し、Faissにインデックス化
embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-xuIvnkBGjBCUBnG1ZAPqT3BlbkFJKYketmjnPGHRdUCvUpoX")
db = FAISS.from_documents(splits, embeddings)

# カスタムプロンプトを定義


# カスタムプロンプトを定義
prompt_template = """
あなたは美容施術の専門家です。

以下の情報に基づいて、最適な美容施術を2つ提案してください。/n

また、それぞれの施術を提案する理由を、機序や適応の観点から説明してください。/n

さらに、提案した施術の後におすすめする施術があれば、それも提案してください。/n

もしも肌質を改善したい場合は、施術の後にホームケアとしてホームケア商品(ガウディスキンorゼオスキンの合う方を判断して選んであげてください)をお勧めしてください。また、その際そのホームケア商品の機序やお勧め理由を詳し述べてください。/n

また、それらはいずれも簡潔に２〜３行以内で述べてください。/n

以下の形式で回答してください。/n

お勧め施術no.1:（施術名)\\n

お勧め理由:（ここで改行）\\n

(理由を述べる)

\\n

お勧め施術no.2:（施術名）\\n

お勧め理由:\\n

(改行して理由を述べる)\\n

また、次回以降は(施術名)を行うことでさらなる効果が見込まれます。

（施術名）

（理由）

ユーザー情報:

肌タイプ: {skin_type}

肌の悩み: {skin_concerns}

過去の施術歴: {past_treatments}

自由記述の悩み: {free_concerns}

関連する美容施術の情報:

{context}

提案:"""

prompt_template = """
あなたは美容施術の専門家です。
以下の情報に基づいて、最適な美容施術を2つ提案してください。/n
また、それぞれの施術を提案する理由を、機序や適応の観点から説明してください。/n
さらに、提案した施術の後におすすめする施術があれば、それも提案してください。/n
もしも肌質を改善したい場合は、施術の後にホームケアとしてホームケア商品(ガウディスキンorゼオスキンの)をお勧めしてください。敏感肌や肌が弱いと感じる人には必ずガウディスキンの方をお勧めしてくださいまた、その際そのホームケア商品の機序やお勧め理由を詳し述べてください。/n
また、それらはいずれも簡潔に箇条書きで述べてください。/n
以下の形式で回答してください。/n


お勧め施術no.1:（施術名)\n
お勧め理由:（ここで改行）\n
(理由を述べる)
\n
お勧め施術no.2:（施術名）\n

お勧め理由:\n
(改行して理由を述べる)\n
また、次回以降は(施術名)を行うことでさらなる効果が見込まれます。
（施術名）
（理由）

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
llm = ChatAnthropic(anthropic_api_key="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",model_name="claude-3-opus-20240229", temperature=0, max_tokens=2048)
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlitアプリケーション
def main():
    st.set_page_config(page_title="美容施術提案アプリ", page_icon=":sparkles:", layout="wide")
    
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #FF69B4;
            text-align: center;
            margin-bottom: 50px;
        }
        .subtitle {
            font-size: 24px;
            font-weight: bold;
            color: #FF1493;
            margin-bottom: 20px;
        }
        .box {
            border: 2px solid #FF69B4;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .stTextArea [data-baseweb=base-input] {
        background-color: #f0f6ff !important;  /* 薄い水色に設定 */
        -webkit-text-fill-color: black;       /* テキストの色を黒に設定 */
        }

        
        </style>
        """,
        unsafe_allow_html=True
        )

    st.markdown('<div class="title">あなたの悩みに合わせた施術を提案します！</div>', unsafe_allow_html=True)

    st.markdown('<div class="subtitle">肌タイプを選択してください</div>', unsafe_allow_html=True)
    skin_type = st.radio("", ("普通肌", "敏感肌"))

    st.markdown('<div class="subtitle">肌の悩みを選択してください</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    selected_concerns = []
    for i, concern in enumerate(skin_concerns):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        if col.checkbox(concern, key=f"concern_{concern}"):
            selected_concerns.append(concern)

    st.markdown('<div class="subtitle">過去に受けた施術を選択してください</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    selected_past_treatments = []
    for i, treatment in enumerate(past_treatments):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        if col.checkbox(treatment, key=f"treatment_{treatment}"):
            selected_past_treatments.append(treatment)

    st.markdown('<div class="subtitle">その他の悩みがあれば自由に記述してください</div>', unsafe_allow_html=True)
    free_concerns = st.text_area("自由に悩みを記述してください", height=200)

    if st.button("おすすめの施術を提案"):
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

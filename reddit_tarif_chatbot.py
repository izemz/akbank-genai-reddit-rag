# Akbank GenAI Bootcamp Projesi: Reddit Tarif Chatbotu (RAG)
# VERİ SETİ: Kaggle Reddit Recipes (CSV)
# DİL YETENEĞİ: İngilizce veriyi işleyip Türkçe yanıt verir.

import os
import sys
import pandas as pd
import streamlit as st
from google import genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def setup_rag_system():
    """Gemini bağlantısını kurar, veriyi yükler ve RAG zincirini oluşturur."""
    
    GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GOOGLE_API_KEY:
        st.error("Gemini API Anahtarı bulunamadı. Lütfen anahtarınızı ayarlayın.")
        return None
        
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    st.write("1. Adım: Reddit CSV veri seti yükleniyor...")
    
    DATA_PATH = "data/recipes.csv" 
    
    docs = []
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            
            documents = []
            for index, row in df.iterrows():
                content = f"TARİF ADI: {row['title']}\n\nTAM TARİF/İÇERİK: {row['comment']}"
                
                if len(content) > 350 and pd.notna(row['comment']) and pd.notna(row['title']):
                    metadata = {
                        "kaynak_baslik": row['title'],
                        "yazan_kullanici": row['user'],
                        "tarih": str(row['date'])
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            docs = documents
            st.success(f"-> '{DATA_PATH}' dosyasından {len(docs)} geçerli Reddit tarifi belgesi yüklendi.")
            
        except Exception as e:
            st.error(f"HATA: CSV yüklenirken veya işlenirken bir sorun oluştu. Detay: {e}")
            return None
    else:
        st.warning(f"!!! '{DATA_PATH}' dosyası bulunamadı. Lütfen BÖLÜM B'deki talimatları uygulayın.")
        docs = [Document(page_content="## Örnek Tarif: Domates Çorbası\nMalzemeler: 3 domates, 1 kaşık un, su. Hazırlanışı: Domatesleri doğra, kavur, un ekle ve suyla kaynat.", metadata={"source": "Demo"})]


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"-> Toplam {len(chunks)} adet veri parçacığı (chunk) oluşturuldu.")

    embeddings = GoogleGenAIEmbeddings(
        model="text-embedding-004", 
        client=client
    )

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        client=client, 
        temperature=0.3 
    )

    system_prompt = (
        "Sen Reddit'teki 'recipes' subreddit'i için bir tarif asistanısın. Görevin, SADECE aşağıdaki 'Bağlam' bölümünde sağlanan İNGİLİZCE tarif bilgilerini kullanarak kullanıcıya **TÜRKÇE** yanıt vermektir. "
        "Eğer bağlamda yeterli bilgi yoksa, kibarca 'Üzgünüm, Reddit verilerimde bu konuda yeterli bilgi bulamadım. Belki başka bir tarif aramalısınız.' şeklinde yanıt ver. "
        "Yanıtına, ilgili tarifin başlığını **(İngilizce Başlığı TÜRKÇE'ye çevirerek)** belirterek başla. Cevaplarını detaylı ve adım adım TÜRKÇE olarak hazırla. "
        "\n\nBağlam: {context}"
    )
    
    question_answer_chain = create_stuff_documents_chain(
        llm, 
        ChatPromptTemplate.from_template(system_prompt)
    )

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    st.success("2. Adım: RAG Zinciri kuruldu ve Chatbot kullanıma hazır.")
    return rag_chain


def main():
    st.set_page_config(page_title="Reddit Tarif Chatbotu", page_icon="🥘")
    st.title("🥘 Reddit Tarif Kitabı (RAG) Chatbotu")
    st.caption("Kaynak: Kaggle 'recipes' subreddit. İngilizce veriden **Türkçe** yanıt üretir.")

    rag_pipeline = setup_rag_system()
    if rag_pipeline is None:
        return 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hangi Reddit tarifini merak ediyorsunuz? (Örn: Tiramisu tarifi nedir?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reddit verilerinde arama yapılıyor ve Türkçe yanıt oluşturuluyor..."):
                try:
                    result = rag_pipeline.invoke({"input": prompt})
                    response = result["answer"]
                except Exception as e:
                    response = f"Sorgu işlenirken bir hata oluştu: {e}"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
# Akbank GenAI Bootcamp Projesi: Reddit Tarif Chatbotu (RAG)

---

## 1. Projenin Amacı (Bootcamp Kriteri 1)

Bu proje, Akbank Generative AI Giriş Bootcamp'i kapsamında **RAG (Retrieval Augmented Generation)** mimarisi kullanarak bir chatbot uygulaması geliştirmeyi amaçlamaktadır.

* **Temel Problem:** Büyük Dil Modelleri (LLM'ler) bazen yanlış veya uydurma bilgi üretebilir.
* **Çözüm:** Harici, spesifik bir veri kaynağını (Reddit tarifleri) kullanarak LLM'in yanıtlarını bu kaynaktaki bilgilerle sınırlandırmak.
* **Dil Yeteneği:** İngilizce olan tarif veri setinden, kullanıcıdan gelen **Türkçe** sorulara doğru, güvenilir ve bağlam bazlı **Türkçe** yanıtlar üretmek.
* **Arayüz:** Proje, bir web arayüzü (Streamlit) üzerinden kullanıcıya sunulmuştur.

---

## 2. Veri Seti Hazırlama (Bootcamp Kriteri 2)

Proje, Kaggle'dan elde edilen ve gerçek dünya verisi içeren bir veri seti kullanmaktadır.

### Veri Seti: Kaggle Reddit Recipes
* **Kaynak:** Kaggle - Reddit Recipes Dataset
* **Format:** CSV (`recipes.csv`)
* **İçerik:** Reddit'in "recipes" subreddit'inden toplanan, tarif başlıkları (`title`) ve tam tarif/açıklama metinlerini (`comment`) içeren gönderiler.

### Veri Seti Hazırlama Metodolojisi
1.  **Yükleme:** CSV dosyası, Pandas kütüphanesi ile `title` ve `comment` sütunları okunarak bir `DataFrame`'e dönüştürülmüştür.
2.  **Filtreleme:** Sadece 350 karakterden uzun, anlamlı metin içeren ve gerekli sütunları dolu olan kayıtlar işlenerek RAG için "Knowledge Base" (Bilgi Tabanı) oluşturulmuştur.
3.  **Dokümantasyon:** Her bir tarif satırı, LangChain'in RAG akışında kullanılmak üzere `Document` nesnelerine çevrilmiştir.

---

## 3. Çalışma Kılavuzu (Bootcamp Kriteri 3)

### 3.1. Ön Gereklilikler
* Python 3.8+
* Git
* Gemini API Anahtarı

### 3.2. Kurulum Adımları
1.  **Projeyi Klonlayın:**
    ```bash
    git clone [SİZİN_GİTHUB_REPO_LİNKİNİZ]
    cd akbank-genai-reddit-rag
    ```
2.  **Sanal Ortamı Kurun ve Etkinleştirin:**
    ```bash
    python -m venv venv
    # Windows için:
    .\venv\Scripts\Activate
    # Mac/Linux için:
    source venv/bin/activate
    ```
3.  **Bağımlılıkları Kurun (`requirements.txt` kullanarak):**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Veri Setini Yerleştirin:** `recipes.csv` dosyasını projenin kök dizinindeki **`data/`** klasörüne kopyalayın.
5.  **API Anahtarını Yapılandırın (Zorunlu):** Projenin kök dizininde **`.streamlit`** klasörünü oluşturun ve içine **`secrets.toml`** dosyasını ekleyerek Gemini API anahtarınızı tanımlayın. (Bu dosya, `.gitignore` sayesinde GitHub'a yüklenmeyecektir.)
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY = "SİZİN_ANAHTARINIZ"
    ```

### 3.3. Projeyi Başlatma
Sanal ortamınız etkin durumdayken Streamlit komutu ile uygulamayı başlatın:
```bash
streamlit run reddit_tarif_chatbot.py

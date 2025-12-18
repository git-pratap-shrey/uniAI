\# uniAI – Syllabus-Aware, Exam-Focused Study Assistant



uniAI is a \*\*Retrieval-Augmented Generation (RAG)\*\* based study assistant designed specifically for \*\*university students\*\*, with a clear priority: \*\*exam scoring over generic learning\*\*.



Unlike general-purpose AI tutors, uniAI is built around syllabus boundaries, unit-wise depth, and the kind of answers students are actually expected to write in exams.



---



\## 🎯 Project Vision



> \*Students don’t need more explanations — they need the \*\*right explanations\*\*, aligned exactly with their \*\*syllabus\*\*, \*\*units\*\*, and \*\*exam patterns\*\*\*.



\### Core Goals



\* Be \*\*syllabus-aware\*\*, not just topic-aware

\* Answer in a \*\*“what to write in exam”\*\* tone

\* Prioritize \*\*definitions, keywords, and structured points\*\*

\* Respect \*\*unit-level boundaries\*\*

\* Clearly distinguish between:



&nbsp; \* \*\*In-syllabus answers\*\* (strict, grounded, no inference)

&nbsp; \* \*\*Out-of-syllabus questions\*\* (explicitly labeled as \*Generic AI Tutor Mode\*)



\### Long-Term Vision



Build a \*\*low-cost, college-wide study assistant\*\* that students can trust for:



\* Exam preparation

\* Theory-heavy subjects

\* Lab viva preparation



---



\## 🧠 What uniAI Can Do (Current Features)



\* Answer questions \*\*only from official notes / PDFs\*\* in syllabus mode

\* Detect and handle:



&nbsp; \* Unit-specific queries (e.g. \*“unit 4 file handling”\*)

&nbsp; \* Follow-up questions (e.g. \*“repeat”\*, \*“summarize”\*)

&nbsp; \* Out-of-syllabus intent (e.g. \*“write code”\*, \*“implementation”\*)

\* Support \*\*multi-turn conversational context\*\* (session-scoped)

\* List \*\*unit-wise topics\*\* and explain \*\*individual concepts\*\*

\* Prevent silent hallucination by:



&nbsp; \* Explicitly saying \*“not found in syllabus”\*

&nbsp; \* Switching to \*\*Generic AI Tutor Mode\*\* when required



---



\## 🧱 Architecture Overview



\### 1️⃣ Data Layer



\* Source: University PDFs (notes, PYQs)

\* Structured as:



```text

source\_code/data/

&nbsp; year\_2/python/

&nbsp;   notes/unit1/

&nbsp;   notes/unit2/

&nbsp;   notes/unit3/

&nbsp;   ...

```



---



\### 2️⃣ Ingestion Pipeline



\* PDF → Text extraction (PyMuPDF, optional OCR)

\* Text cleaning and normalization

\* Fixed-size chunking

\* Metadata tagging:



&nbsp; \* `unit`

&nbsp; \* `category` (notes / pyq)

&nbsp; \* `source`

\* Embedding generation (local)

\* Storage in a persistent vector database



---



\### 3️⃣ Retrieval (RAG)



\* \*\*ChromaDB\*\* used as the vector store

\* Supports two retrieval strategies:



&nbsp; \* \*\*Semantic retrieval\*\* for concept-level queries

&nbsp; \* \*\*Metadata-only retrieval\*\* for unit overviews (topics listing)

\* Correct handling of Chroma filter constraints (`$and` logic)



---



\### 4️⃣ Generation



\* \*\*Gemini (cloud)\*\* used for text generation

\* Prompt design enforces:



&nbsp; \* Exam-oriented tone

&nbsp; \* Syllabus boundaries

&nbsp; \* Clear mode labeling

\* Chat memory is:



&nbsp; \* Short-term

&nbsp; \* Session-scoped

&nbsp; \* Injected only at generation time (never embedded)



---



\### 5️⃣ API \& UI



\* Backend: \*\*Django\*\*

\* API endpoints:



&nbsp; \* `/api/query`

&nbsp; \* `/api/health`

\* Frontend:



&nbsp; \* Minimal HTML + CSS

&nbsp; \* Vanilla JavaScript

&nbsp; \* `marked.js` for markdown rendering

\* Stateless backend (chat history sent from frontend)



---



\## ✅ What Has Been Implemented



\### ✔ Core RAG Pipeline



End-to-end pipeline is functional:



```text

PDF → Text → Chunks → Embeddings → ChromaDB → Retrieval → Answer

```



---



\### ✔ Syllabus Awareness



\* Unit detection (e.g. `unit 1`, `unit 2`, ...)

\* Unit-scoped retrieval

\* Unit overview handling (topics listing)



---



\### ✔ Exam-Focused Answering



\* Definition-first structure

\* Bullet-point formatting

\* Emphasis on keywords

\* Minimal unnecessary elaboration



---



\### ✔ Conversational Memory (Scoped)



\* Handles follow-up queries such as:



&nbsp; \* “repeat”

&nbsp; \* “summarize”

&nbsp; \* “explain that again”

\* Memory is:



&nbsp; \* Short-term

&nbsp; \* Session-based

&nbsp; \* Not stored in the vector database



---



\### ✔ Mode Switching



\* \*\*Syllabus Mode\*\*



&nbsp; \* Strictly grounded in notes

&nbsp; \* No inference

\* \*\*Generic AI Tutor Mode\*\*



&nbsp; \* Explicitly labeled

&nbsp; \* Used only when the question is outside syllabus



---



\### ✔ Practical Engineering Work



\* Correct handling of:



&nbsp; \* ChromaDB filter semantics

&nbsp; \* LLM rate limits

&nbsp; \* Null / error responses

&nbsp; \* Frontend safety for rendering

\* Environment-based configuration using `.env`

\* Cloudflare Tunnel support for sharing the dev server



---



\## 🛠️ Tech Stack



\### Backend



\* Python

\* Django

\* ChromaDB



\### AI / ML



\* Retrieval-Augmented Generation (RAG)

\* Local embeddings via \*\*Ollama (MXBAI)\*\*

\* \*\*Gemini API\*\* for text generation



\### Data Processing



\* PyMuPDF (PDF extraction)

\* Optional OCR support

\* Custom chunking and cleaning logic



\### Frontend



\* HTML + CSS

\* Vanilla JavaScript

\* marked.js (Markdown rendering)



\### Dev \& Infra



\* `.env`-based configuration

\* Git + GitHub

\* Cloudflare Tunnel

\* Local-first, cost-aware design



---



\## 🚧 Current Limitations



\* Fixed-size chunking (semantic chunking planned)

\* No reranking or evaluation metrics yet

\* Only one subject fully ingested (Python)

\* No persistent long-term memory

\* Cloud LLM rate limits during heavy testing



---



\## 🔮 Roadmap



\* Semantic / structure-aware chunking

\* Local generation fallback to reduce cloud cost

\* More subjects and academic years

\* Unit-level indexing and summaries

\* Answer citations and confidence indicators

\* Scalable hosting for college-wide deployment



---



\## ✨ Why uniAI Is Different



Most AI study tools try to \*teach\*.



\*\*uniAI is designed to help students \*score\*.\*\*



It is intentionally:



\* Less creative

\* More constrained

\* More exam-oriented



That trade-off is deliberate — and it defines the project.



---



\## 📌 Status



\* Project stage: \*\*Active development / prototype\*\*

\* Target users (current): Self + small group of classmates

\* Future goal: College-wide deployment (subject to feasibility)



---



If you’re reading this as a reviewer or collaborator: the focus of uniAI is not novelty, but \*\*alignment with real academic needs\*\* and \*\*practical engineering trade-offs\*\*.




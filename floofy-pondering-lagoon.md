# Qwen-COBOL Enterprise Assistant — Piano R&D (Maggio 2026)

## Context

**Progetto**: costruire un assistente LLM specializzato per analisi, spiegazione, refactoring e traduzione di codice COBOL legacy, da deployare on-premise.

**Obiettivo dichiarato**: **superare COBOL-Coder** (SOTA open-source attuale, Qwen2.5-Coder-based) su COBOLEval e COBOL-JavaTrans, mantenendo **Qwen3.6-27B-Instruct dense** come modello base.

**Vincoli irrinunciabili**:
- **Zero budget monetario**. No API a pagamento, no GPU a pagamento, no storage a pagamento.
- Solo risorse free tier: Lightning AI Studio (L40S), Kaggle (2x T4), Google AI Studio (Gemini 2.5 Flash gratuito), HF Hub privato.
- Sistema operativo: Windows 11 → **WSL2 Ubuntu 24.04 obbligatorio** per gnucobol, vLLM, Docker, training scripts.

### Prior art e target da battere

| Modello | COBOLEval Compile | COBOLEval Pass@1 | Java→COBOL Pass@1 |
|---|---|---|---|
| GPT-4o | 41.8% | 16.4 | ~0 |
| **COBOL-Coder (SOTA)** | **73.95%** | **49.33** | **34.93** |
| **Target stretch** | **>75%** | **>50** | **>36** |
| Target realistic | ≥70% | ≥45 | ≥30 |
| Target minimo | ≥60% | ≥35 | ≥20 |

Fonti: [COBOL-Coder paper](https://arxiv.org/html/2604.03986v1), [bloop COBOLEval](https://bloop.ai/blog/evaluating-llms-on-cobol).

**Concorrenza commerciale** (out of scope): [IBM watsonx Code Assistant for Z v2.8](https://www.ibm.com/products/watsonx-code-assistant-z). Non competiamo commercialmente, posizionamento è open-source/R&D.

### Le 5 leve per superare COBOL-Coder (a budget zero)

COBOL-Coder usa solo SFT su Qwen2.5-Coder. Noi compensiamo il gap di foundation di Qwen3.6-27B (non code-pretrained) con:

1. **CPT realistico** (volume disponibile, 2-3 epoch) per embeddare la sintassi
2. **Thinking traces esplicite** (`<thinking>...</thinking>` di Qwen3.6 hybrid mode) in ogni esempio SFT — vero edge
3. **DPO post-SFT** con compiler ground truth — preference tuning assente in COBOL-Coder
4. **Dataset SFT 40-50k esempi** = 7,052 MainframeBench (MIT pronto) + 30-40k generati da teacher self-hosted (XMainframe-10.5b o Qwen vanilla) + 4-6k "gold" via Gemini free tier — zero costo
5. **LazyGraphRAG + Neo4j** a inference time per task multi-programma

**Location**: `c:\Users\AlessandroCatania\Desktop\personal_projects\Argos\Qwen-COBOL` (sibling di ARGOS-2).

---

## Hardware e quota free tier

| Risorsa | Free tier | Uso |
|---|---|---|
| Lightning AI Studio | L40S 48GB, ore non pubblicate (sanity W1 misura) | Training CPT/SFT/DPO + teacher inference batch |
| Kaggle Notebooks | 2x T4 16GB, 30h/settimana | Data prep, validation parallel, teacher inference (slot secondario), eval batch |
| Google AI Studio | Gemini 2.5 Flash, 1500 req/giorno gratis | Quality boost teacher per il 10-15% di esempi critici |
| HF Hub privato | LFS quota standard (~50GB privati gratis) | Storage raw corpus + adapter DoRA checkpoint |
| Local disk Windows | spazio utente | repo, notebook, mini-corpus per debug |

**Critico**: Lightning Drive ha solo **10GB**. Tutto il corpus raw e i checkpoint base devono vivere su HF Hub, scaricati in streaming a training time. Solo gli **adapter DoRA (~500MB-2GB)** vivono nello Studio.

---

## Strategia teacher (zero costo)

| Stage | Sorgente | Volume | Rationale |
|---|---|---|---|
| Import diretto | **MainframeBench (Fsoft-AIC, MIT)** | 7,052 esempi pronti | Già curato e validato. Code summarization (2523) + QA (2598) + MCQ (1931). Zero generazione. |
| Bootstrap "gold" | **Gemini 2.5 Flash** via Google AI Studio free | 4-6k esempi | 1500 req/giorno gratis × 4-5 giorni. Quality boost per task difficili (translation REDEFINES, modernization spaghetti). |
| Bulk teacher | **XMainframe-instruct-10.5b** OR **Qwen3.6-27B vanilla** self-hosted via vLLM su Lightning L40S | 30-40k esempi | XMainframe è già COBOL-aware (DeepSeek-Coder + Mainframe-Instruct), probabilmente teacher migliore di Qwen vanilla. Decisione W3 dopo confronto 200 esempi pilota di ciascuno. |
| DPO rejected | Stesso teacher bulk con prompt "almost correct variant" | 3-5k coppie | Filtro compiler: chosen compila, rejected no. LLM judge (Gemini) valida la differenza è ragionevole. |

**Totale dataset SFT**: ~40-50k esempi (di cui ~14% pronti da MainframeBench). DPO: 3-5k.

**Validazione qualità**: ogni teacher output passa per compiler validation o test-validation automatica prima di entrare nel dataset. Campione 200 esempi ispezionati manualmente.

---

## Storage strategy

Dato il vincolo 10GB Lightning Drive:

- **Raw corpus COBOL** (parsed, validated, deduped): vive su **HF Hub privato** come dataset versionato. Lightning lo legge in **streaming** via `datasets.load_dataset(..., streaming=True)`.
- **Synthetic dataset**: stessa cosa, su HF Hub privato.
- **Modello base Qwen3.6-27B 4-bit**: ~14GB, va in `~/.cache/huggingface/` su Lightning Studio. Bisogna chiarire se rientra nello quota dei 10GB Drive o è separato (verificare in W1, possibile bisogno di rotation cache).
- **Checkpoint DoRA**: solo adapter (500MB-2GB), versionati su HF Hub privato dopo ogni epoch.
- **W&B**: solo metriche e log, no artefatti pesanti.
- **Local Windows**: clone repo, notebook draft, mini-corpus di 1k snippet per debug locale.

**Pipeline a flusso**: data prep produce shard parquet che vengono uploadati a HF Hub immediatamente; training li riconsuma in streaming. Niente full corpus on-disk simultaneamente.

---

## Decisioni architetturali

### Modello base — Qwen3.6-27B-Instruct dense

**Razionale onesto**: parte svantaggiato come foundation rispetto a Qwen2.5-Coder, ma le 5 leve sopra possono colmare e superare.

**Config Unsloth + DoRA**:
- 4-bit nf4 quantization
- DoRA r=128 alpha=256 per CPT, r=64 alpha=128 per SFT/DPO
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Gradient checkpointing `"unsloth"`
- Seq len **8192** (chunking per programmi più lunghi)
- Optimizer `paged_adamw_8bit`, LR `5e-5` (CPT) → `3e-5` (SFT) → `1e-5` (DPO)
- gradient_accumulation per batch effettivo 16-32

**Decision gate fallback (fine W1, post-sanity)**: misurare token/sec su L40S e quota mensile residua. Se completare CPT + SFT + DPO + eval del 27B richiede più ore di quante disponibili → **fallback immediato a Qwen3-14B** (no synthetic data generation sprecata sul 27B).

### Retrieval — LazyGraphRAG + Neo4j + Qdrant
- **LazyGraphRAG** ([Microsoft](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/))
- **Neo4j Community** (free, Docker locale): nodi `Program/Paragraph/Variable/Copybook`, edges `CALLS/PERFORMS/USES/COPIES`
- **Qdrant** (free, Docker locale): embedding `bge-m3` multilingue
- Hybrid: entry vettoriale + traversal grafico + summarization LLM lazy

### Training pipeline — CPT → SFT → DPO

1. **CPT realistico**: volume reale disponibile (probabilmente 20-50M token COBOL validato + augmentation sintetica via teacher su template canonici), 2-3 epoch.
2. **SFT**: 40-60k esempi, 3 epoch, curriculum learning manuale (sort dataset per difficoltà prima di SFTTrainer).
3. **DPO**: 3-5k coppie, 1-2 epoch, LR basso.

### Quantizzazione finale
- AWQ-4bit (~15GB) per consumer GPU
- FP8 weight-activation (~27GB) per L40S/H100

---

## Dataset pubblici disponibili (mappa esaustiva)

Ricerca esaustiva sui dataset COBOL pubblici accessibili a maggio 2026. Tutti gratuiti, divisi per uso nel piano.

### A. Codice raw COBOL (per CPT)

| Dataset | Volume / contenuto | Licenza | Accesso | Uso nel piano |
|---|---|---|---|---|
| **[bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)** | 67.5TB totali / 900B token / 658 linguaggi. COBOL underrepresented (<1% probabilmente) | Permissive + unlicensed | HF Hub, streaming. Filter `lang == 'cobol'` o estensioni `.cob/.cbl/.cpy` | Fonte primaria CPT raw |
| **[bigcode/the-stack-v2-dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)** | Versione deduplicata di v2 | Stesso | HF Hub streaming | Alternativa pre-dedupata (risparmia step) |
| **[bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack)** (v1) | 3TB / 358 linguaggi | Permissive | HF Hub | Backup se v2 ha meno COBOL pulito |
| **[X-COBOL](https://zenodo.org/records/7968845)** ([arxiv 2306.04892](https://arxiv.org/abs/2306.04892)) | 182 repository COBOL minate da GitHub. 8 CSV metadata + directory `COBOL_Files/` | Open access Zenodo | Download diretto Zenodo | CPT raw, complementare a The Stack |
| **[XMainframe Training Corpus](https://github.com/FSoft-AI4Code/XMainframe)** | **236M token** sul mainframe + COBOL constructs (codice GitHub + docs online) | MIT (paper [arxiv 2408.04660](https://arxiv.org/abs/2408.04660)) | GitHub repo XMainframe | **Core CPT corpus** — già curato e dimensionato |
| **[opensourcecobol](https://github.com/opensourcecobol)** + **[Martinfx/Cobol](https://github.com/Martinfx/Cobol)** | Decine di programmi COBOL community (OpenCobol/GnuCOBOL, AS400 COBOL, CL/CLP/CLLE/CBLLE dialects) | Vari permissive | git clone | Diversità di dialetti per CPT |
| **NIST COBOL 85 Test Suite** ([passato da GnuCOBOL](https://gnucobol.sourceforge.io/)) | 9740+ test programs in 1300 gruppi | Public domain (NIST) | Distribuito con gnucobol-bin | **Gold standard sintattico** per CPT + validation |
| **[openmainframeproject/cobol-programming-course](https://github.com/openmainframeproject/cobol-programming-course)** | Materiali educativi + esempi annotati IBM/Open Mainframe Project | Apache 2.0 | git clone | Esempi commentati per code explanation task |
| **MVS 3.8j Turnkey (TK4-)** via [Hercules emulator](http://www.hercules-390.org/) | Programmi COBOL storici (IBM OS/360, DOS/360, MVS 3.8j) | Public domain (pre-1979 IBM) | Hercules + tk4 image | Diversità di stile COBOL legacy storico (opzionale) |
| **[GitHub topic:cobol](https://github.com/topics/cobol?l=cobol)** | Trending + curated COBOL repos | Vari (filtrare permissive) | GitHub Search API | Augmentation X-COBOL con repo più recenti |

**Stima volume CPT realistico**: combinando The Stack v2 (lang=COBOL) + X-COBOL + XMainframe corpus + opensourcecobol + NIST test suite, target **realistico 100-200M token** (XMainframe da sé porta 236M, quindi 100M+ è raggiungibile).

### B. Benchmark di valutazione

| Benchmark | Contenuto | Licenza | Accesso | Uso |
|---|---|---|---|---|
| **[COBOLEval](https://github.com/BloopAI/COBOLEval)** (BloopAI) | 146 problemi HumanEval portati in COBOL con test runner | MIT | git clone | **Eval primaria** vs COBOL-Coder |
| **[COBOL-JavaTrans](https://github.com/COBOL-Coder)** | 143 task pair bidirezionali COBOL↔Java + test cases | Verificare repo (paper Apr 2026) | git clone COBOL-Coder | **Eval secondaria translation** |
| **[CobolCodeBench](https://huggingface.co/datasets/harshini-kumar/CobolCodeBench)** + [Framework](https://github.com/CobolCodeBench/CobolCodeBench-Framework/) | 46 task complessi adattati da BigCodeBench-Hard (financial, data processing) | HF Hub access | `load_dataset("harshini-kumar/CobolCodeBench")` | Eval terziaria, scenari enterprise |
| **[Fsoft-AIC/MainframeBench](https://huggingface.co/datasets/Fsoft-AIC/MainframeBench)** | **7052 esempi** (2598 QA + 1931 MCQ + 2523 COBOL code summarization). 5.92 MB | **MIT** | `load_dataset("Fsoft-AIC/MainframeBench", 'COBOL_code_summarization')` | **Eval ampia + parte del SFT** (summarization 2523 esempi gratis pronti) |
| **[OpenCBS](https://arxiv.org/abs/2206.06260)** | 43 post tecnici ricreati come defect benchmark | Da paper (verificare distribuzione) | Da ricostruire dai 43 posts del paper | Eval defect detection (opzionale, fase 2) |

### C. Dati SFT/Instruct pronti (riducono il volume da generare)

| Dataset | Esempi pronti | Tipo task | Licenza | Uso |
|---|---|---|---|---|
| **MainframeBench COBOL summarization split** | 2,523 esempi | code → summary in inglese | MIT | **Integra direttamente nel SFT** dataset (~6% del totale 40k) |
| **MainframeBench QA split** | 2,598 esempi | Q&A su mainframe/COBOL | MIT | SFT per task "spiegazione concettuale" |
| **MainframeBench MCQ split** | 1,931 esempi | Multiple choice mainframe | MIT | SFT per task "knowledge probing" |
| **COBOL-Coder training data** ([github](https://github.com/COBOL-Coder)) | Sconosciuto al momento — il paper menziona "automated data augmentation pipeline" | Verificare licenza repo | git clone | Se rilasciato: importare direttamente; se meno, replicare pipeline |

**Risparmio reale**: 7,052 esempi MainframeBench (MIT) → -6-10% di esempi da generare via teacher self-hosted.

### D. Modelli COBOL-aware pre-esistenti (reference / teacher alternativi)

| Modello | Parametri | Provenance | Uso nel piano |
|---|---|---|---|
| **[Fsoft-AIC/XMAiNframe-instruct-10.5b](https://huggingface.co/Fsoft-AIC/XMAiNframe-instruct-10.5b)** | 10.5B | DeepSeek-Coder base + Mainframe-Instruct SFT | **Possibile teacher self-hosted alternativo** (più mirato di Qwen vanilla per il dominio COBOL). Entra in L40S 48GB facilmente. |
| **[Fsoft-AIC/XMAiNframe-instruct-7b](https://huggingface.co/Fsoft-AIC/XMAiNframe-instruct-7b)** | 7B | DeepSeek-Coder base + Mainframe-Instruct | Teacher leggero per Kaggle T4 |
| **[Fsoft-AIC/XMAiNframe-base-7b](https://huggingface.co/Fsoft-AIC/XMAiNframe-base-7b)** | 7B | DeepSeek-Coder base only | Reference per benchmark |
| **[muhtasham/santacoder-finetuned-the-stack-cobol](https://huggingface.co/muhtasham/santacoder-finetuned-the-stack-cobol)** | 1.1B | SantaCoder fine-tuned su Stack COBOL | Reference storico (2024-ish) |
| **COBOL-Coder weights** | ~32B (Qwen2.5-Coder based) | Paper [arxiv 2604.03986](https://arxiv.org/html/2604.03986v1) | Verificare se pesi pubblici su HF Hub; se sì → eval di confronto diretto |

**Decisione su teacher self-hosted**: **XMainframe-instruct-10.5b** è probabilmente teacher migliore di Qwen3.6-27B vanilla per il dominio COBOL (è già fine-tunato sul dominio). Lo aggiungiamo come opzione: in W3, generare un sottoinsieme con XMainframe e uno con Qwen vanilla, ispezionare qualità, scegliere quale usare per il bulk.

---

## Strategia di ingest aggiornata

Il modulo `pipeline/ingest.py` consuma in cascata:

1. **XMainframe Training Corpus** (priorità 1, già curato, 236M token, MIT)
2. **The Stack v2 dedup** filter `lang=cobol` (priorità 2, volume incerto ma reale)
3. **X-COBOL** da Zenodo (priorità 3, 182 repo)
4. **opensourcecobol + Martinfx/Cobol + openmainframeproject/cobol-programming-course** (priorità 4, diversità di dialetti + esempi commentati)
5. **NIST COBOL 85 test suite** (priorità 5, gold standard sintattico, distribuito con gnucobol)

Output: dataset HF Hub privato `<username>/cobol-cpt-corpus` versionato.

Per SFT, integrare:
- 7,052 esempi MainframeBench (MIT) direttamente
- 30-40k esempi generati in W3 (XMainframe-instruct-10.5b o Qwen vanilla self-hosted)
- 4-6k esempi "gold" tramite Gemini 2.5 Flash free tier

Per eval:
- COBOLEval (146 problemi) — primario
- COBOL-JavaTrans (143 pair) — translation
- CobolCodeBench (46 problemi enterprise) — secondario
- MainframeBench (sotto-split non usato in SFT) — terziario

---

## Struttura del repository

```
../Qwen-COBOL/
├── README.md                       # Setup WSL2 + Lightning + Kaggle istruzioni
├── pyproject.toml                  # Python 3.11, uv-managed
├── PREREQUISITES.md                # WSL2 setup, Docker setup, account setup
├── config/
│   ├── training_qwen36_27b.yaml
│   ├── training_fallback_14b.yaml
│   └── inference.yaml
├── data/                           # gitignored, vive su HF Hub
├── src/
│   ├── pipeline/
│   │   ├── ingest.py               # Stack v2 + X-COBOL + altri (stream → HF Hub)
│   │   ├── parse.py                # tree-sitter-cobol → AST
│   │   ├── clean.py                # rimozione cols 1-6, 73-80
│   │   ├── dedupe.py               # MinHash LSH datasketch
│   │   ├── validate.py             # gnucobol batch wrapper (NON subprocess per file)
│   │   ├── copybook_inliner.py
│   │   └── difficulty_scorer.py    # per curriculum learning
│   ├── synth/
│   │   ├── import_mainframebench.py # Carica MainframeBench MIT e reformatta in ChatML
│   │   ├── teacher_vllm.py         # XMainframe-10.5b OR Qwen3.6-27B via vLLM batch
│   │   ├── teacher_gemini.py       # Gemini 2.5 Flash via Google AI Studio
│   │   ├── teacher_compare.py      # script W3 per scegliere XMainframe vs Qwen vanilla
│   │   ├── distill_orchestrator.py # routing teacher per task type
│   │   ├── thinking_traces.py
│   │   ├── dpo_pairs.py
│   │   └── curriculum_sampler.py
│   ├── train/
│   │   ├── cpt.py
│   │   ├── sft.py                  # custom curriculum prima di SFTTrainer
│   │   └── dpo.py
│   ├── retrieval/
│   │   ├── graph_builder.py        # AST → Neo4j
│   │   ├── vector_index.py         # Qdrant + bge-m3
│   │   ├── lazy_graphrag.py
│   │   └── hybrid_retriever.py
│   ├── eval/
│   │   ├── cobolceval_runner.py
│   │   ├── cobol_javatrans_runner.py
│   │   ├── compile_check.py        # batch cobc wrapper
│   │   ├── semantic_eq.py
│   │   └── leaderboard_report.py
│   └── deploy/
│       ├── merge_dora.py
│       ├── quantize_awq.py
│       ├── quantize_fp8.py
│       └── vllm_serve.sh
├── notebooks/
│   ├── kaggle_01_data_ingest.ipynb
│   ├── kaggle_02_synth_gen_gemini.ipynb
│   ├── lightning_03_teacher_batch.ipynb  # genera bulk SFT con Qwen vanilla
│   ├── lightning_04_cpt.ipynb
│   ├── lightning_05_sft.ipynb
│   ├── lightning_06_dpo.ipynb
│   └── kaggle_07_eval_full.ipynb
├── docker/
│   ├── neo4j/                      # docker-compose per Neo4j locale
│   ├── qdrant/                     # docker-compose per Qdrant locale
│   └── vllm-inference/             # container deployment
└── docs/
    ├── DATA_CARD.md
    ├── MODEL_CARD.md
    ├── BENCHMARK_RESULTS.md
    └── DELTA_VS_COBOL_CODER.md
```

---

## Timeline (7-8 settimane)

### Settimana 1 — Setup, prereq, sanity, decision gate
**Goal**: infrastruttura pronta, baseline misurato, decisione 27B vs 14B presa.
- **Giorno 1-2**: setup WSL2 Ubuntu 24.04 nativo su Windows. Installare `gnucobol`, `docker`, `tree-sitter`, `tree-sitter-cobol`. Setup Lightning AI account + L40S Studio. Setup Kaggle. Setup Google AI Studio (Gemini API key gratis). Setup HF Hub privato.
- **Giorno 2-3**: creare repo `Qwen-COBOL` con scheletro. Verificare `tree-sitter-cobol` maturità su 10 file COBOL test reali.
- **Giorno 3-4**: **sanity training** su L40S (4-6h), Qwen3.6-27B + DoRA + 500 sample dummy. Misurare token/sec, ore stimate per CPT 50M token / SFT 50k esempi / DPO 5k pairs. Confrontare con quota mensile Lightning visibile dal dashboard.
- **Giorno 4-5**: **eval baseline** Qwen3.6-27B vanilla su COBOLEval (146 problemi). Verificare COBOL-Coder pesi pubblici e rieseguire eval per conferma.
- **Giorno 5**: **DECISION GATE**. Se ore stimate > quota disponibile → switch a Qwen3-14B (più veloce ~3x), regenerare config, rilasciare W2.
- Setup W&B project, leaderboard.

### Settimana 2 — Data pipeline (Kaggle T4)
**Goal**: corpus COBOL validato e deduplicato (target 100-200M token), su HF Hub privato.
- Ingest in cascata da 5 sorgenti pubbliche (vedi sezione "Dataset pubblici disponibili"):
  1. **XMainframe Training Corpus** (236M token, MIT) — fonte primaria
  2. **The Stack v2 dedup** filter `lang=cobol` — fonte secondaria streaming
  3. **X-COBOL** Zenodo (182 repo) — dialetti + metadata
  4. **opensourcecobol + Martinfx/Cobol + openmainframeproject/cobol-programming-course** — diversità
  5. **NIST COBOL 85 test suite** (gold sintattico, distribuita con gnucobol-bin)
- `parse.py`: tree-sitter-cobol → AST in parquet shards.
- `clean.py`: rimuovi cols 1-6 e 73-80, normalizza dialetti (mapping dialect→GnuCOBOL syntax).
- `validate.py`: **batch wrapper** parallel `multiprocessing.Pool` size=CPU_count, batch da 100 file. Target ≤24h su Kaggle (32 CPU). Skip per file già in NIST test suite (assunti corretti).
- `dedupe.py`: MinHash LSH Jaccard 0.7 (saltare se uso `the-stack-v2-dedup`).
- `copybook_inliner.py`: inject `.CPY` in WORKING-STORAGE prompt.
- `difficulty_scorer.py`: LOC, ciclomatic complexity, REDEFINES/OCCURS/CALL count.
- Push corpus pulito su HF Hub privato come dataset versionato `<username>/cobol-cpt-corpus`.
- **Reality check fine W2**: con XMainframe da solo abbiamo 236M token. Target ≥100M dopo cleaning/validation è realistico → no augmentation sintetica necessaria. Se cleaning rimuove più del previsto e si scende sotto 50M, attivare augmentation in W3.

### Settimana 3 — Synthetic data generation (MainframeBench + Gemini + teacher self-hosted)
**Goal**: 40-60k SFT + 3-5k DPO. Mix di dataset pronto + generato.

- **Task A — Import diretto MainframeBench** (MIT license, gratis, pronto):
  - 2,523 COBOL code summarization → task "Code Explanation" (~6% del totale)
  - 2,598 QA mainframe → task "concept explanation" (separato)
  - 1,931 MCQ → task "knowledge probing"
  - Reformatting in ChatML con thinking traces aggiunte dove utile.

- **Task B — Gemini bootstrap** (Kaggle notebook, Google AI Studio API gratis):
  - Genera 4-6k esempi "gold" sui task più difficili: translation REDEFINES complesse, modernization GO TO spaghetti, copybook complessi.
  - 1500 req/giorno × 4 giorni = 6000 req. Caching aggressivo via local SQLite.
  - Output con `<thinking>` traces formato Qwen3.6 hybrid.

- **Task C — Bulk teacher self-hosted** (Lightning L40S vLLM batch):
  - **Decisione teacher**: confrontare XMainframe-instruct-10.5b (già COBOL-aware) vs Qwen3.6-27B vanilla generando 200 esempi con ciascuno. Valutare qualità con ispezione manuale + compiler validation rate. Scegliere il vincitore per il bulk.
  - Servire il teacher scelto via vLLM FP8 (XMainframe-10.5b sta in <10GB FP8; Qwen3.6-27B in ~27GB).
  - Batch inference su 30-40k prompt templates derivati da snippet validati.
  - Distribuire su sessioni notturne / pause Studio per non sforare quota training W4-W6.

- **Task D — DPO pairs**: teacher genera "almost correct" rejected per ogni gold chosen. Prompt template: "Genera una variante quasi-corretta ma con un errore in PIC clause / paragrafo invocato / REDEFINES mapping." Filtro: compiler valida chosen, invalida rejected; se entrambi compilano, test semantico decide. LLM judge (Gemini) per validare la differenza è "ragionevole, non assurda".

- **Task E — CPT augmentation** (solo se W2 reality check < 50M token): teacher genera 100 varianti per template canonici → ~30-50M token aggiuntivi.

- Tutti output su HF Hub privato `<username>/cobol-sft-dataset` e `<username>/cobol-dpo-dataset`.

### Settimana 4 — CPT (Lightning L40S)
**Goal**: Qwen3.6-27B con sintassi COBOL embedded.
- 2-3 epoch su corpus reale (20-50M token) + eventuale augmentation, DoRA r=128 alpha=256.
- Streaming dataset da HF Hub. Checkpoint adapter ogni 1000 step → HF Hub privato.
- **Eval intermedio post-CPT**: rerun COBOLEval senza SFT. Atteso lift +10-15 punti compilation.

### Settimana 5 — SFT con curriculum (Lightning L40S)
**Goal**: modello che segue 4 task type bene.
- 3 epoch su 40-60k esempi, curriculum: epoch 1 easy, epoch 2 misti, epoch 3 hard.
- DoRA r=64 alpha=128, LR 3e-5.
- Logging W&B, early stopping su val_loss.
- **Eval intermedio post-SFT**: COBOLEval target ≥65% compilation, ≥40 Pass@1.

### Settimana 6 — DPO + LazyGraphRAG
**Goal**: preference tuning + retrieval system.
- DPO 1-2 epoch, 3-5k pairs, LR 1e-5. Tempo ~12-24h L40S.
- **Eval post-DPO**: target ≥70% compilation, ≥45 Pass@1.
- In parallelo (locale Docker su WSL2): `graph_builder.py` popola Neo4j; `vector_index.py` indicizza Qdrant; `lazy_graphrag.py` implementato.

### Settimana 7 — Quantizzazione + Deploy + Eval finale
**Goal**: modello servibile + report completo.
- `merge_dora.py`: merge CPT+SFT+DPO adapter nel base.
- `quantize_awq.py` → AWQ 4-bit con Marlin (~15GB).
- `quantize_fp8.py` → FP8 weight-activation (~27GB).
- `vllm_serve.sh`: docker compose vLLM + Open WebUI. Test su L40S Lightning per FP8 (gratis).
- **Eval finale completo** + ablation per leva (CPT-only, +SFT, +DPO, +retrieval).
- `MODEL_CARD.md`, `BENCHMARK_RESULTS.md`, `DELTA_VS_COBOL_CODER.md`.

### Settimana 8 (buffer)
Se target stretch non raggiunto: più augmentation sintetica raw, hyperparam tuning, esperimenti reranking del retrieval.

---

## Librerie e prerequisiti

### Sistema (WSL2 Ubuntu 24.04 obbligatorio)
```bash
sudo apt install gnucobol docker.io docker-compose
# Verificare: cobc --version → almeno 3.2
```

### Python (uv-managed)
```toml
python = ">=3.11,<3.13"
unsloth = ">=2026.4.0"
trl = ">=0.16"
peft = ">=0.14"
transformers = ">=4.50"
bitsandbytes = ">=0.45"
vllm = ">=0.8"
autoawq = ">=0.3"
llm-compressor = ">=0.4"
qdrant-client = ">=1.12"
neo4j = ">=5.20"
tree-sitter = ">=0.23"
tree-sitter-cobol = "*"
datasketch = "*"
google-generativeai = "*"          # Gemini 2.5 Flash via free tier
huggingface-hub = ">=0.26"
datasets = ">=3.0"                 # streaming mode
wandb = "*"
```

---

## File critici e pattern riutilizzabili

Nessun codice in ARGOS-2 è importabile direttamente. Pattern come reference:
- Config YAML: `ARGOS-2/config.yaml`
- Eval harness: `ARGOS-2/eval/gaia_eval.py` come template per `cobolceval_runner.py`

---

## Verifica end-to-end

### Test per fase
1. **W1 setup**: WSL2+gnucobol funzionano (compila 5 file test); sanity training 4h L40S no-OOM; baseline Qwen3.6-27B su COBOLEval misurato; decision gate documentato.
2. **W2 data**: ≥20k snippet validati su HF Hub privato; reality check token CPT.
3. **W3 synth**: 200 esempi random ispezionati → quality OK; copybook inlined; thinking ben formato.
4. **W4 CPT**: val loss decrescente; COBOLEval post-CPT > vanilla baseline di almeno +5pt compile.
5. **W5 SFT**: COBOLEval ≥65% compile, ≥40 Pass@1.
6. **W6 DPO**: COBOLEval ≥70% compile, ≥45 Pass@1; win-rate ≥60% vs SFT-only.
7. **W6 retrieval**: precision@5 ≥75% su query test.

### Test di accettazione
- **Stretch**: >75% compile, >50 Pass@1 (batte COBOL-Coder)
- **Realistic**: ≥70% compile, ≥45 Pass@1 (parity)
- **Minimum**: ≥60% compile, ≥35 Pass@1 (sopra GPT-4o)

### Rischi noti e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Lightning quota L40S insufficiente per 27B | Decision gate W1 → fallback Qwen3-14B; usare Studio solo durante training attivo, spegnere altrimenti |
| Storage 10GB Drive | Streaming pipeline, adapter-only checkpoint, raw su HF Hub |
| COBOL pubblico < 20M token | Augmentation sintetica raw via Qwen vanilla in W3 |
| Foundation gap Qwen3.6 vs Qwen2.5-Coder | CPT più lungo (2-3 epoch su volume reale + augmentation) |
| Qualità teacher self-hosted < Sonnet/GPT-4o | Filtro compiler validation, ispezione manuale W3, Gemini boost per 10-15% critico |
| DPO instability | Filtrare coppie con compiler diff < 1 categoria errore; LLM judge per validazione |
| `tree-sitter-cobol` bug su dialetti | W1 test su 10 file reali; fallback parser regex se troppo instabile |
| vLLM/gnucobol non-Windows | WSL2 prereq esplicito in `PREREQUISITES.md` |
| Compiler validation lento | Batch wrapper invece di 1 subprocess/file; Kaggle CPU 32 core parallel |
| Curriculum non nativo Unsloth | Sort manuale dataset per `difficulty_score` prima di passare a SFTTrainer |
| Target stretch non raggiunto | W8 buffer; realistic e minimum target restano valid success |
| Lightning Studio idle consuma quota | Spegnere manualmente quando non si usa; auto-resume da HF Hub checkpoint |

---

## Out of scope

- Adapter MCP / IDE plugin (fase 2)
- Dialetti non-GnuCOBOL (IBM Enterprise, Micro Focus) (fase 2)
- Training su dati clienti proprietari
- Pubblicazione HF Hub pubblica (decisione separata a fine progetto)
- Stack JAX/TPU
- Competizione commerciale con IBM watsonx
- API teacher a pagamento (Sonnet/GPT-4o)

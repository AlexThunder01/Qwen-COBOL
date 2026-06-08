# Research Log — Qwen-COBOL

Record cronologico onesto del progetto, con numeri esatti, vicoli ciechi e correzioni.
Materia prima per il paper. Aggiornato man mano. (Ultimo agg.: 2026-06-05)

---

## 0. Obiettivo e vincoli

- **Goal**: LLM specializzato COBOL che batta COBOL-Coder (SOTA su COBOLEval).
- **SOTA da battere** (COBOL-Coder, arXiv 2604.03986): **73.95% compile, 49.33% Pass@1**.
- **Vincolo distintivo**: **budget monetario ZERO** — solo free tier (Lightning, Kaggle,
  Google AI Studio, HF Hub, Modal credits, GCP €258 trial). Tutto riproducibile.
- **Benchmark**: COBOLEval (BloopAI), 146 problemi (HumanEval portato in COBOL),
  compilati/eseguiti con GnuCOBOL 3.x.

## 1. Infrastruttura compute (il "come" a budget zero)

| Piattaforma | Ruolo | Note |
|---|---|---|
| Modal (serverless A100/H100) | eval, inference batch (teacher) | crediti free, ~$28 iniziali |
| Lightning AI (L40S, RTX PRO 6000) | tentativo training | crediti free esauriti |
| Kaggle (CPU, 2×T4) | data processing, ingest corpus | 30h/sett free |
| Google AI Studio / Alibaba DashScope | teacher per dati sintetici | free tier |
| GCP (A100, €258 trial) | training finale | quota GPU richiesta |
| HF Hub | storage dataset/adapter | privato |

**Lezione infra**: il free tier è frammentato e ogni piattaforma ha vincoli diversi
(Lightning sessioni limitate, Modal serverless costoso per run lunghe, Kaggle T4 troppo
piccola per 27B). La strategia ottimale è assegnare ogni workload alla piattaforma giusta.

## 2. W1 — Baseline (con CORREZIONE CRITICA)

### Il baseline "sbagliato" (misurazione iniziale)
Primo eval di Qwen3.6-27B su COBOLEval con **raw completion** (prompt → continuazione):
- 32.88% compile, **0.00% Pass@1** (raw), 10.27% Pass@1 (con normalizzazione STOP RUN→GOBACK).
- Conclusione (errata) di allora: "il modello è pessimo in COBOL".

### La scoperta: Qwen3.6-27B è INSTRUCT, non base
Dal model card: "Pre-training & Post-training" → instruction-tuned, **thinking-by-default**,
multimodale. NON esiste versione base pubblica separata. → il raw completion è la modalità
SBAGLIATA per un instruct.

### Il vero baseline (chat mode, harness corretto)
**Qwen3.6-27B vanilla, chat mode, harness corretto: 28.1% compile, 20.55% Pass@1 (30/146).**
- Lo "0%" che ci ha tormentato era un **artefatto di misurazione**, non incapacità.
- Siamo già al **~42% della strada verso SOTA** (49.33%) col modello vanilla puro.

### INSIGHT CHIAVE per la strategia
Dei 41 programmi che compilano, **30 passano (~73%)**. Cioè: **quando il modello compila,
la logica è quasi sempre giusta. Il collo di bottiglia è la COMPILAZIONE (sintassi COBOL),
non l'algoritmo.** → l'SFT deve insegnare a scrivere COBOL che *compila*.

### Analisi dei fallimenti vanilla (105/146 non compilano) — pattern di errore
Distribuzione esatta degli errori cobc (informa la strategia dati):
- 22x `unexpected *` (formato commenti/colonne) · 12x `END PROGRAM` (strutturale)
- 12x `continuation character expected` (righe > 72 char) · 11x `requires one subscript`
  (array senza indice) · 10x `FUNCTION unknown` (es. FUNCTION INT inesistente)
- 8x `+, expecting UNTIL` (espressione in PERFORM VARYING FROM) · vari minori
→ ~34 formato + ~29 sintassi GnuCOBOL + ~15 strutturali. Tutti pattern *insegnabili* con
esempi compile-validi. Il prompt di generate-from-spec è stato aggiornato per fare scrivere
al teacher COBOL fixed-format che evita esattamente questi errori.

## 3. W2 — Corpus COBOL (per CPT)

- **Risultato**: 15.934 programmi, **~45M token** puliti su HF (`cobol-cpt-corpus`).
- **Fonte chiave**: The Stack v2 dedup COBOL (~22k file) scaricato da **Software Heritage S3**
  (la v2 è metadata-only, il content va preso da S3 via blob_id + account AWS). +X-COBOL.
- **Gotcha riproducibilità**: The Stack v2 è metadata-only; XMainframe corpus 236M è
  proprietario (NON pubblico — verificato esaustivamente); il COBOL pubblico facilmente
  accessibile (v1 inline) è scarso (~4M token), ma via SWH S3 si arriva a ~80M raw → 45M puliti.
- **Cleanup**: rimossi 15% non-COBOL (script Python/Plone con ext .cbl, XML, dati), cap file
  giganti a 64k char (un progetto medicale giapponese "ORCA" dominava il 15% dei token).

## 4. W3 — Dati SFT sintetici (multi-teacher)

Dataset finale `cobol-sft-dataset`, ~18k esempi:

| Split | Esempi | Teacher | Task |
|---|---|---|---|
| mainframebench | 7.052 | Fsoft-AIC/MainframeBench (umano, MIT) | explain, QA, MCQ |
| teacher_bulk | 9.851 | XMAiNframe-instruct-10.5b (domain expert) | explain, refactor, translate, debug |
| alibaba_gold | 946 | qwen3-coder-plus / 235b-thinking / max (frontier) | task difficili + reasoning |
| generate_spec_valid | **887** | DeepSeek-V4-pro, GLM-5.1, Qwen3.7-max ecc (frontier multi-famiglia) | **generate-from-spec** (task COBOLEval) |

Nota generate-from-spec (FINALE): **2.002 generati → 887 compile-validi (44.3%)** dopo
re-indentazione (commenti `*` da col 1 a col 7, 1ª riga). Cascata di 7 teacher frontier
multi-famiglia con quote separate (DeepSeek V4-pro, qwen3-coder-next, GLM-5.1, qwen3.7-max
+ snapshot, qwen3-coder-plus-2025-09-23) per diversità + runway. Tasso ~44% per formato a
colonne incoerente dei teacher (specie non-Qwen). 1.115 falliti salvati in `cobol-spec-failed`
con errore cobc (auto-fix loop disponibile ma non necessario: 887 = milestone ottimale, ~5%
del dataset sul task target). **Dataset SFT totale: ~18.7k esempi.**

### Decisioni strategiche W3
- **CPT saltato**: il bottleneck (W1) è comportamento/sintassi, non conoscenza sintattica
  generica; il corpus 45M è troppo piccolo per CPT che renda; budget/tempo. COBOL-Coder ha
  vinto con SFT sintetico, non con CPT.
- **Gemini scartato**: free tier 2.5-flash = 20 RPD (inutilizzabile); alternative free
  (Gemma 4 31B) più deboli dello student → un teacher più debole peggiora i dati.
- **Alibaba DashScope come teacher**: 70M token free su ~70 modelli, inclusi frontier
  (qwen3.7-max, 235b-thinking) che battono lo student Qwen3.6-27B → distillazione valida.

### Provenienza dati / teacher usati (per riproducibilità)

**Teacher di distillazione, per split:**
- `teacher_bulk` (9.851): **Fsoft-AIC/XMAiNframe-instruct-10.5b** (domain expert COBOL,
  self-hosted via transformers su Modal A100).
- `alibaba_gold` (946): cascata DashScope — `qwen3-coder-plus`, `qwen3-235b-a22b-thinking-2507`
  (thinking), `qwen3-max`, `qwq-plus`.
- `generate_spec` 1ª run (433): cascata DashScope — `qwen3.7-max-preview`, `qwen3.7-plus`,
  `qwen3.6-plus`, `qwen3-max`, `qwen3.6-max-preview`.
- `generate_spec` espansione (in corso): cascata multi-famiglia — **`deepseek-v4-pro`** (DeepSeek,
  famiglia diversa), `qwen3-coder-next` (code specialist), **`glm-5.1`** (Zhipu GLM, famiglia
  diversa), `qwen3.7-max` (+snapshot 2026-05-20/05-17), `qwen3-coder-plus-2025-09-23`,
  `qwen3.7-plus-2026-05-26`. Diversità di famiglia (DeepSeek+GLM) per ridurre il bias Qwen-centrico.
- `mainframebench` (7.052): NON distillato — esempi umani curati (Fsoft-AIC/MainframeBench, MIT).

**Principio cascata**: si usano teacher ≥ student (Qwen3.6-27B, ~77% SWE-bench); ogni modello
DashScope ha quota free separata (1M+1M token), si passa al successivo all'esaurimento.

**Teacher SCARTATI (negative results, importanti per il paper):**
- **Gemini 2.5/3/3.5-flash**: free tier = **20 RPD** (non 1500 come per le vecchie 1.5) →
  inutilizzabile per volume. La 1.5-flash è deprecata.
- **Gemma 4 31B / Gemini 3.1 Flash-Lite** (1500 RPD): scartati perché **più deboli dello
  student** → un teacher inferiore degrada i dati (distillazione controproducente).
- **XMainframe corpus 236M**: proprietario, NON pubblico (verificato). Usato solo il MODELLO
  (XMAiNframe-10.5b) come teacher, non il corpus.

### Tecnica: generate-from-spec + compile-validation + auto-fix loop
Gap diagnosticato: i dati erano tutti explain/refactor, ma COBOLEval testa generate-from-spec.
Generati ~433 esempi (scheletro COBOL + spec → programma completo) con teacher frontier,
poi **compile-validati con GnuCOBOL** (solo i compilanti diventano gold). I falliti +
errore cobc vanno in un **auto-fix loop**: rimandati al teacher per la correzione guidata
dall'errore, ri-compilati, i recuperati rientrano. (Self-correction guidata dal compilatore.)

## 5. W4 — Training (il viaggio, in corso)

### Tentativi e lezioni
- **Lightning RTX PRO 6000 (Blackwell, 96GB)**: 27B QLoRA a 29-36s/it → 1 epoca ~$23, fuori
  budget. Tentato 4-bit→bf16, gradient checkpointing off, DoRA→LoRA. Il packing di unsloth
  veniva IGNORATO (1116 step invece dei ~400 attesi). Solo ~300 step fatti prima di esaurire
  i crediti → checkpoint sotto-addestrato.
- **Lezione packing**: unsloth ignora `packing=True` in SFTConfig → fatto **packing MANUALE**
  (tokenizza tutto, concatena con EOS, chunk da 2048) → ~400 step per epoca, 3x meno.
- **SFT-300 (checkpoint sotto-addestrato)**: con thinking OFF produce risposte VUOTE (il
  modello è addestrato sul formato thinking → senza thinking emette EOS subito); con thinking
  ON rambla (~4000 token di reasoning, lento). Inconcludente, è un throwaway.

### Setup finale (GCP)
- `scripts/train_sft.py` standalone: **bf16** su A100-80GB (qualità piena, no tassa dequant),
  fallback `--load-4bit` per A100-40GB. Packing manuale, 1 epoca, push su HF.
- **Adapter: DoRA r=128 + rsLoRA** (scelto per qualità massima robusta, avendo memoria/budget
  su A100-80GB). DoRA > LoRA (vicino al full-FT ai ranghi bassi); rsLoRA stabilizza il rango
  alto. ~1.4x più lento di LoRA, accettabile. GaLore (full-param, ceiling più alto) scartato
  per rischio implementativo dopo i molti bug infra — affidabilità > guadagno marginale su SFT.
  NB: inizialmente DoRA→LoRA su Lightning per il budget; ripristinato DoRA ora che c'è margine.
- Budget Modal esaurito (~$7) → training spostato su GCP €258.

## 6. Metodologia di eval e la SAGA del bug del harness (lezioni per il paper)

Per giorni TUTTI gli eval davano ~0% Pass@1 → si credeva modello/training rotti. Era un
**bug di estrazione del testo** in chat mode. Catena di bug trovati guardando l'errore reale
di cobc (non indovinando):
1. **Innesto sul prompt** → WORKING-STORAGE duplicata + codice a colonna 1.
2. **Regex ```cobol con `\s*` goloso** → mangiava l'indentazione della 1ª riga → header a
   colonna 1 → `invalid indicator at column 7` (formato a colonne COBOL rotto).
3. **Stesso bug in 3 script** (eval, validazione, generazione) — quello in generazione ha
   CORROTTO i dati generate-from-spec (1ª riga a col 1).

**Fix**: regex `r"```(?:cobol)?[ \t]*\r?\n(.*?)```"` (preserva indentazione); usare il
programma intero del modello (interfaccia copiata dal prompt = corretta), non innestare;
GOBACK normalization per i sottoprogrammi; re-indentazione dei commenti `*` a colonna 7.

**LEZIONE METODOLOGICA (paper-worthy)**: per valutare un modello chat su un benchmark di
*completion* con linguaggio *column-sensitive* (COBOL), l'estrazione del codice dalla
risposta conversazionale è il punto fragile e silenzioso. Un bug di estrazione produce 0%
indistinguibile da un fallimento del modello. → **guardare sempre l'errore del compilatore,
non ipotizzare.** Il vero baseline (20.55%) era nascosto da bug di misurazione.

## 7. Risultati finora

| Modello | Compile | Pass@1 | Note |
|---|---|---|---|
| COBOL-Coder (SOTA) | 73.95% | 49.33% | da battere |
| Qwen3.6-27B vanilla (chat, harness corretto) | **28.1%** | **20.55%** | vero baseline |
| Qwen3.6-27B SFT-300 (throwaway) | — | — | sotto-addestrato, inconcludente |
| Qwen3.6-27B SFT pieno | TBD | TBD | in arrivo (GCP) |

## 8. Contributi/lezioni per il paper

1. **Budget-zero è fattibile ma frammentato**: metodologia riproducibile per specializzare
   un 27B in un dominio di nicchia usando solo free tier.
2. **Il vero baseline di un instruct moderno su COBOL è già non banale** (20.55%): metà del
   lavoro è misurare correttamente. Il bottleneck è la sintassi, non la logica.
3. **Pitfall di misurazione**: chat-mode eval su completion-benchmark column-sensitive →
   l'estrazione del codice è una fonte di errore sistematica e invisibile (0% fasullo).
4. **Distillazione multi-teacher a budget zero**: umano (MainframeBench) + domain expert
   (XMAiNframe) + frontier (Alibaba) con compile-validation e auto-fix loop guidato dal
   compilatore.
5. **The Stack v2 via Software Heritage S3** come fonte COBOL riproducibile (vs corpora
   proprietari come XMainframe-236M).

## 9. Environment / Riproducibilità

### Benchmark
- **COBOLEval**: github.com/BloopAI/COBOLEval @ commit
  `0bb96c3114bb2bb28e221e9d6000614781f8609d` (pinnato per riproducibilità).
- 146 problemi (`data/CobolEval.jsonl`). Patch a `scripts/evaluation.py` per abilitare
  l'esecuzione del binary (nel repo originale è commentata per security; container isolato).
- **GnuCOBOL** 3.x (apt `gnucobol`, Debian trixie). Flag compile-check: `cobc -w -fformat=variable -c`.

### Metodologia eval (modal_eval_sft.py)
- Modello in **chat mode** (Qwen3.6-27B è instruct). `apply_chat_template` con
  `enable_thinking` (False=output diretto, True=reasoning, max_tokens 2048/8192).
- **swap_sections**: riordina begin→working_storage→linkage→procedure; forza
  `PROCEDURE DIVISION USING LINKED-ITEMS.` (replica `scripts/generate.py` del paper COBOLEval).
- **Estrazione**: regex `r"```(?:cobol)?[ \t]*\r?\n(.*?)```"` (preserva indentazione),
  prende il blocco completo più lungo (IDENTIFICATION+PROCEDURE).
- **GOBACK normalization**: `STOP RUN` → `GOBACK` (i sottoprogrammi devono ritornare al caller).
- Sampling: `temperature=0.0` (greedy, deterministico). `gpu_memory_utilization=0.92`.

### Stack software
- Eval: `python:3.11-slim-trixie`, **vLLM 0.21.0** (img spec `>=0.8.0`), `transformers>=4.50`,
  `huggingface_hub>=0.26`, env `VLLM_USE_FLASHINFER_SAMPLER=0`. GPU A100-80GB (Modal).
- Training (`scripts/train_sft.py`): `torch`, `transformers>=4.51`, `peft>=0.13`,
  `accelerate>=1.0`, `bitsandbytes>=0.44`, `datasets`. LoRA r=64, α=128, dropout=0,
  target = tutti i proj (q,k,v,o,gate,up,down). Packing manuale a 2048 token.
  batch 4 × grad-accum 4 = eff 16, lr 2e-4 cosine, warmup 0.05, 1 epoca, optim paged_adamw_8bit.
  bf16 (A100-80GB) o 4-bit nf4 (A100-40GB).
- Teacher self-hosted (XMAiNframe): `transformers` su Modal A100 (no vLLM per conflitto `aimv2`).
- DashScope: endpoint `dashscope-intl.aliyuncs.com/compatible-mode/v1` (OpenAI-compatible).

### Modelli e dati
- Base/student: **Qwen/Qwen3.6-27B** (instruct, thinking-by-default, multimodale), bf16.
- Corpus CPT: The Stack v2 (`bigcode/the-stack-v2-dedup`, config COBOL) + Software Heritage S3
  (`s3://softwareheritage/content/{blob_id}`, account AWS) + X-COBOL.
- Repo HF (privati): `AlexThunder0/cobol-cpt-corpus`, `cobol-sft-dataset`,
  `cobol-spec-failed`, `qwen-cobol-27b-sft`.

### Compute
- Modal (eval, teacher inference): A100-80GB / H100, ~$0.000694/s A100.
- GCP (training finale): A100-80GB, regione us-central1, Deep Learning VM `common-cu124`.
- Kaggle (corpus ingest): CPU / 2×T4. Lightning (tentativi training): L40S, RTX PRO 6000.

## 10. Aperti / da fare

- [ ] Training SFT pieno (GCP A100-80GB, bf16, 1 epoca)
- [ ] Eval SFT finale vs vanilla 20.55% e vs SOTA 49.33%
- [ ] Analisi failure modes post-SFT → iterazione mirata (DPO? più generate-from-spec?)
- [ ] Aggiornare MODEL_CARD/DATA_CARD/BENCHMARK_RESULTS coi numeri reali finali

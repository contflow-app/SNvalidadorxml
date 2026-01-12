import io
import os
import re
import gc
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from lxml import etree

# Optional (IA)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Config
# =========================
APP_TITLE = "Validador NFCom 62 — Lote Grande (ZIP) + IA (Consolidado)"
LOGO_PATH = "Logo-Contare-ISP-1.png"
TRAINING_CSV = os.path.join("data", "training_data.csv")

CATEGORIES = ["SCM", "SVA_EBOOK", "SVA_LOCACAO", "SVA_TV_STREAMING", "SVA_OUTROS"]
AI_ALLOWED = set(CATEGORIES)

ALERTA_CCLASS = "1100101"
ALERTA_TEXTO = (
    "⚠️ Atenção: Foi identificado cClass **1100101** no lote. "
    "Esse cClass indica que o item demonstrado na NFCom foi faturado por outra empresa "
    "do grupo econômico ou terceiros. É obrigatório o colaborador verificar esta situação."
)

st.set_page_config(page_title=APP_TITLE, layout="wide")


# =========================
# Workspace (disco /tmp)
# =========================
def get_workspace_dir() -> str:
    if "WS_DIR" not in st.session_state:
        st.session_state["WS_DIR"] = tempfile.mkdtemp(prefix="nfcom_ws_")
    return st.session_state["WS_DIR"]


def ws_path(*parts) -> str:
    p = Path(get_workspace_dir(), *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def read_file_text(path: str, max_chars: int = 400_000) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            b = f.read()
        t = b.decode("utf-8", errors="ignore")
        if len(t) > max_chars:
            return t[:max_chars] + "\n\n[...cortado para preview...]\n"
        return t
    except Exception as e:
        return f"[Falha ao abrir arquivo: {e}]"


# =========================
# Text utils
# =========================
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    trans = str.maketrans(
        {
            "á": "a", "à": "a", "ã": "a", "â": "a",
            "é": "e", "ê": "e",
            "í": "i",
            "ó": "o", "õ": "o", "ô": "o",
            "ú": "u",
            "ç": "c",
        }
    )
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(str(x).replace(",", "."))
    except Exception:
        return 0.0


def num_to_br(x) -> str:
    try:
        v = float(x)
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(x)


# =========================
# XML helpers
# =========================
def parse_xml(xml_bytes: bytes) -> etree._ElementTree:
    # recover=True para XMLs “sujos” (lote real)
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    return etree.parse(io.BytesIO(xml_bytes), parser)


def get_ns(tree: etree._ElementTree) -> Dict[str, str]:
    root = tree.getroot()
    default_ns = root.nsmap.get(None)
    return {"n": default_ns} if default_ns else {}


def xp(node, ns, expr: str):
    if ns:
        try:
            return node.xpath(expr, namespaces=ns)
        except Exception:
            return node.xpath(expr)
    return node.xpath(expr)


def first_text(node, ns, expr: str) -> str:
    nodes = xp(node, ns, expr)
    if not nodes:
        return ""
    n = nodes[0]
    if isinstance(n, etree._Element):
        return (n.text or "").strip()
    return str(n).strip()


def extract_chave_acesso(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    for path in [
        ".//n:infNFCom/@Id", ".//infNFCom/@Id",
        ".//n:infNFe/@Id", ".//infNFe/@Id",
        ".//n:infCte/@Id", ".//infCte/@Id",
    ]:
        ids = xp(root, ns, path)
        if ids:
            m = re.search(r"\d{44}", str(ids[0]))
            if m:
                return m.group(0)
    xml_str = etree.tostring(root, encoding="unicode")
    m2 = re.search(r"\d{44}", xml_str)
    return m2.group(0) if m2 else ""


def get_nf_model(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    return first_text(root, ns, ".//n:ide/n:mod | .//ide/mod").strip()


def get_emitente(tree: etree._ElementTree) -> Tuple[str, str]:
    root = tree.getroot()
    ns = get_ns(tree)
    cnpj = first_text(root, ns, ".//n:emit/n:CNPJ | .//emit/CNPJ").strip()
    xnome = first_text(root, ns, ".//n:emit/n:xNome | .//emit/xNome").strip()
    return cnpj, xnome


def get_competencia_mes(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)
    comp = first_text(root, ns, ".//n:gFat/n:CompetFat | .//gFat/CompetFat").strip()
    if comp:
        m = re.search(r"(\d{4})[-/]?(\d{2})", comp)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        return comp[:7]
    demi = first_text(root, ns, ".//n:ide/n:dEmi | .//ide/dEmi").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", demi)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return datetime.now().strftime("%Y-%m")


def get_ufs_emit_dest(tree: etree._ElementTree) -> Tuple[str, str]:
    root = tree.getroot()
    ns = get_ns(tree)
    uf_emit = first_text(root, ns, ".//n:emit/n:enderEmit/n:UF | .//emit/enderEmit/UF")
    uf_dest = first_text(root, ns, ".//n:dest/n:enderDest/n:UF | .//dest/enderDest/UF")
    return uf_emit.strip(), uf_dest.strip()


def is_dest_pf_or_pj_nao_contrib(tree: etree._ElementTree) -> bool:
    root = tree.getroot()
    ns = get_ns(tree)
    dest = xp(root, ns, ".//n:dest | .//dest")
    if not dest:
        return False
    d = dest[0]
    cpf = first_text(d, ns, "./n:CPF | ./CPF").strip()
    cnpj = first_text(d, ns, "./n:CNPJ | ./CNPJ").strip()
    ind = first_text(d, ns, "./n:indIEDest | ./indIEDest").strip()
    # mesma regra que você já usava
    if cpf and (not ind or ind in ("2", "9")):
        return True
    if cnpj and ind == "9":
        return True
    return False


def expected_cfop_scm(tree: etree._ElementTree) -> str:
    uf_e, uf_d = get_ufs_emit_dest(tree)
    if uf_e and uf_d:
        return "5307" if uf_e == uf_d else "6307"
    # fallback
    return "5307"


# =========================
# Cancelamento detection
# =========================
def contains_cancel_words(text: str) -> bool:
    t = normalize_text(text or "")
    return ("cancelamento" in t) or ("cancelad" in t) or ("cancelou" in t)


def detect_cancelamento_event_bytes(xml_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Detecta XML de evento cancelamento:
      tpEvento=110111 e xEvento/xMotivo com 'cancel'
    """
    try:
        tree = parse_xml(xml_bytes)
    except Exception:
        return (False, None)

    root = tree.getroot()
    ns = get_ns(tree)
    tp = first_text(root, ns, ".//n:tpEvento | .//tpEvento")
    if tp != "110111":
        return (False, None)

    xevt = first_text(root, ns, ".//n:xEvento | .//xEvento")
    xmot = first_text(root, ns, ".//n:xMotivo | .//xMotivo")
    if not (contains_cancel_words(xevt) or contains_cancel_words(xmot)):
        return (False, None)

    ch_nfcom = first_text(root, ns, ".//n:chNFCom | .//chNFCom")
    if ch_nfcom:
        return (True, ch_nfcom)

    xml_str = etree.tostring(root, encoding="unicode")
    m = re.search(r"\d{44}", xml_str)
    return (True, m.group(0) if m else None)


def detect_cancelamento_by_words(xml_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Fallback: se achar cancelamento em xEvento/xMotivo em qualquer XML.
    """
    try:
        tree = parse_xml(xml_bytes)
    except Exception:
        return (False, None)

    root = tree.getroot()
    ns = get_ns(tree)
    textos: List[str] = []
    for n in xp(root, ns, ".//n:xMotivo | .//xMotivo | .//n:xEvento | .//xEvento"):
        if isinstance(n, etree._Element) and n.text:
            textos.append(n.text)
    if not any(contains_cancel_words(t) for t in textos):
        return (False, None)
    return (True, extract_chave_acesso(tree))


# =========================
# Training (aprendizado)
# =========================
def training_init():
    os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
    if not os.path.exists(TRAINING_CSV):
        pd.DataFrame(
            columns=["emit_cnpj", "desc_norm", "descricao_exemplo", "categoria_aprovada", "created_at", "source"]
        ).to_csv(TRAINING_CSV, index=False, encoding="utf-8")


@st.cache_data
def training_load() -> pd.DataFrame:
    training_init()
    try:
        return pd.read_csv(TRAINING_CSV, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=["emit_cnpj", "desc_norm", "descricao_exemplo", "categoria_aprovada", "created_at", "source"])


def training_append(rows: List[Dict[str, Any]]):
    training_init()
    df = training_load()
    df2 = pd.DataFrame(rows)
    pd.concat([df, df2], ignore_index=True).to_csv(TRAINING_CSV, index=False, encoding="utf-8")
    training_load.clear()


def training_lookup_map(df_train: pd.DataFrame, emit_cnpj: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if df_train.empty:
        return m

    if emit_cnpj:
        df_c = df_train[df_train["emit_cnpj"] == emit_cnpj]
        for _, r in df_c.iterrows():
            dn = r.get("desc_norm", "")
            cat = r.get("categoria_aprovada", "")
            if dn and cat:
                m[dn] = cat

    df_g = df_train[df_train["emit_cnpj"] == ""]
    for _, r in df_g.iterrows():
        dn = r.get("desc_norm", "")
        cat = r.get("categoria_aprovada", "")
        if dn and cat and dn not in m:
            m[dn] = cat
    return m


def training_merge_uploaded(uploaded_file) -> Tuple[bool, str]:
    """
    Aceita:
      - Interno: emit_cnpj, desc_norm, categoria_aprovada
      - Simples: CNPJ, descricao, CLASSIFICACAO VALIDADA
    Detecta header "Unnamed" em linhas iniciais.
    """
    training_init()
    name = (getattr(uploaded_file, "name", "") or "").lower()
    data = uploaded_file.read()

    def _read_any() -> pd.DataFrame:
        if name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(data), dtype=str).fillna("")
        return pd.read_csv(io.BytesIO(data), dtype=str).fillna("")

    def _read_with_header_row(header_row: int) -> pd.DataFrame:
        if name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(data), dtype=str, header=header_row).fillna("")
        return pd.read_csv(io.BytesIO(data), dtype=str, header=header_row).fillna("")

    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols_raw = {c: normalize_text(str(c)).replace(" ", "_") for c in df.columns}
        return df.rename(columns=cols_raw)

    try:
        df_in = _read_any()
    except Exception as e:
        return False, f"Não consegui ler a base: {e}"

    df_in = _norm_cols(df_in)

    if all(str(c).startswith("unnamed") for c in df_in.columns) or len(df_in.columns) <= 2:
        if name.endswith(".xlsx"):
            df_raw = pd.read_excel(io.BytesIO(data), dtype=str, header=None).fillna("")
        else:
            df_raw = pd.read_csv(io.BytesIO(data), dtype=str, header=None).fillna("")
        header_idx = None
        for i in range(min(10, len(df_raw))):
            row = " ".join([normalize_text(x) for x in df_raw.iloc[i].astype(str).tolist()])
            if "descricao" in row and ("categoria" in row or "classificacao" in row):
                header_idx = i
                break
        if header_idx is not None:
            try:
                df_in = _norm_cols(_read_with_header_row(header_idx))
            except Exception:
                pass

    def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    is_internal = {"emit_cnpj", "desc_norm", "categoria_aprovada"}.issubset(set(df_in.columns))
    now = datetime.now().isoformat(timespec="seconds")

    if is_internal:
        df_norm = df_in.copy()
        df_norm["created_at"] = df_norm.get("created_at", now)
        df_norm["source"] = df_norm.get("source", "importado")
        if "descricao_exemplo" not in df_norm.columns:
            df_norm["descricao_exemplo"] = df_norm["desc_norm"]
    else:
        col_cnpj = pick_col(df_in, ["cnpj", "emit_cnpj"])
        col_desc = pick_col(df_in, ["descricao", "descrição", "xprod", "produto", "servico", "serviço"])
        col_class = pick_col(df_in, ["classificacao_validada", "classificacao", "classificação", "categoria", "categoria_aprovada"])
        if not col_desc or not col_class:
            return False, f"Layout não reconhecido. Colunas encontradas: {list(df_in.columns)}"

        df_norm = pd.DataFrame()
        if col_cnpj:
            df_norm["emit_cnpj"] = df_in[col_cnpj].astype(str).str.replace(r"\D+", "", regex=True)
        else:
            df_norm["emit_cnpj"] = ""

        df_norm["descricao_exemplo"] = df_in[col_desc].astype(str)
        df_norm["desc_norm"] = df_norm["descricao_exemplo"].map(normalize_text)

        def map_cat(x: str) -> str:
            t = normalize_text(x).replace("_", " ").strip()
            if t == "scm":
                return "SCM"
            if t == "sva":
                return "SVA_OUTROS"
            t2 = t.upper().replace(" ", "_")
            if t2 in AI_ALLOWED:
                return t2
            if "SVA" in t2:
                return "SVA_OUTROS"
            return "SVA_OUTROS"

        df_norm["categoria_aprovada"] = df_in[col_class].astype(str).map(map_cat)
        df_norm["created_at"] = now
        df_norm["source"] = "importado_simples"

    df_norm = df_norm.fillna("")
    df_norm = df_norm[df_norm["desc_norm"].astype(str).str.len() > 0]
    df_norm = df_norm[df_norm["categoria_aprovada"].isin(AI_ALLOWED)]
    if df_norm.empty:
        return False, "Após normalização, não sobrou nenhuma linha válida."

    df_current = training_load()
    out = pd.concat([df_current, df_norm], ignore_index=True)
    out = out.drop_duplicates(subset=["emit_cnpj", "desc_norm"], keep="last")
    out.to_csv(TRAINING_CSV, index=False, encoding="utf-8")
    training_load.clear()
    return True, f"Base importada: {len(df_norm)} linhas válidas."


# =========================
# Heurística (fallback) — melhorada p/ STREAMING/TV
# =========================
SCM_KEYWORDS = [
    "fibra", "banda larga", "internet", "link", "dedicado", "ftth",
    "scm", "wifi", "wi-fi", "acesso", "conectividade", "rede", "dados", "velocidade"
]

# streaming/TV (SVA_TV_STREAMING) — aqui está o ajuste que resolve seu caso
SVA_TV_KEYWORDS = [
    "streaming", "iptv", "tv", "televisao", "televisão", "conteudo", "conteúdo",
    "filme", "filmes", "series", "séries", "canais", "cine", "hbomax", "netflix",
    "prime video", "amazon", "disney", "globoplay", "paramount", "telecine"
]

SVA_EBOOK_KEYWORDS = ["ebook", "e-book", "livro digital", "biblioteca digital", "leitura"]
SVA_LOC_KEYWORDS = ["locacao", "locação", "comodato", "aluguel", "equipamento", "roteador", "onu", "cpe", "modem"]

# SVA genérico
SVA_GENERIC = ["antivirus", "backup", "email", "ip fixo", "suporte", "cloud", "voip", "telefonia", "sva"]

# Alguns sinais fracos em descrição que NÃO devem jogar pra SCM
NEGATIVE_FOR_SCM = ["streaming", "iptv", "tv", "filmes", "series", "ebook", "locacao", "comodato", "aluguel"]


def heuristic_category(desc: str, cclass: str = "") -> Tuple[str, float, str]:
    """
    Heurística conservadora e “anti-erro CFOP”:
    - CFOP NÃO entra aqui (porque CFOP é um dos principais erros)
    - Prioriza palavras-chave fortes (ex.: STREAMING => SVA_TV_STREAMING)
    - Só dá SCM alto quando descrição é claramente conectividade e NÃO contém sinais de SVA
    """
    d = normalize_text(desc)
    cc = (str(cclass or "").strip())

    if not d:
        return ("SVA_OUTROS", 0.50, "Descrição vazia")

    # 1) Forte: TV/Streaming (resolve seu exemplo)
    if any(k in d for k in SVA_TV_KEYWORDS):
        # se também tiver SCM, reduz confiança (ambíguo)
        if any(k in d for k in SCM_KEYWORDS):
            return ("SVA_TV_STREAMING", 0.75, "Streaming/TV + sinais de SCM (ambíguo)")
        return ("SVA_TV_STREAMING", 0.95, "Palavras-chave fortes de Streaming/TV")

    # 2) Forte: eBook
    if any(k in d for k in SVA_EBOOK_KEYWORDS):
        if any(k in d for k in SCM_KEYWORDS):
            return ("SVA_EBOOK", 0.75, "eBook + sinais de SCM (ambíguo)")
        return ("SVA_EBOOK", 0.95, "Palavras-chave eBook")

    # 3) Forte: locação
    if any(k in d for k in SVA_LOC_KEYWORDS):
        if any(k in d for k in SCM_KEYWORDS):
            return ("SVA_LOCACAO", 0.75, "Locação + sinais de SCM (ambíguo)")
        return ("SVA_LOCACAO", 0.93, "Palavras-chave locação")

    # 4) SVA genérico
    if any(k in d for k in SVA_GENERIC):
        if any(k in d for k in SCM_KEYWORDS):
            return ("SVA_OUTROS", 0.70, "SVA genérico + sinais de SCM (ambíguo)")
        return ("SVA_OUTROS", 0.88, "Palavras-chave SVA")

    # 5) SCM — somente se bem claro e sem “pistas de SVA”
    if any(k in d for k in SCM_KEYWORDS) and not any(k in d for k in NEGATIVE_FOR_SCM):
        return ("SCM", 0.92, "Palavras-chave SCM (sem pistas de SVA)")

    # 6) Sem evidência forte
    # Aqui a gente NÃO “chuta SCM”: vai pra SVA_OUTROS com baixa confiança p/ revisão
    if cc == ALERTA_CCLASS:
        return ("SVA_OUTROS", 0.60, "cClass 1100101 (alerta); manter revisão")
    return ("SVA_OUTROS", 0.60, "Sem evidência forte (revisão)")


# =========================
# OpenAI (Consolidado) — prompt melhorado (CFOP não é prova)
# =========================
AI_SYSTEM_CONSOL = """Você é especialista fiscal em NFCom (Modelo 62).
Classifique ITENS CONSOLIDADOS em UMA categoria:
- SCM
- SVA_EBOOK
- SVA_LOCACAO
- SVA_TV_STREAMING
- SVA_OUTROS

Você deve considerar de forma conjunta:
- DESCRIÇÃO (principal)
- cClass (pode estar errado por clientes, então é apenas um sinal auxiliar)
- CFOP (NÃO é prova de SCM; CFOP pode estar errado e é um erro comum em SVA)
- volume (ocorrências e total) como sinal de padrão

Regras de confiança (muito importante):
- Só dê alta confiança (>=0.90) quando a DESCRIÇÃO for claramente compatível com a categoria.
- Se cClass conflitar com descrição, reduza a confiança e explique o conflito.
- Se o caso for ambíguo, use SVA_OUTROS com baixa confiança e explique por que é ambíguo.

Retorne SOMENTE JSON válido no formato:
{"items":[{"id":"...","ia_sugestao":"SCM|SVA_EBOOK|SVA_LOCACAO|SVA_TV_STREAMING|SVA_OUTROS","ia_confianca":0.0-1.0,"ia_motivo":"..."}]}
"""


def get_openai_client():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def _strip_fences(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(t: str) -> Optional[str]:
    t = _strip_fences(t)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start : i + 1]
    return None


def _safe_json_loads(t: str) -> Optional[dict]:
    t = _strip_fences(t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    js = _extract_json_object(t)
    if not js:
        return None
    js = js.replace("\u201c", '"').replace("\u201d", '"')
    js = re.sub(r",\s*([}\]])", r"\1", js)
    try:
        obj = json.loads(js)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def ai_classify_consolidated(df_consol: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df_consol.copy()
    if df.empty:
        df["ia_sugestao"] = ""
        df["ia_confianca"] = 0.0
        df["ia_motivo"] = ""
        return df

    client = get_openai_client()
    if client is None:
        sug, conf, mot = [], [], []
        for _, r in df.iterrows():
            cat, c, why = heuristic_category(r.get("descricao_exemplo", ""), "")
            sug.append(cat)
            conf.append(float(c))
            mot.append(f"Sem OpenAI. Heurística: {why}")
        df["ia_sugestao"] = sug
        df["ia_confianca"] = conf
        df["ia_motivo"] = mot
        return df

    items = []
    for i, r in df.iterrows():
        items.append({
            "id": str(r.get("desc_norm", ""))[:120] or f"row_{i}",
            "descricao_exemplo": str(r.get("descricao_exemplo", ""))[:260],
            "cClass_distintos": str(r.get("cClass_distintos", ""))[:220],
            "CFOP_distintos": str(r.get("CFOP_distintos", ""))[:220],
            "qtd_ocorrencias": int(r.get("qtd_ocorrencias", 0) or 0),
            "total_vServ": float(r.get("total_vServ", 0.0) or 0.0),
            "categoria_atual": str(r.get("categoria_sugerida", ""))[:30],
        })

    out_rows = []
    for start in range(0, len(items), 30):
        chunk = items[start:start+30]
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": AI_SYSTEM_CONSOL},
                    {"role": "user", "content": json.dumps({"items": chunk}, ensure_ascii=False)}
                ],
                temperature=0.0,
                max_output_tokens=2200,
            )
            data = _safe_json_loads(resp.output_text or "")
        except Exception:
            data = None

        if not data or "items" not in data:
            for it in chunk:
                cat, c, why = heuristic_category(it.get("descricao_exemplo", ""), "")
                out_rows.append({
                    "id": it["id"],
                    "ia_sugestao": cat,
                    "ia_confianca": float(c),
                    "ia_motivo": f"Falha IA. Heurística: {why}"
                })
            continue

        got = data.get("items", [])
        by_id = {r.get("id"): r for r in got if isinstance(r, dict)}

        for it in chunk:
            r = by_id.get(it["id"], {})
            sug = (r.get("ia_sugestao") or "SVA_OUTROS").strip()
            if sug not in AI_ALLOWED:
                sug = "SVA_OUTROS"
            try:
                c = float(r.get("ia_confianca", 0.60) or 0.60)
            except Exception:
                c = 0.60
            c = max(0.0, min(1.0, c))
            motivo = (r.get("ia_motivo") or "").strip()[:260]
            out_rows.append({"id": it["id"], "ia_sugestao": sug, "ia_confianca": c, "ia_motivo": motivo})

    out_df = pd.DataFrame(out_rows)
    df["_id_join"] = df["desc_norm"].astype(str).str.slice(0, 120)
    out_df["_id_join"] = out_df["id"].astype(str).str.slice(0, 120)
    df = df.merge(out_df[["_id_join", "ia_sugestao", "ia_confianca", "ia_motivo"]], on="_id_join", how="left")
    df.drop(columns=["_id_join"], inplace=True)
    df["ia_sugestao"] = df["ia_sugestao"].fillna("")
    df["ia_confianca"] = df["ia_confianca"].fillna(0.0)
    df["ia_motivo"] = df["ia_motivo"].fillna("")
    return df


# =========================
# Itens NFCom
# =========================
def extract_items_nfcom(tree: etree._ElementTree, file_name: str) -> List[Dict[str, Any]]:
    root = tree.getroot()
    ns = get_ns(tree)
    dets = xp(root, ns, ".//n:det | .//det")
    items = []
    for idx, det in enumerate(dets, start=1):
        cclass = first_text(det, ns, "./n:prod/n:cClass | ./prod/cClass")
        xprod = first_text(det, ns, "./n:prod/n:xProd | ./prod/xProd")
        cfop = first_text(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")

        vitem = to_float(first_text(det, ns, "./n:prod/n:vItem | ./prod/vItem"))
        vprod = to_float(first_text(det, ns, "./n:prod/n:vProd | ./prod/vProd"))
        vdesc = to_float(first_text(det, ns, "./n:prod/n:vDesc | ./prod/vDesc"))
        vout = to_float(first_text(det, ns, "./n:prod/n:vOutro | ./prod/vOutro"))

        items.append({
            "arquivo": file_name,
            "item": idx,
            "cClass": str(cclass).strip(),
            "descricao": xprod,
            "CFOP": str(cfop).strip(),
            "vItem": float(vitem),
            "vProd": float(vprod),
            "vDesc": float(vdesc),
            "vOutros": float(vout),
            "vServ": float(vprod),
        })
    return items


# =========================
# Correção XML (SVA remove CFOP | SCM insere CFOP)
# =========================
def simulate_and_or_correct_xml_nfcom(
    tree: etree._ElementTree,
    df_dec: pd.DataFrame,
    corr_auto_threshold: float,
    corrigir_descontos: bool,
    apply_changes: bool,
    inserir_cfop_scm: bool,
    ajustar_cfop_pf_ncontrib: bool,
) -> Tuple[bytes, List[Dict[str, Any]], bool]:
    """
    Corrige em cópia do XML:
      - remove CFOP quando categoria SVA e confiança >= threshold
      - INSERE CFOP quando categoria SCM e confiança >= threshold e CFOP ausente (5307/6307)
      - (opcional) ajusta CFOP do SCM para PF/PJ não contribuinte (5307/6307) quando divergente
      - paliativo desconto (vProd=vItem se vProd < vItem)
    """
    root = tree.getroot()
    original_xml = etree.tostring(tree, encoding="utf-8", xml_declaration=True)

    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)

    decisions = {int(r["item"]): (str(r["categoria_fiscal_ia"]), float(r["confianca_ia"])) for _, r in df_dec.iterrows()}
    dets = xp(copy_root, ns, ".//n:det | .//det")

    changes: List[Dict[str, Any]] = []

    # parâmetros p/ cfop scm
    cfop_exp = expected_cfop_scm(tree)
    dest_pf_ncontrib = is_dest_pf_or_pj_nao_contrib(tree)

    for idx, det in enumerate(dets, start=1):
        cat, conf = decisions.get(idx, ("SVA_OUTROS", 0.0))
        cat = (cat or "SVA_OUTROS").strip()
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        # pega cfop nodes
        cfop_nodes = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
        cfop_text = ""
        if cfop_nodes:
            cfop_text = (cfop_nodes[0].text or "").strip()

        # 1) Remover CFOP do SVA (com confiança alta)
        if cat.startswith("SVA_") and conf >= corr_auto_threshold:
            if cfop_nodes:
                old_cfop = cfop_text
                changes.append({
                    "item": idx,
                    "acao": "REMOVER_CFOP_SVA",
                    "detalhe": f"Remover CFOP='{old_cfop}' (cat={cat}, conf={conf:.2f})"
                })
                if apply_changes:
                    for node in cfop_nodes:
                        parent = node.getparent()
                        if parent is not None:
                            parent.remove(node)

        # 2) Inserir CFOP para SCM aprovado (se estiver faltando)
        if inserir_cfop_scm and cat == "SCM" and conf >= corr_auto_threshold:
            if not cfop_text:
                changes.append({
                    "item": idx,
                    "acao": "INSERIR_CFOP_SCM",
                    "detalhe": f"Inserir CFOP='{cfop_exp}' (cat=SCM, conf={conf:.2f})"
                })
                if apply_changes:
                    prod_nodes = xp(det, ns, "./n:prod | ./prod")
                    if prod_nodes:
                        prod_node = prod_nodes[0]
                        if ns and "n" in ns:
                            cfop_elem = etree.SubElement(prod_node, f"{{{ns['n']}}}CFOP")
                        else:
                            cfop_elem = etree.SubElement(prod_node, "CFOP")
                        cfop_elem.text = cfop_exp

            # 3) Ajustar CFOP quando PF/PJ não contribuinte (opcional e seguro)
            # Só mexe se for SCM aprovado, e apenas se divergente do esperado
            if ajustar_cfop_pf_ncontrib and dest_pf_ncontrib:
                # recarrega cfop após possível inserção
                cfop_nodes2 = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
                cur = (cfop_nodes2[0].text or "").strip() if cfop_nodes2 else ""
                if cur and cur != cfop_exp:
                    changes.append({
                        "item": idx,
                        "acao": "AJUSTAR_CFOP_PF_NCONTRIB",
                        "detalhe": f"Ajustar CFOP '{cur}' -> '{cfop_exp}' (PF/PJ não contrib., conf={conf:.2f})"
                    })
                    if apply_changes and cfop_nodes2:
                        cfop_nodes2[0].text = cfop_exp

        # 4) Paliativo desconto
        if corrigir_descontos:
            vitem_nodes = xp(det, ns, "./n:prod/n:vItem | ./prod/vItem")
            vprod_nodes = xp(det, ns, "./n:prod/n:vProd | ./prod/vProd")
            if vitem_nodes and vprod_nodes:
                vi_text = (vitem_nodes[0].text or "").strip()
                vp_text = (vprod_nodes[0].text or "").strip()
                vi = to_float(vi_text)
                vp = to_float(vp_text)
                if vp < vi:
                    changes.append({"item": idx, "acao": "AJUSTAR_VPROD", "detalhe": f"vProd {vp_text} -> {vi_text} (paliativo desconto)"})
                    if apply_changes:
                        vprod_nodes[0].text = vi_text

    changed_flag = len(changes) > 0
    if apply_changes:
        new_xml = etree.tostring(new_tree, encoding="utf-8", xml_declaration=True)
        return new_xml, changes, changed_flag

    return original_xml, changes, changed_flag


# =========================
# Excel report
# =========================
def generate_excel_report(**dfs) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            if df is None:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
    output.seek(0)
    return output.read()


# =========================
# Processing ZIP streaming
# =========================
def decide_df_items(
    df_items: pd.DataFrame,
    client_cnpj: str,
    df_train: pd.DataFrame,
    suggest_threshold: float
) -> pd.DataFrame:
    """
    Decide categoria/confianca usando:
      - aprendizado (1.0)
      - heurística aprimorada (descrição + cClass)
    """
    df = df_items.copy()

    df["desc_norm"] = df["descricao"].fillna("").map(normalize_text)
    df["id_desc"] = df["desc_norm"].map(lambda x: re.sub(r"[^a-z0-9]+", "_", x)[:80])
    df["flag_cclass_1100101"] = df["cClass"].astype(str).str.strip().eq(ALERTA_CCLASS)

    train_map = training_lookup_map(df_train, client_cnpj or "")
    df["categoria_training"] = df["desc_norm"].map(lambda dn: train_map.get(dn, ""))
    df["confianca_training"] = df["categoria_training"].map(lambda c: 1.0 if c in AI_ALLOWED else 0.0)

    # heurística com cClass
    heur = df.apply(lambda r: heuristic_category(r.get("descricao", ""), r.get("cClass", "")), axis=1)
    df["categoria_heur"], df["confianca_heur"], df["motivo_heur"] = zip(*heur)

    def _final(r):
        if r["categoria_training"] in AI_ALLOWED:
            return pd.Series(["aprendizado", r["categoria_training"], 1.0, "Aprovado na base de aprendizado"])
        cat = r["categoria_heur"]
        conf = float(r["confianca_heur"])
        mot = r["motivo_heur"]

        # regra: se confiança estiver MUITO baixa, marca para revisão (mas mantém categoria)
        if conf < float(suggest_threshold):
            mot = f"{mot} | abaixo do limiar ({suggest_threshold:.2f}) — revisar"

        if cat not in AI_ALLOWED:
            cat = "SVA_OUTROS"
        return pd.Series(["heuristica", cat, conf, mot])

    df[["origem_ia", "categoria_fiscal_ia", "confianca_ia", "motivo_ia"]] = df.apply(_final, axis=1)
    return df


def process_zip_streaming(
    zip_bytes: bytes,
    cancel_keys: set,
    apply_changes: bool,
    corr_auto_threshold: float,
    corrigir_descontos: bool,
    inserir_cfop_scm: bool,
    ajustar_cfop_pf_ncontrib: bool,
    df_train: pd.DataFrame,
    suggest_threshold: float,
) -> dict:
    """
    Processa ZIP grande sem estourar RAM:
      - salva originais em /tmp
      - extrai itens
      - decide categoria por aprendizado/heurística
      - reabre originais e gera saídas no disco
      - guarda apenas paths + metadados
    """
    out_dir = Path(get_workspace_dir(), "files")
    out_dir.mkdir(parents=True, exist_ok=True)

    invalid = []
    canceled = []
    files_meta = []
    items_all = []

    client_cnpj = ""
    client_nome = ""
    month_ref = ""

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".xml"):
                continue

            logical_name = f"ZIP::{info.filename}"
            base_name = info.filename.replace("\\", "/").split("/")[-1]

            try:
                xml_bytes = zf.read(info)
            except Exception as e:
                invalid.append({"arquivo": logical_name, "erro": f"Falha ao ler do ZIP: {e}"})
                continue

            # descarta evento cancelamento
            is_evt, chave_evt = detect_cancelamento_event_bytes(xml_bytes)
            if is_evt:
                canceled.append({"arquivo_base": base_name, "chave": chave_evt, "status": "evento_cancelamento"})
                del xml_bytes
                continue

            # descarta cancelamento por palavras
            is_can, chave_can = detect_cancelamento_by_words(xml_bytes)
            if is_can:
                canceled.append({"arquivo_base": base_name, "chave": chave_can, "status": "cancelado_por_texto"})
                del xml_bytes
                continue

            # parse
            try:
                tree = parse_xml(xml_bytes)
            except Exception as e:
                invalid.append({"arquivo": logical_name, "erro": f"XML inválido: {e}"})
                del xml_bytes
                continue

            # modelo
            model = get_nf_model(tree)
            if model and model != "62":
                invalid.append({"arquivo": logical_name, "erro": f"Modelo {model} != 62 (ignorado)."})
                del xml_bytes
                del tree
                continue

            chave = extract_chave_acesso(tree)
            if cancel_keys and chave and chave in cancel_keys:
                canceled.append({"arquivo_base": base_name, "chave": chave, "status": "lista_canceladas"})
                del xml_bytes
                del tree
                continue

            if not client_cnpj:
                client_cnpj, client_nome = get_emitente(tree)
            if not month_ref:
                month_ref = get_competencia_mes(tree)

            # itens
            items_all.extend(extract_items_nfcom(tree, logical_name))

            # grava original no disco (para preview sob demanda)
            orig_path = out_dir / f"{base_name}.orig.xml"
            try:
                with open(orig_path, "wb") as f:
                    f.write(etree.tostring(tree, encoding="utf-8", xml_declaration=True))
            except Exception:
                orig_path = None

            files_meta.append({
                "arquivo": logical_name,
                "base_name": base_name,
                "chave": chave,
                "orig_path": str(orig_path) if orig_path else "",
                "out_path": "",
                "changed": False,
                "changes_count": 0,
            })

            del xml_bytes
            del tree
            gc.collect()

    df_items = pd.DataFrame(items_all) if items_all else pd.DataFrame()
    if df_items.empty:
        return {"empty": True, "invalid": invalid, "canceled": canceled}

    # decide classificação (leve, mais precisa para streaming etc.)
    df_items = decide_df_items(
        df_items=df_items,
        client_cnpj=client_cnpj,
        df_train=df_train,
        suggest_threshold=suggest_threshold,
    )

    # corrige XMLs (reabre do disco)
    changes_all = []
    meta_by_file = {m["arquivo"]: m for m in files_meta}

    for m in files_meta:
        logical_name = m["arquivo"]
        base_name = m["base_name"]
        orig_path = m["orig_path"]

        if not orig_path or not os.path.exists(orig_path):
            continue

        try:
            with open(orig_path, "rb") as f:
                orig_bytes = f.read()
            tree = parse_xml(orig_bytes)
        except Exception as e:
            invalid.append({"arquivo": logical_name, "erro": f"Falha ao reabrir original: {e}"})
            continue

        df_dec = df_items.loc[df_items["arquivo"] == logical_name, ["item", "categoria_fiscal_ia", "confianca_ia"]].copy()

        xml_out, changes, changed_flag = simulate_and_or_correct_xml_nfcom(
            tree=tree,
            df_dec=df_dec,
            corr_auto_threshold=corr_auto_threshold,
            corrigir_descontos=corrigir_descontos,
            apply_changes=apply_changes,
            inserir_cfop_scm=inserir_cfop_scm,
            ajustar_cfop_pf_ncontrib=ajustar_cfop_pf_ncontrib,
        )

        out_path = out_dir / f"{base_name}.out.xml"
        with open(out_path, "wb") as f:
            f.write(xml_out)

        meta_by_file[logical_name]["out_path"] = str(out_path)
        meta_by_file[logical_name]["changed"] = bool(changed_flag)
        meta_by_file[logical_name]["changes_count"] = int(len(changes))

        for c in changes:
            changes_all.append({
                "arquivo": logical_name,
                "base_name": base_name,
                "chave": m.get("chave", ""),
                "item": c.get("item"),
                "acao": c.get("acao"),
                "detalhe": c.get("detalhe"),
                "modo": "APLICADO" if apply_changes else "SUGERIDO",
            })

        del tree
        del xml_out
        gc.collect()

    df_files = pd.DataFrame(list(meta_by_file.values()))
    df_changes = pd.DataFrame(changes_all) if changes_all else pd.DataFrame(
        columns=["arquivo", "base_name", "chave", "item", "acao", "detalhe", "modo"]
    )

    return {
        "empty": False,
        "client_cnpj": client_cnpj,
        "client_nome": client_nome,
        "month_ref": month_ref,
        "df_items": df_items,
        "df_files": df_files,
        "df_changes": df_changes,
        "invalid": invalid,
        "canceled": canceled,
    }


def build_zip_on_disk(df_files: pd.DataFrame, mode: str) -> str:
    """
    mode: todos | somente_mudanca | somente_sem_mudanca
    Gera ZIP em disco, não em memória.
    """
    zip_path = ws_path(f"downloads/lote_{mode}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for _, r in df_files.iterrows():
            if mode == "somente_mudanca" and not bool(r.get("changed", False)):
                continue
            if mode == "somente_sem_mudanca" and bool(r.get("changed", False)):
                continue
            out_path = r.get("out_path", "")
            if out_path and os.path.exists(out_path):
                arcname = os.path.basename(out_path).replace(".out.xml", ".xml")
                z.write(out_path, arcname=arcname)
    return zip_path


# =========================
# Main
# =========================
def main():
    training_init()

    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH)
    with c2:
        st.markdown(f"## {APP_TITLE}")
        st.caption("Desenvolvido por Raul Martins — Contare (modo lote grande/ZIP)")

    # Sidebar
    st.sidebar.header("Configurações")

    apply_changes = st.sidebar.radio(
        "Correção dos XMLs",
        ["Aplicar correções no XML", "Apenas sugerir (não altera XML)"],
        index=0,
        key="apply_mode",
    ) == "Aplicar correções no XML"

    corrigir_descontos = st.sidebar.checkbox(
        "Paliativo descontos: vProd = vItem quando vProd < vItem",
        value=False,
        key="chk_desconto",
    )

    inserir_cfop_scm = st.sidebar.checkbox(
        "Inserir CFOP quando SCM for aprovado (se estiver faltando)",
        value=True,
        key="chk_ins_cfop_scm",
        help="Se o item for classificado como SCM com confiança alta e estiver sem CFOP, insere 5307/6307 conforme UF emit/dest."
    )

    ajustar_cfop_pf_ncontrib = st.sidebar.checkbox(
        "Ajustar CFOP do SCM para PF/PJ não contribuinte (5307/6307) quando divergente",
        value=False,
        key="chk_adj_pf",
        help="Só altera quando SCM aprovado e destinatário PF/PJ não contribuinte."
    )

    corr_auto_threshold = st.sidebar.slider(
        "Limiar para aplicar correções (confiança mínima)",
        0.50, 1.00, 0.95, 0.01,
        key="thr_apply",
        help="A remoção de CFOP do SVA e inserção de CFOP no SCM só ocorrem acima desse limiar."
    )

    suggest_threshold = st.sidebar.slider(
        "Limiar de revisão (abaixo disso marca como revisar)",
        0.50, 1.00, 0.85, 0.01,
        key="thr_review",
    )

    st.sidebar.markdown("---")
    enable_ai_consol = st.sidebar.checkbox("Ativar IA no CONSOLIDADO (OpenAI)", value=True, key="ai_consol")
    ai_model = st.sidebar.text_input("Modelo OpenAI", value="gpt-4o-mini", key="ai_model")
    ia_consol_min_conf = st.sidebar.slider(
        "Confiança mínima para pré-preencher categoria_aprovada",
        0.50, 1.00, 0.90, 0.01,
        key="ai_conf_min"
    )

    # Base aprendizado
    st.sidebar.markdown("---")
    st.sidebar.subheader("Base de aprendizado")
    up_train = st.sidebar.file_uploader("Importar base (CSV/XLSX)", type=["csv", "xlsx"], key="up_train")
    if up_train is not None:
        ok, msg = training_merge_uploaded(up_train)
        (st.sidebar.success(msg) if ok else st.sidebar.error(msg))

    df_train = training_load()
    st.sidebar.download_button(
        "Baixar base (CSV)",
        data=df_train.to_csv(index=False).encode("utf-8"),
        file_name="training_data.csv",
        key="dl_base",
    )

    # Lista canceladas (opcional)
    st.sidebar.markdown("---")
    cancel_file = st.sidebar.file_uploader("Chaves canceladas (CSV/TXT)", type=["csv", "txt"], key="up_cancel")
    cancel_keys = set()
    if cancel_file is not None:
        raw = cancel_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin1", errors="ignore")
        cancel_keys = set(re.findall(r"\d{44}", text))

    # Upload ZIP
    st.subheader("Upload (ZIP)")
    up_zip = st.file_uploader("Envie 1 arquivo ZIP com XMLs", type=["zip"], accept_multiple_files=False, key="zip")
    btn = st.button("Processar ZIP", type="primary", disabled=(up_zip is None), key="btn_proc")

    if btn and up_zip is not None:
        zip_bytes = up_zip.read()
        with st.status("Processando ZIP (streaming)...", expanded=True) as status:
            status.write("Lendo e filtrando XMLs (cancelados/ignorados)...")
            res = process_zip_streaming(
                zip_bytes=zip_bytes,
                cancel_keys=cancel_keys,
                apply_changes=apply_changes,
                corr_auto_threshold=corr_auto_threshold,
                corrigir_descontos=corrigir_descontos,
                inserir_cfop_scm=inserir_cfop_scm,
                ajustar_cfop_pf_ncontrib=ajustar_cfop_pf_ncontrib,
                df_train=df_train,
                suggest_threshold=suggest_threshold,
            )
            status.write("Finalizando estruturas de dados...")
            st.session_state["RESULTS"] = res
            # limpa zips anteriores (pra não confundir downloads)
            st.session_state.pop("ZIP_ALL", None)
            st.session_state.pop("ZIP_CHG", None)
            st.session_state.pop("ZIP_OK", None)
            status.update(label="Processamento concluído.", state="complete")
        del zip_bytes
        gc.collect()

    res = st.session_state.get("RESULTS")
    if not res:
        st.info("Envie o ZIP e clique em **Processar ZIP**.")
        return

    if res.get("empty"):
        st.warning("Nenhum XML NFCom ativo foi processado.")
        if res.get("invalid"):
            st.subheader("Ignorados")
            st.dataframe(pd.DataFrame(res["invalid"]), use_container_width=True)
        if res.get("canceled"):
            st.subheader("Cancelados/Descartados")
            st.dataframe(pd.DataFrame(res["canceled"]), use_container_width=True)
        return

    df_items: pd.DataFrame = res["df_items"]
    df_files: pd.DataFrame = res["df_files"]
    df_changes: pd.DataFrame = res["df_changes"]
    invalid = res.get("invalid", [])
    canceled = res.get("canceled", [])
    client_cnpj = res.get("client_cnpj", "")
    client_nome = res.get("client_nome", "")
    month_ref = res.get("month_ref", "")

    # Alertas 1100101
    if "flag_cclass_1100101" in df_items.columns and df_items["flag_cclass_1100101"].any():
        st.sidebar.warning(ALERTA_TEXTO)
        with st.sidebar.expander("Itens cClass 1100101"):
            st.sidebar.dataframe(
                df_items[df_items["flag_cclass_1100101"]][["arquivo", "item", "descricao", "cClass", "CFOP", "vServ"]].head(300),
                use_container_width=True,
                height=260,
            )
        st.warning(ALERTA_TEXTO, icon="⚠️")

    # Dashboard leve
    st.subheader("Resumo do lote")
    total_docs = len(df_files)
    total_changed = int(df_files["changed"].sum()) if "changed" in df_files.columns else 0
    total_itens = len(df_items)
    total_v = float(df_items["vServ"].sum()) if "vServ" in df_items.columns else 0.0
    a, b, c, d = st.columns(4)
    a.metric("Docs ativos", total_docs)
    b.metric("Docs com mudança", total_changed)
    c.metric("Itens", total_itens)
    d.metric("Total vServ", num_to_br(total_v))

    # Resumo por categoria
    df_cat = df_items.groupby("categoria_fiscal_ia", as_index=False).agg(
        qtd_itens=("item", "count"),
        total_vServ=("vServ", "sum")
    )
    st.dataframe(df_cat, use_container_width=True)
    st.bar_chart(df_cat.set_index("categoria_fiscal_ia")["total_vServ"])

    # Consolidação (todos / revisar)
    st.subheader("Consolidação por item (para validação manual)")
    modo_consol = st.radio(
        "Consolidar a partir de:",
        ["Todos os itens do lote (recomendado)", "Somente itens a revisar (baixa confiança)"],
        index=0,
        horizontal=True,
        key="modo_consol",
    )
    if modo_consol.startswith("Somente"):
        df_base = df_items[df_items["confianca_ia"].astype(float) < corr_auto_threshold].copy()
    else:
        df_base = df_items.copy()

    def _join_unique(series, max_items=10):
        vals = [str(v).strip() for v in series.dropna().unique().tolist() if str(v).strip()]
        return " | ".join(vals[:max_items]) if vals else ""

    df_consol = (
        df_base.groupby("desc_norm", as_index=False)
        .agg(
            descricao_exemplo=("descricao", "first"),
            categoria_sugerida=("categoria_fiscal_ia", "first"),
            confianca_min=("confianca_ia", "min"),
            qtd_ocorrencias=("desc_norm", "size"),
            total_vServ=("vServ", "sum"),
            cClass_distintos=("cClass", _join_unique),
            CFOP_distintos=("CFOP", _join_unique),
        )
    )

    # IA no consolidado (botão)
    if enable_ai_consol:
        if st.button("Rodar IA no consolidado (considera descrição + cClass + volume)", key="btn_ai_consol"):
            with st.spinner("IA no consolidado..."):
                df_ai = ai_classify_consolidated(df_consol, model=ai_model)
                st.session_state["AI_CONSOL"] = df_ai

    df_ai_last = st.session_state.get("AI_CONSOL")
    if isinstance(df_ai_last, pd.DataFrame) and not df_ai_last.empty:
        df_ai_last = df_ai_last[["desc_norm", "ia_sugestao", "ia_confianca", "ia_motivo"]].copy()
        df_consol = df_consol.merge(df_ai_last, on="desc_norm", how="left")
    else:
        df_consol["ia_sugestao"] = ""
        df_consol["ia_confianca"] = 0.0
        df_consol["ia_motivo"] = ""

    # categoria aprovada (pré-preenche com IA quando conf alta)
    def _default_aprov(r):
        try:
            conf = float(r.get("ia_confianca", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        sug = (r.get("ia_sugestao", "") or "").strip()
        if sug in AI_ALLOWED and conf >= ia_consol_min_conf:
            return sug
        return r.get("categoria_sugerida", "SVA_OUTROS")

    df_consol["categoria_aprovada"] = df_consol.apply(_default_aprov, axis=1)

    st.caption("Edite **categoria_aprovada** (lista consolidada) e aplique em massa. Isso salva no aprendizado e aplica no lote.")
    edited = st.data_editor(
        df_consol[
            ["descricao_exemplo", "cClass_distintos", "CFOP_distintos", "qtd_ocorrencias", "total_vServ",
             "categoria_sugerida", "ia_sugestao", "ia_confianca", "categoria_aprovada"]
        ],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "categoria_aprovada": st.column_config.SelectboxColumn("categoria_aprovada", options=CATEGORIES, required=True),
        },
        key="editor_consol",
    )

    if st.button("Aplicar aprovações (massivo) + salvar aprendizado", key="btn_apply_mass"):
        df_apply = df_consol[["desc_norm"]].copy()
        df_apply["categoria_aprovada"] = edited["categoria_aprovada"].values
        map_aprov = dict(zip(df_apply["desc_norm"], df_apply["categoria_aprovada"]))

        df_items2 = df_items.copy()
        mask = df_items2["desc_norm"].isin(map_aprov.keys())
        df_items2.loc[mask, "categoria_fiscal_ia"] = df_items2.loc[mask, "desc_norm"].map(map_aprov)
        df_items2.loc[mask, "confianca_ia"] = 1.0
        df_items2.loc[mask, "motivo_ia"] = "Aprovado manualmente (consolidado)"
        df_items2.loc[mask, "origem_ia"] = "aprendizado"

        # salva aprendizado
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for dn, cat in map_aprov.items():
            if dn and cat in AI_ALLOWED:
                exemplo = df_items2.loc[df_items2["desc_norm"] == dn, "descricao"].astype(str).head(1).tolist()
                rows.append({
                    "emit_cnpj": client_cnpj or "",
                    "desc_norm": dn,
                    "descricao_exemplo": (exemplo[0] if exemplo else "")[:250],
                    "categoria_aprovada": cat,
                    "created_at": now,
                    "source": "aprovacao_consolidada",
                })
        if rows:
            training_append(rows)

        res["df_items"] = df_items2
        st.session_state["RESULTS"] = res
        st.success("Aprovações aplicadas ao dataframe. Agora clique em **Regerar saídas** para refletir nos XMLs.")
        st.rerun()

    # Regerar saídas
    st.markdown("---")
    st.subheader("Regerar saídas (corrigidos/sugeridos) com base na classificação atual")
    st.caption("Reabre originais do disco e regrava os .out.xml (sem RAM alta).")
    if st.button("Regerar saídas agora", key="btn_regerar"):
        out_dir = Path(get_workspace_dir(), "files")
        changes_all = []
        meta = df_files.to_dict("records")
        for m in meta:
            logical_name = m["arquivo"]
            base_name = m["base_name"]
            orig_path = m.get("orig_path", "")
            if not orig_path or not os.path.exists(orig_path):
                continue
            try:
                with open(orig_path, "rb") as f:
                    orig_bytes = f.read()
                tree = parse_xml(orig_bytes)
            except Exception:
                continue

            df_dec = res["df_items"].loc[res["df_items"]["arquivo"] == logical_name, ["item", "categoria_fiscal_ia", "confianca_ia"]].copy()

            xml_out, changes, changed_flag = simulate_and_or_correct_xml_nfcom(
                tree=tree,
                df_dec=df_dec,
                corr_auto_threshold=corr_auto_threshold,
                corrigir_descontos=corrigir_descontos,
                apply_changes=apply_changes,
                inserir_cfop_scm=inserir_cfop_scm,
                ajustar_cfop_pf_ncontrib=ajustar_cfop_pf_ncontrib,
            )

            out_path = out_dir / f"{base_name}.out.xml"
            with open(out_path, "wb") as f:
                f.write(xml_out)

            m["out_path"] = str(out_path)
            m["changed"] = bool(changed_flag)
            m["changes_count"] = int(len(changes))

            for c in changes:
                changes_all.append({
                    "arquivo": logical_name,
                    "base_name": base_name,
                    "chave": m.get("chave", ""),
                    "item": c.get("item"),
                    "acao": c.get("acao"),
                    "detalhe": c.get("detalhe"),
                    "modo": "APLICADO" if apply_changes else "SUGERIDO",
                })

            del tree
            del xml_out
            gc.collect()

        res["df_files"] = pd.DataFrame(meta)
        res["df_changes"] = pd.DataFrame(changes_all) if changes_all else pd.DataFrame(columns=["arquivo","base_name","chave","item","acao","detalhe","modo"])
        st.session_state["RESULTS"] = res
        # limpa zips anteriores
        st.session_state.pop("ZIP_ALL", None)
        st.session_state.pop("ZIP_CHG", None)
        st.session_state.pop("ZIP_OK", None)
        st.success("Saídas regeneradas.")
        st.rerun()

    # Atualiza refs
    res = st.session_state.get("RESULTS", res)
    df_items = res["df_items"]
    df_files = res["df_files"]
    df_changes = res["df_changes"]

    # Preview sob demanda
    st.markdown("---")
    st.subheader("Pré-visualização sob demanda")
    sel = st.selectbox("Selecione um arquivo", df_files["arquivo"].tolist(), key="sel_preview")
    row = df_files[df_files["arquivo"] == sel].iloc[0]

    st.write(
        f"**Arquivo:** `{row['base_name']}` | **Chave:** `{row.get('chave','')}` | "
        f"**Mudanças:** {int(row.get('changes_count',0))} | **Changed:** {bool(row.get('changed', False))}"
    )
    df_c = df_changes[df_changes["arquivo"] == sel].copy() if not df_changes.empty else pd.DataFrame()
    if not df_c.empty:
        st.dataframe(df_c[["item", "acao", "detalhe", "modo"]], use_container_width=True, height=240)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original (disco)")
        st.text_area("orig", value=read_file_text(row.get("orig_path","")), height=380, key="ta_orig")
    with col2:
        st.caption("Saída (disco)")
        st.text_area("out", value=read_file_text(row.get("out_path","")), height=380, key="ta_out")

    # Downloads ZIP em disco
    st.markdown("---")
    st.subheader("Downloads (ZIP gerado em disco)")
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Preparar ZIP – Todos", key="prep_all"):
            st.session_state["ZIP_ALL"] = build_zip_on_disk(df_files, "todos")
        zp = st.session_state.get("ZIP_ALL")
        if zp and os.path.exists(zp):
            with open(zp, "rb") as f:
                st.download_button("Baixar ZIP – Todos", data=f, file_name="xml_ativos.zip", mime="application/zip", key="dl_all")

    with cB:
        if st.button("Preparar ZIP – Somente com mudança", key="prep_chg"):
            st.session_state["ZIP_CHG"] = build_zip_on_disk(df_files, "somente_mudanca")
        zp = st.session_state.get("ZIP_CHG")
        if zp and os.path.exists(zp):
            with open(zp, "rb") as f:
                st.download_button("Baixar ZIP – Mudanças", data=f, file_name="xml_mudancas.zip", mime="application/zip", key="dl_chg")

    with cC:
        if st.button("Preparar ZIP – Sem mudança", key="prep_ok"):
            st.session_state["ZIP_OK"] = build_zip_on_disk(df_files, "somente_sem_mudanca")
        zp = st.session_state.get("ZIP_OK")
        if zp and os.path.exists(zp):
            with open(zp, "rb") as f:
                st.download_button("Baixar ZIP – Sem mudança", data=f, file_name="xml_sem_mudanca.zip", mime="application/zip", key="dl_ok")

    # Excel
    st.markdown("---")
    st.subheader("Relatório Excel")
    df_alertas = pd.DataFrame()
    if "flag_cclass_1100101" in df_items.columns and df_items["flag_cclass_1100101"].any():
        df_alertas = df_items[df_items["flag_cclass_1100101"]].copy()

    df_consol_excel = df_consol.copy()
    df_status = df_files[["arquivo","base_name","chave","changed","changes_count","orig_path","out_path"]].copy()

    df_ign = pd.DataFrame(invalid) if invalid else pd.DataFrame(columns=["arquivo","erro"])
    df_can = pd.DataFrame(canceled) if canceled else pd.DataFrame(columns=["arquivo_base","chave","status"])

    try:
        excel_bytes = generate_excel_report(
            Resumo=pd.DataFrame([{
                "cliente_cnpj": client_cnpj,
                "cliente_nome": client_nome,
                "competencia": month_ref,
                "modo": "APLICADO" if apply_changes else "SUGERIDO",
                "docs": len(df_files),
                "itens": len(df_items),
                "total_vServ": float(df_items["vServ"].sum()) if "vServ" in df_items.columns else 0.0,
                "thr_apply": corr_auto_threshold,
            }]),
            Detalhamento_Itens=df_items.drop(columns=["id_desc"], errors="ignore"),
            Resumo_Categoria=df_cat,
            Consolidado=df_consol_excel,
            Alertas_1100101=df_alertas if not df_alertas.empty else None,
            Status_Arquivos=df_status,
            Log_Mudancas=df_changes if not df_changes.empty else None,
            Cancelados=df_can if not df_can.empty else None,
            Ignorados=df_ign if not df_ign.empty else None,
            Base_Aprendizado=training_load(),
        )
        st.download_button(
            "Baixar Excel – Relatório completo",
            data=excel_bytes,
            file_name="relatorio_nfcom_lote_grande.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel",
        )
    except Exception as e:
        st.error(f"Falha ao gerar Excel (openpyxl ausente?). Erro: {e}")

    st.markdown("<hr><p style='text-align:center;font-size:12px;'>Desenvolvido por Raul Martins – Contare</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

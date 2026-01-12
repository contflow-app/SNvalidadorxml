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
APP_TITLE = "Validador NFCom 62 — Lote Grande (200MB) + IA (Consolidado)"
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


def get_ufs(tree: etree._ElementTree) -> Tuple[str, str]:
    root = tree.getroot()
    ns = get_ns(tree)
    uf_emit = first_text(root, ns, ".//n:emit/n:enderEmit/n:UF | .//emit/enderEmit/UF").strip()
    uf_dest = first_text(root, ns, ".//n:dest/n:enderDest/n:UF | .//dest/enderDest/UF").strip()
    return uf_emit, uf_dest


# =========================
# Cancelamento detection
# =========================
def contains_cancel_words(text: str) -> bool:
    t = normalize_text(text or "")
    return ("cancelamento" in t) or ("cancelad" in t)


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
        col_cnpj = pick_col(df_in, ["cnpj", "cpf", "emit_cnpj"])
        col_desc = pick_col(df_in, ["descricao", "descrição", "xprod", "produto", "servico", "serviço"])
        col_class = pick_col(df_in, ["classificacao_validada", "classificacao", "classificação", "categoria_fiscal_ia", "categoria", "categoria_aprovada"])
        if not col_desc or not col_class:
            return False, f"Layout não reconhecido. Aceito:\n\nInterno: emit_cnpj, desc_norm, categoria_aprovada\nSimples: CNPJ, descricao, CLASSIFICACAO VALIDADA"

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
# Heurística (fallback) — AJUSTADA
# =========================
# IMPORTANTE: removido "plano" de SCM_KEYWORDS (era o principal motivo de "STREAMING - PLANO" virar SCM)
SCM_KEYWORDS = [
    "fibra", "banda larga", "internet", "link", "dedicado", "ftth", "scm",
    "wifi", "acesso", "conectividade", "rede", "dados", "ip", "pppoe"
]

SVA_EBOOK_KEYWORDS = ["ebook", "e-book", "livro digital", "biblioteca digital", "leitura"]
SVA_LOC_KEYWORDS = ["locacao", "locação", "comodato", "aluguel", "equipamento", "roteador", "onu", "cpe", "modem"]
SVA_TV_KEYWORDS = [
    "tv", "iptv", "streaming", "filme", "filmes", "serie", "series", "cinema", "cine",
    "watch", "video", "vídeo", "conteudo", "conteúdo", "televisao", "televisão", "canal"
]
SVA_GENERIC = ["antivirus", "backup", "email", "ip fixo", "suporte", "cloud", "voip", "telefonia", "sva"]


def heuristic_category(desc: str, cclass: str = "", cfop: str = "") -> Tuple[str, float, str]:
    """
    Heurística mais segura:
      - PRIORIDADE para SVA_TV/LOC/EBOOK se houver evidência (mesmo que apareça palavra genérica).
      - SCM só quando houver evidência real de conectividade (fibra/internet/link/etc).
      - Considera cClass como sinal adicional (ex.: 0600601 + streaming => SVA_TV_STREAMING com alta confiança).
      - CFOP NÃO é sinal de correção (muitos clientes erram). No máximo reduz confiança se conflitar.
    """
    d = normalize_text(desc)
    c = (cclass or "").strip()
    f = (cfop or "").strip()

    if not d:
        return ("SVA_OUTROS", 0.50, "Descrição vazia")

    # 1) SVA específicos primeiro (evita STREAMING virar SCM por termos genéricos)
    if any(k in d for k in SVA_EBOOK_KEYWORDS):
        conf = 0.95
        motivo = "Palavras-chave eBook"
        return ("SVA_EBOOK", conf, motivo)

    if any(k in d for k in SVA_LOC_KEYWORDS):
        conf = 0.93
        motivo = "Palavras-chave locação/equipamento"
        return ("SVA_LOCACAO", conf, motivo)

    if any(k in d for k in SVA_TV_KEYWORDS):
        # boost por cClass conhecido no teu caso
        if c == "0600601":
            return ("SVA_TV_STREAMING", 0.97, "Streaming/TV + cClass 0600601 (forte evidência SVA)")
        return ("SVA_TV_STREAMING", 0.94, "Palavras-chave TV/Streaming")

    # 2) SVA genérico
    if any(k in d for k in SVA_GENERIC):
        return ("SVA_OUTROS", 0.88, "Palavras-chave SVA")

    # 3) SCM depois, somente com evidência real
    if any(k in d for k in SCM_KEYWORDS):
        # se contém muitos sinais de SVA, não deixa virar SCM (ex.: "plano streaming" etc)
        if any(k in d for k in (SVA_TV_KEYWORDS + SVA_LOC_KEYWORDS + SVA_EBOOK_KEYWORDS)):
            return ("SVA_OUTROS", 0.70, "Texto contém sinais mistos (SVA vs SCM). Mantendo SVA_OUTROS para revisão.")
        return ("SCM", 0.95, "Palavras-chave SCM (conectividade)")

    return ("SVA_OUTROS", 0.60, "Sem evidência forte (vai para revisão)")


# =========================
# OpenAI (Consolidado)
# =========================
AI_SYSTEM_CONSOL = """Você é especialista fiscal em NFCom (Modelo 62).
Classifique ITENS CONSOLIDADOS em UMA categoria:
- SCM
- SVA_EBOOK
- SVA_LOCACAO
- SVA_TV_STREAMING
- SVA_OUTROS

Considere juntos:
- DESCRIÇÃO
- cClass (sinal fiscal forte; divergência com descrição reduz confiança)
- CFOP (apenas SCM deve ter CFOP; SVA idealmente sem CFOP)
- volume (ocorrências e total)

Regras:
- Só dê alta confiança (>=0.90) quando descrição + cClass forem coerentes.
- Se ambíguo, use SVA_OUTROS com baixa confiança e explique.

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
        # fallback heurística
        sug, conf, mot = [], [], []
        for _, r in df.iterrows():
            cat, c, why = heuristic_category(r.get("descricao_exemplo", ""), "", "")
            sug.append(cat); conf.append(float(c)); mot.append(f"Sem OpenAI. Heurística: {why}")
        df["ia_sugestao"] = sug
        df["ia_confianca"] = conf
        df["ia_motivo"] = mot
        return df

    items = []
    for i, r in df.iterrows():
        items.append({
            "id": str(r.get("desc_norm", ""))[:120] or f"row_{i}",
            "descricao_exemplo": str(r.get("descricao_exemplo", ""))[:240],
            "cClass_distintos": str(r.get("cClass_distintos", ""))[:220],
            "CFOP_distintos": str(r.get("CFOP_distintos", ""))[:220],
            "qtd_ocorrencias": int(r.get("qtd_ocorrencias", 0) or 0),
            "total_vServ": float(r.get("total_vServ", 0.0) or 0.0),
            "categoria_atual": str(r.get("categoria_sugerida", ""))[:30],
        })

    out_rows = []
    for start in range(0, len(items), 35):
        chunk = items[start:start+35]
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": AI_SYSTEM_CONSOL},
                {"role": "user", "content": json.dumps({"items": chunk}, ensure_ascii=False)}
            ],
            temperature=0.0,
            max_output_tokens=2000,
        )
        data = _safe_json_loads(resp.output_text or "")
        if not data or "items" not in data:
            for it in chunk:
                cat, c, why = heuristic_category(it.get("descricao_exemplo", ""), "", "")
                out_rows.append({"id": it["id"], "ia_sugestao": cat, "ia_confianca": float(c), "ia_motivo": f"Falha IA. Heurística: {why}"})
            continue

        got = data.get("items", [])
        by_id = {r.get("id"): r for r in got if isinstance(r, dict)}

        for it in chunk:
            r = by_id.get(it["id"], {})
            sug = (r.get("ia_sugestao") or "SVA_OUTROS").strip()
            if sug not in AI_ALLOWED:
                sug = "SVA_OUTROS"
            try:
                c = float(r.get("ia_confianca", 0.6) or 0.6)
            except Exception:
                c = 0.6
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
            "cClass": cclass,
            "descricao": xprod,
            "CFOP": cfop,
            "vItem": float(vitem),
            "vProd": float(vprod),
            "vDesc": float(vdesc),
            "vOutros": float(vout),
            "vServ": float(vprod),
        })
    return items


def simulate_and_or_correct_xml_nfcom(
    tree: etree._ElementTree,
    df_dec: pd.DataFrame,
    corr_auto_threshold: float,
    corrigir_descontos: bool,
    apply_changes: bool,
    inserir_cfop_scm: bool,
    cfop_intra: str,
    cfop_inter: str,
) -> Tuple[bytes, List[Dict[str, Any]], bool]:
    """
    Corrige em cópia do XML:
      - remove CFOP quando categoria SVA e confiança >= threshold
      - INSERE CFOP quando categoria SCM e confiança >= threshold e não existe CFOP
      - paliativo desconto (vProd=vItem se vProd < vItem)
    """
    root = tree.getroot()
    original_xml = etree.tostring(tree, encoding="utf-8", xml_declaration=True)

    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)

    uf_emit, uf_dest = get_ufs(tree)
    expected_cfop = ""
    if uf_emit and uf_dest:
        expected_cfop = (cfop_intra if uf_emit == uf_dest else cfop_inter).strip()

    # item -> (cat, conf)
    decisions = {
        int(r["item"]): (str(r["categoria_fiscal_ia"]), float(r["confianca_ia"]))
        for _, r in df_dec.iterrows()
    }
    dets = xp(copy_root, ns, ".//n:det | .//det")

    changes: List[Dict[str, Any]] = []

    for idx, det in enumerate(dets, start=1):
        cat, conf = decisions.get(idx, ("SVA_OUTROS", 0.0))

        # remover CFOP do SVA (conforme regra do escritório)
        if cat.startswith("SVA_") and conf >= corr_auto_threshold:
            cfop_nodes = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
            if cfop_nodes:
                old_cfop = (cfop_nodes[0].text or "").strip()
                changes.append({"item": idx, "acao": "REMOVER_CFOP_SVA", "detalhe": f"Remover CFOP='{old_cfop}' (cat={cat}, conf={conf:.2f})"})
                if apply_changes:
                    for node in cfop_nodes:
                        parent = node.getparent()
                        if parent is not None:
                            parent.remove(node)

        # inserir CFOP quando SCM aprovado e está vazio
        if inserir_cfop_scm and cat == "SCM" and conf >= corr_auto_threshold and expected_cfop:
            cfop_nodes = xp(det, ns, "./n:prod/n:CFOP | ./prod/CFOP")
            cur = ""
            if cfop_nodes and isinstance(cfop_nodes[0], etree._Element):
                cur = (cfop_nodes[0].text or "").strip()

            if not cur:
               

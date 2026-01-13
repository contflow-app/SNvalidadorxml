import io
import re
import zipfile
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import yaml
from lxml import etree

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

LOGO_PATH = "Logo-Contare-ISP-1.png"

st.set_page_config(page_title="Validador NFCom 62 - Contare", layout="wide")


# ====================================================
# Namespace NFCom
# ====================================================

def get_ns(tree: etree._ElementTree) -> Dict[str, str]:
    root = tree.getroot()
    default_ns = root.nsmap.get(None)
    return {"n": default_ns} if default_ns else {}


# ====================================================
# Leitura de arquivos de configuração
# ====================================================

@st.cache_data
def load_rules(path: str = "rules.yaml") -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []
    except FileNotFoundError:
        st.warning(f"Arquivo '{path}' não encontrado.")
        return []


@st.cache_data
def load_cclass_config(path: str = "cclass_config.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        st.warning(f"Arquivo '{path}' não encontrado.")
        return {}


# ====================================================
# Helpers numéricos
# ====================================================

def to_float(value: str) -> float:
    try:
        if value is None:
            return 0.0
        return float(str(value).replace(",", "."))
    except Exception:
        return 0.0


def num_to_br(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        x = float(value)
        s = f"{x:,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(value)


# ====================================================
# Palavras-chave SCM / SVA e classificação inteligente
# ====================================================

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    trans = str.maketrans({
        "á": "a", "ã": "a", "â": "a", "à": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "õ": "o", "ô": "o",
        "ú": "u",
        "ç": "c",
    })
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ⚠️ Ajuste importante:
# - Removi "plano" da lista SCM (isso puxava STREAMING para SCM).
# - Adicionei STREAMING/TV/IPTV/etc como SVA forte.
SCM_KEYWORDS_STRONG = [
    "fibra", "fibra optica", "fibra óptica",
    "banda larga", "internet", "link", "link dedicado",
    "ftth", "velocidade", "scm",
    "dados", "conexao", "conexão",
    "wifi", "wi-fi",
    "provedor", "acesso", "link ip", "link rede",
    "conectividade", "rede",
]

# SVA forte (quando aparecer, deve remover CFOP mesmo se cClass estiver “errado”)
SVA_STREAMING_TV_STRONG = [
    "streaming", "tv", "iptv", "televisao", "televisão", "series", "séries", "filmes", "cine",
    "watch", "prime", "netflix", "disney", "hbo", "globoplay", "paramount", "starz"
]

SVA_EBOOK_STRONG = ["ebook", "e-book", "livro digital", "biblioteca digital", "leitura"]
SVA_LOCACAO_STRONG = ["locacao", "locação", "comodato", "aluguel", "equipamento", "roteador", "onu", "cpe"]
SVA_GENERIC = [
    "antivirus", "anti-virus", "anti vírus", "antivírus",
    "e-mail", "email", "correio eletronico", "correio eletrônico",
    "suporte premium", "ip fixo", "voip", "telefonia",
    "servico adicional", "serviço adicional",
    "servicos adicionais", "serviços adicionais",
    "backup", "cloud",
    "sva"
]


def detect_desc_category(desc_norm: str) -> Tuple[str, float, str]:
    """
    Classificação por descrição (forte) para corrigir casos como:
    cClass SCM mas descrição é STREAMING/TV → deve ser SVA e remover CFOP.
    Retorna (classe, confiança, motivo):
      - SCM / SVA / INDEFINIDO
    """
    d = desc_norm or ""
    if not d:
        return ("INDEFINIDO", 0.0, "Descrição vazia")

    # SVA fortíssimo (streaming/tv)
    if any(k in d for k in SVA_STREAMING_TV_STRONG):
        return ("SVA", 0.98, "Descrição indica Streaming/TV (SVA forte)")

    if any(k in d for k in SVA_EBOOK_STRONG):
        return ("SVA", 0.97, "Descrição indica eBook (SVA forte)")

    if any(k in d for k in SVA_LOCACAO_STRONG):
        return ("SVA", 0.96, "Descrição indica locação/comodato (SVA forte)")

    # SCM forte
    if any(k in d for k in SCM_KEYWORDS_STRONG) and not any(k in d for k in SVA_GENERIC):
        return ("SCM", 0.95, "Descrição indica SCM (SCM forte)")

    # SVA genérico
    if any(k in d for k in SVA_GENERIC) and not any(k in d for k in SCM_KEYWORDS_STRONG):
        return ("SVA", 0.88, "Descrição indica SVA (genérico)")

    return ("INDEFINIDO", 0.60, "Sem evidência forte na descrição")


def classify_item_scm_sva(xprod: str, cclass: str, cclass_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retorna:
      - class_por_descricao: SCM / SVA / INDEFINIDO
      - conf_desc: 0-1
      - class_por_cclass: SCM / SVA / INDEFINIDO
      - class_final_sugerida: SCM / SVA / INDEFINIDO
      - sugestao_forte: True/False
      - motivo
    """
    desc_norm = normalize_text(xprod or "")
    cclass = (cclass or "").strip()

    sva_cclasses = set(cclass_cfg.get("sva_cclasses", []) or [])
    scm_cclasses = set(cclass_cfg.get("scm_cclasses", []) or [])

    # 1) Por descrição (agora com SVA Streaming forte)
    class_desc, conf_desc, motivo_desc = detect_desc_category(desc_norm)

    # 2) Por cClass
    if cclass in scm_cclasses:
        class_cclass = "SCM"
    elif cclass in sva_cclasses:
        class_cclass = "SVA"
    else:
        class_cclass = "INDEFINIDO"

    # 3) Final (conservador, mas com exceção: SVA forte por descrição)
    motivo = []
    sugestao_forte = False

    # EXCEÇÃO PRINCIPAL: se descrição for SVA forte (>=0.96), prevalece (remove CFOP)
    if class_desc == "SVA" and conf_desc >= 0.96:
        class_final = "SVA"
        sugestao_forte = True
        motivo.append(motivo_desc + "; prevalece sobre cClass/CFOP para evitar importar SCM errado.")
    else:
        # comportamento conservador anterior: só “forte” quando conflito direto com mapeamento
        if class_desc == "SCM" and class_cclass == "SVA":
            class_final = "SCM"
            sugestao_forte = True
            motivo.append("Descrição SCM forte; cClass mapeado como SVA.")
        elif class_desc == "SVA" and class_cclass == "SCM":
            class_final = "SVA"
            sugestao_forte = True
            motivo.append("Descrição SVA; cClass mapeado como SCM.")
        else:
            if class_cclass in ("SCM", "SVA"):
                class_final = class_cclass
                motivo.append(f"Classificação mantida pelo cClass ({class_cclass}), sem evidência forte de erro.")
            elif class_desc in ("SCM", "SVA"):
                class_final = class_desc
                motivo.append(f"{motivo_desc}. Sem mapeamento de cClass; sugestão fraca.")
            else:
                class_final = "INDEFINIDO"
                motivo.append("Sem evidência clara (descrição/cClass).")

    return {
        "class_por_descricao": class_desc,
        "conf_desc": float(conf_desc),
        "class_por_cclass": class_cclass,
        "class_attach": class_desc,
        "class_final_sugerida": class_final,
        "motivo": " ".join(motivo),
        "sugestao_forte": bool(sugestao_forte),
    }


# ====================================================
# XML helpers
# ====================================================

def parse_xml(file_bytes: bytes) -> etree._ElementTree:
    try:
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        return etree.parse(io.BytesIO(file_bytes), parser)
    except Exception as e:
        raise ValueError(f"XML inválido: {e}")


def get_xpath_nodes(tree: etree._ElementTree, xpath: str):
    ns = get_ns(tree)
    root = tree.getroot()
    if ns:
        return root.xpath(xpath, namespaces=ns)
    return root.xpath(xpath)


def get_nf_model(tree: etree._ElementTree) -> str:
    ns = get_ns(tree)
    root = tree.getroot()
    if ns:
        n = root.xpath(".//n:ide/n:mod", namespaces=ns)
    else:
        n = root.xpath(".//ide/mod")
    return (n[0].text or "").strip() if n else ""


def extract_chave_acesso(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    ns = get_ns(tree)

    if ns:
        ids = root.xpath(".//n:infNFCom/@Id", namespaces=ns)
    else:
        ids = root.xpath(".//infNFCom/@Id")

    cand = ids[0] if ids else ""
    m = re.search(r"\d{44}", cand)
    if m:
        return m.group(0)

    xml_str = etree.tostring(root, encoding="unicode")
    m2 = re.search(r"\d{44}", xml_str)
    return m2.group(0) if m2 else ""


def detect_cancelamento_event_bytes(xml_bytes: bytes) -> Tuple[bool, Optional[str]]:
    try:
        tree = parse_xml(xml_bytes)
    except Exception:
        return False, None

    root = tree.getroot()
    ns = get_ns(tree)

    if ns:
        tp_nodes = root.xpath(".//n:tpEvento | .//tpEvento", namespaces=ns)
        xevt_nodes = root.xpath(".//n:xEvento | .//xEvento", namespaces=ns)
        xmot_nodes = root.xpath(".//n:xMotivo | .//xMotivo", namespaces=ns)
        ch_nodes = root.xpath(".//n:chNFCom | .//n:chNFe | .//chNFCom | .//chNFe", namespaces=ns)
    else:
        tp_nodes = root.xpath(".//tpEvento")
        xevt_nodes = root.xpath(".//xEvento")
        xmot_nodes = root.xpath(".//xMotivo")
        ch_nodes = root.xpath(".//chNFCom | .//chNFe")

    if not tp_nodes:
        return False, None

    tp = (tp_nodes[0].text or "").strip()
    if tp != "110111":
        return False, None

    xevt_txt = normalize_text((xevt_nodes[0].text or "") if xevt_nodes else "")
    xmot_txt = normalize_text((xmot_nodes[0].text or "") if xmot_nodes else "")
    if ("cancel" not in xevt_txt) and ("cancel" not in xmot_txt):
        return False, None

    chave = (ch_nodes[0].text or "").strip() if ch_nodes else None
    return True, chave


# ====================================================
# Motor de REGRAS (rules.yaml)
# ====================================================

def apply_rule_to_node(rule: Dict[str, Any], node, file_name: str) -> List[Dict[str, Any]]:
    erros = []
    tipo = rule.get("tipo")

    if tipo == "regex":
        txt = (node.text or "").strip()
        pattern = rule.get("parametros", {}).get("pattern", "")
        if pattern and not re.match(pattern, txt):
            erros.append({
                "arquivo": file_name,
                "regra_id": rule.get("id"),
                "descricao_regra": rule.get("descricao"),
                "campo_xpath": rule.get("xpath"),
                "valor_encontrado": txt,
                "mensagem_erro": rule.get("mensagem_erro"),
                "sugestao_correcao": rule.get("sugestao_correcao"),
                "nivel": rule.get("nivel", "erro"),
            })

    elif tipo == "obrigatorio":
        txt = (node.text or "").strip() if node is not None else ""
        if not txt:
            erros.append({
                "arquivo": file_name,
                "regra_id": rule.get("id"),
                "descricao_regra": rule.get("descricao"),
                "campo_xpath": rule.get("xpath"),
                "valor_encontrado": txt,
                "mensagem_erro": rule.get("mensagem_erro"),
                "sugestao_correcao": rule.get("sugestao_correcao"),
                "nivel": "erro",
            })

    elif tipo == "lista_valores":
        txt = (node.text or "").strip()
        valores_ok = rule.get("parametros", {}).get("valores_permitidos", [])
        if valores_ok and txt not in valores_ok:
            erros.append({
                "arquivo": file_name,
                "regra_id": rule.get("id"),
                "descricao_regra": rule.get("descricao"),
                "campo_xpath": rule.get("xpath"),
                "valor_encontrado": txt,
                "mensagem_erro": rule.get("mensagem_erro"),
                "sugestao_correcao": rule.get("sugestao_correcao"),
                "nivel": "erro",
            })

    return erros


def validate_with_rules_yaml(tree, rules, file_name):
    erros = []
    root = tree.getroot()
    ns = get_ns(tree)

    # Detectar CRT (Simples Nacional = 1; Normal = 3)
    if ns:
        crt_nodes = root.xpath(".//n:emit/n:CRT", namespaces=ns)
    else:
        crt_nodes = root.xpath(".//emit/CRT")
    crt = (crt_nodes[0].text or "").strip() if crt_nodes else ""

    for rule in rules:
        tipo = rule.get("tipo")
        rule_id = rule.get("id")
        xpath = rule.get("xpath", "")
        params = rule.get("parametros", {}) or {}

        # Ignorar regras de PIS/COFINS por item para SN (CRT != 3)
        if rule_id in ("R_DET_PIS_VPIS_OBRIG", "R_DET_COFINS_VCOFINS_OBRIG") and crt != "3":
            continue

        if tipo == "condicional":
            if ns:
                base_nodes = root.xpath(xpath, namespaces=ns)
            else:
                base_nodes = root.xpath(xpath)

            for node in base_nodes:
                cond_xpath = params.get("condicao_xpath")
                cond_vals = params.get("condicao_valores", [])
                alvo_xpath = params.get("alvo_xpath")
                alvo_esperado = params.get("alvo_valor_esperado")
                obrig = params.get("alvo_obrigatorio", True)

                if cond_xpath:
                    if ns:
                        cond_nodes = node.xpath(cond_xpath, namespaces=ns)
                    else:
                        cond_nodes = node.xpath(cond_xpath)
                    cond_text = (cond_nodes[0].text or "").strip() if cond_nodes else ""
                else:
                    cond_text = ""

                if cond_vals and cond_text not in cond_vals:
                    continue

                if alvo_xpath:
                    if ns:
                        alvo_nodes = node.xpath(alvo_xpath, namespaces=ns)
                    else:
                        alvo_nodes = node.xpath(alvo_xpath)
                    alvo_text = (alvo_nodes[0].text or "").strip() if alvo_nodes else ""
                else:
                    alvo_text = ""

                erro = False
                if alvo_esperado is not None:
                    if alvo_text != str(alvo_esperado):
                        erro = True
                else:
                    if obrig and not alvo_text:
                        erro = True
                    if not obrig and alvo_text:
                        erro = True

                if erro:
                    erros.append({
                        "arquivo": file_name,
                        "regra_id": rule_id,
                        "descricao_regra": rule.get("descricao"),
                        "campo_xpath": f"{xpath}/{alvo_xpath}",
                        "valor_encontrado": alvo_text,
                        "mensagem_erro": rule.get("mensagem_erro"),
                        "sugestao_correcao": rule.get("sugestao_correcao"),
                        "nivel": rule.get("nivel", "erro"),
                    })

        else:
            nodes = get_xpath_nodes(tree, xpath)
            if not nodes and tipo == "obrigatorio":
                erros.append({
                    "arquivo": file_name,
                    "regra_id": rule_id,
                    "descricao_regra": rule.get("descricao"),
                    "campo_xpath": xpath,
                    "valor_encontrado": "",
                    "mensagem_erro": rule.get("mensagem_erro"),
                    "sugestao_correcao": rule.get("sugestao_correcao"),
                    "nivel": "erro",
                })
            else:
                for node in nodes:
                    erros.extend(apply_rule_to_node(rule, node, file_name))

    return erros


# ====================================================
# Regras customizadas (Python)
# ====================================================

def is_dest_pf_or_pj_nao_contrib(tree):
    root = tree.getroot()
    ns = get_ns(tree)
    dests = root.xpath(".//n:dest", namespaces=ns) if ns else root.xpath(".//dest")
    if not dests:
        return False

    d = dests[0]
    if ns:
        cpf_nodes = d.xpath("./n:CPF", namespaces=ns)
        cnpj_nodes = d.xpath("./n:CNPJ", namespaces=ns)
        ind_nodes = d.xpath("./n:indIEDest", namespaces=ns)
    else:
        cpf_nodes = d.xpath("./CPF")
        cnpj_nodes = d.xpath("./CNPJ")
        ind_nodes = d.xpath("./indIEDest")

    cpf = (cpf_nodes[0].text or "").strip() if cpf_nodes else ""
    cnpj = (cnpj_nodes[0].text or "").strip() if cnpj_nodes else ""
    ind = (ind_nodes[0].text or "").strip() if ind_nodes else ""

    if cpf and (not ind or ind in ("2", "9")):
        return True
    if cnpj and ind == "9":
        return True
    return False


def validate_cfop_pf_pj_nao_contrib(tree, file_name, cclass_cfg):
    erros = []
    if not is_dest_pf_or_pj_nao_contrib(tree):
        return erros

    root = tree.getroot()
    ns = get_ns(tree)
    sva_cclasses = cclass_cfg.get("sva_cclasses", [])

    if ns:
        dets = root.xpath(".//n:det", namespaces=ns)
        uf_e = root.xpath(".//n:emit/n:enderEmit/n:UF", namespaces=ns)
        uf_d = root.xpath(".//n:dest/n:enderDest/n:UF", namespaces=ns)
    else:
        dets = root.xpath(".//det")
        uf_e = root.xpath(".//emit/enderEmit/UF")
        uf_d = root.xpath(".//dest/enderDest/UF")

    uf_emit = (uf_e[0].text or "").strip() if uf_e else ""
    uf_dest = (uf_d[0].text or "").strip() if uf_d else ""

    cfops_ok = ["5307"] if (uf_emit and uf_dest and uf_emit == uf_dest) else ["6307"]

    for det in dets:
        if ns:
            cclass = det.xpath("./n:prod/n:cClass", namespaces=ns)
            cfop = det.xpath("./n:prod/n:CFOP", namespaces=ns)
        else:
            cclass = det.xpath("./prod/cClass")
            cfop = det.xpath("./prod/CFOP")

        cclass = (cclass[0].text or "").strip() if cclass else ""
        cfop = (cfop[0].text or "").strip() if cfop else ""

        if cclass in sva_cclasses:
            continue
        if not cfop:
            continue
        if cfop not in cfops_ok:
            erros.append({
                "arquivo": file_name,
                "regra_id": "R_CFOP_PF_NCONTRIB",
                "descricao_regra": "PF/PJ não contribuinte deve usar CFOP 5307/6307",
                "campo_xpath": ".//det/prod/CFOP",
                "valor_encontrado": cfop,
                "mensagem_erro": f"CFOP '{cfop}' incompatível.",
                "sugestao_correcao": f"Ajustar CFOP para {cfops_ok}.",
                "nivel": "erro",
            })
    return erros


def validate_sva_cfop_zero(tree, file_name, cclass_cfg):
    erros = []
    sva = cclass_cfg.get("sva_cclasses", [])
    root = tree.getroot()
    ns = get_ns(tree)
    dets = root.xpath(".//n:det", namespaces=ns) if ns else root.xpath(".//det")

    for det in dets:
        if ns:
            cclass = det.xpath("./n:prod/n:cClass", namespaces=ns)
            cfop = det.xpath("./n:prod/n:CFOP", namespaces=ns)
        else:
            cclass = det.xpath("./prod/cClass")
            cfop = det.xpath("./prod/CFOP")

        cclass = (cclass[0].text or "").strip() if cclass else ""
        cfop = (cfop[0].text or "").strip() if cfop else ""

        if cclass not in sva:
            continue
        if not cfop:
            continue

        erros.append({
            "arquivo": file_name,
            "regra_id": "R_SVA_CFOP_ZERO",
            "descricao_regra": "Itens SVA não devem possuir CFOP",
            "campo_xpath": ".//det/prod/CFOP",
            "valor_encontrado": cfop,
            "mensagem_erro": f"CFOP '{cfop}' inválido para SVA",
            "sugestao_correcao": "Remover CFOP de SVAs.",
            "nivel": "erro",
        })
    return erros


def validate_custom_rules(tree, file_name, cclass_cfg):
    erros = []
    erros.extend(validate_cfop_pf_pj_nao_contrib(tree, file_name, cclass_cfg))
    erros.extend(validate_sva_cfop_zero(tree, file_name, cclass_cfg))
    return erros


# ====================================================
# Extração DETALHADA de itens
# ====================================================

def extract_item_details(tree, file_name, cclass_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    itens = []
    ns = get_ns(tree)
    root = tree.getroot()
    dets = root.xpath(".//n:det", namespaces=ns) if ns else root.xpath(".//det")

    for det in dets:
        if ns:
            cclass_nodes = det.xpath("./n:prod/n:cClass", namespaces=ns)
            xprod_nodes = det.xpath("./n:prod/n:xProd", namespaces=ns)
            cfop_nodes = det.xpath("./n:prod/n:CFOP", namespaces=ns)
            qfat_nodes = det.xpath("./n:prod/n:qFaturada", namespaces=ns)
            umed_nodes = det.xpath("./n:prod/n:uMed", namespaces=ns)
            vitem_nodes = det.xpath("./n:prod/n:vItem", namespaces=ns)
            vprod_nodes = det.xpath("./n:prod/n:vProd", namespaces=ns)
            vdesc_nodes = det.xpath("./n:prod/n:vDesc", namespaces=ns)
            voutros_nodes = det.xpath("./n:prod/n:vOutro", namespaces=ns)

            vbcicms_nodes = det.xpath("./n:imposto/n:ICMS/n:vBC", namespaces=ns)
            picms_nodes = det.xpath("./n:imposto/n:ICMS/n:pICMS", namespaces=ns)
            vicms_nodes = det.xpath("./n:imposto/n:ICMS/n:vICMS", namespaces=ns)

            ppist_nodes = det.xpath("./n:imposto/n:PIS/n:pPIS", namespaces=ns)
            vpist_nodes = det.xpath("./n:imposto/n:PIS/n:vPIS", namespaces=ns)

            pcof_nodes = det.xpath("./n:imposto/n:COFINS/n:pCOFINS", namespaces=ns)
            vcof_nodes = det.xpath("./n:imposto/n:COFINS/n:vCOFINS", namespaces=ns)
        else:
            cclass_nodes = det.xpath("./prod/cClass")
            xprod_nodes = det.xpath("./prod/xProd")
            cfop_nodes = det.xpath("./prod/CFOP")
            qfat_nodes = det.xpath("./prod/qFaturada")
            umed_nodes = det.xpath("./prod/uMed")
            vitem_nodes = det.xpath("./prod/vItem")
            vprod_nodes = det.xpath("./prod/vProd")
            vdesc_nodes = det.xpath("./prod/vDesc")
            voutros_nodes = det.xpath("./prod/vOutro")

            vbcicms_nodes = det.xpath("./imposto/ICMS/vBC")
            picms_nodes = det.xpath("./imposto/ICMS/pICMS")
            vicms_nodes = det.xpath("./imposto/ICMS/vICMS")

            ppist_nodes = det.xpath("./imposto/PIS/pPIS")
            vpist_nodes = det.xpath("./imposto/PIS/vPIS")

            pcof_nodes = det.xpath("./imposto/COFINS/pCOFINS")
            vcof_nodes = det.xpath("./imposto/COFINS/vCOFINS")

        cclass = (cclass_nodes[0].text or "").strip() if cclass_nodes else ""
        xprod = (xprod_nodes[0].text or "").strip() if xprod_nodes else ""
        cfop = (cfop_nodes[0].text or "").strip() if cfop_nodes else ""
        qfat = (qfat_nodes[0].text or "").strip() if qfat_nodes else ""
        umed = (umed_nodes[0].text or "").strip() if umed_nodes else ""

        vitem_text = (vitem_nodes[0].text or "").strip() if vitem_nodes else ""
        vprod_text = (vprod_nodes[0].text or "").strip() if vprod_nodes else ""
        vdesc_text = (vdesc_nodes[0].text or "").strip() if vdesc_nodes else ""
        voutros_text = (voutros_nodes[0].text or "").strip() if voutros_nodes else ""

        vbcicms_text = (vbcicms_nodes[0].text or "").strip() if vbcicms_nodes else ""
        picms_text = (picms_nodes[0].text or "").strip() if picms_nodes else ""
        vicms_text = (vicms_nodes[0].text or "").strip() if vicms_nodes else ""

        ppist_text = (ppist_nodes[0].text or "").strip() if ppist_nodes else ""
        vpist_text = (vpist_nodes[0].text or "").strip() if vpist_nodes else ""

        pcof_text = (pcof_nodes[0].text or "").strip() if pcof_nodes else ""
        vcof_text = (vcof_nodes[0].text or "").strip() if vcof_nodes else ""

        vitem = to_float(vitem_text)
        vprod = to_float(vprod_text)
        vdesc = to_float(vdesc_text)
        voutros = to_float(voutros_text)
        vbcicms = to_float(vbcicms_text)
        picms = to_float(picms_text)
        vicms = to_float(vicms_text)
        ppist = to_float(ppist_text)
        vpist = to_float(vpist_text)
        pcof = to_float(pcof_text)
        vcof = to_float(vcof_text)

        vserv = vprod

        class_info = classify_item_scm_sva(xprod, cclass, cclass_cfg)

        itens.append({
            "arquivo": file_name,
            "cClass": cclass,
            "descricao": xprod,
            "CFOP": cfop,
            "qFaturada": qfat,
            "uMed": umed,
            "vItem": vitem,
            "vProd": vprod,
            "vDesc": vdesc,
            "vOutros": voutros,
            "vServ": vserv,
            "vBCICMS": vbcicms,
            "pICMS": picms,
            "vICMS": vicms,
            "pPIS": ppist,
            "vPIS": vpist,
            "pCOFINS": pcof,
            "vCOFINS": vcof,
            "class_desc": class_info["class_por_descricao"],
            "conf_desc": class_info["conf_desc"],
            "class_cclass": class_info["class_por_cclass"],
            "class_final_sugerida": class_info["class_final_sugerida"],
            "motivo_classificacao": class_info["motivo"],
            "sugestao_forte": class_info["sugestao_forte"],
        })

    return itens


# ====================================================
# Correção dos XMLs
#   - Remove CFOP de SVA (inclui STREAMING forte por descrição)
#   - Insere CFOP quando SCM aprovado e sem CFOP (quando possível inferir 5307/6307)
#   - Ajuste PF/PJ não contrib mantém 5307/6307
# ====================================================

def generate_corrected_xml(tree, cclass_cfg, corrigir_descontos: bool, usar_class_inteligente: bool):
    root = tree.getroot()
    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)

    sva_cclasses = set(cclass_cfg.get("sva_cclasses", []) or [])
    scm_cclasses = set(cclass_cfg.get("scm_cclasses", []) or [])

    # UF emit/dest
    ns_orig = get_ns(tree)
    orig_root = tree.getroot()
    if ns_orig:
        uf_e = orig_root.xpath(".//n:emit/n:enderEmit/n:UF", namespaces=ns_orig)
        uf_d = orig_root.xpath(".//n:dest/n:enderDest/n:UF", namespaces=ns_orig)
    else:
        uf_e = orig_root.xpath(".//emit/enderEmit/UF")
        uf_d = orig_root.xpath(".//dest/enderDest/UF")

    uf_emit = (uf_e[0].text or "").strip() if uf_e else ""
    uf_dest = (uf_d[0].text or "").strip() if uf_d else ""

    dest_pf_nao_contrib = is_dest_pf_or_pj_nao_contrib(tree)

    # CFOP inferido por UF (usado para inserir quando SCM e sem CFOP)
    if uf_emit and uf_dest and uf_emit == uf_dest:
        cfop_inferido = "5307"
    elif uf_emit and uf_dest:
        cfop_inferido = "6307"
    else:
        cfop_inferido = None

    dets = copy_root.xpath(".//n:det", namespaces=ns) if ns else copy_root.xpath(".//det")

    for det in dets:
        if ns:
            cclass_nodes = det.xpath("./n:prod/n:cClass", namespaces=ns)
            cfop_nodes = det.xpath("./n:prod/n:CFOP", namespaces=ns)
            xprod_nodes = det.xpath("./n:prod/n:xProd", namespaces=ns)
            vitem_nodes = det.xpath("./n:prod/n:vItem", namespaces=ns)
            vprod_nodes = det.xpath("./n:prod/n:vProd", namespaces=ns)
        else:
            cclass_nodes = det.xpath("./prod/cClass")
            cfop_nodes = det.xpath("./prod/CFOP")
            xprod_nodes = det.xpath("./prod/xProd")
            vitem_nodes = det.xpath("./prod/vItem")
            vprod_nodes = det.xpath("./prod/vProd")

        cclass_text = (cclass_nodes[0].text or "").strip() if cclass_nodes else ""
        xprod_text = (xprod_nodes[0].text or "").strip() if xprod_nodes else ""
        cfop_text = (cfop_nodes[0].text or "").strip() if cfop_nodes else ""

        # Decide SCM/SVA
        if usar_class_inteligente:
            info = classify_item_scm_sva(xprod_text, cclass_text, cclass_cfg)

            # Novo: se descrição dá SVA forte (STREAMING/TV etc), class_final já sai SVA forte.
            class_final = info["class_final_sugerida"]

            # fallback: se ainda indefinido, usa mapeamento cclass
            if class_final == "INDEFINIDO":
                if cclass_text in sva_cclasses:
                    class_final = "SVA"
                elif cclass_text in scm_cclasses:
                    class_final = "SCM"
        else:
            if cclass_text in sva_cclasses:
                class_final = "SVA"
            elif cclass_text in scm_cclasses:
                class_final = "SCM"
            else:
                # sem inteligente e sem mapeamento: tenta descrição forte
                dclass, dconf, _ = detect_desc_category(normalize_text(xprod_text))
                class_final = dclass if (dclass in ("SCM", "SVA") and dconf >= 0.96) else "INDEFINIDO"

        # 1) Remover CFOP de SVA
        if class_final == "SVA" and cfop_nodes:
            for node in cfop_nodes:
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)
            cfop_nodes = []
            cfop_text = ""

        # 2) Inserir/ajustar CFOP para SCM (especialmente quando aprovado/corrigido e sem CFOP)
        #    - PF/PJ não contrib: força 5307/6307
        #    - Caso geral: se estiver sem CFOP e conseguirmos inferir UF, insere 5307/6307
        if class_final == "SCM":
            target_cfop = None

            if dest_pf_nao_contrib and cfop_inferido:
                target_cfop = cfop_inferido
            else:
                # pedido do usuário: inserir CFOP quando SCM aprovado e corrigido.
                # usamos inferido por UF quando disponível.
                if (not cfop_text) and cfop_inferido:
                    target_cfop = cfop_inferido

            if target_cfop and (not cfop_text or cfop_text != target_cfop):
                if not cfop_nodes:
                    if ns:
                        prod_nodes = det.xpath("./n:prod", namespaces=ns)
                    else:
                        prod_nodes = det.xpath("./prod")
                    if prod_nodes:
                        prod_node = prod_nodes[0]
                        if ns and "n" in ns:
                            cfop_elem = etree.SubElement(prod_node, f"{{{ns['n']}}}CFOP")
                        else:
                            cfop_elem = etree.SubElement(prod_node, "CFOP")
                        cfop_elem.text = target_cfop
                else:
                    cfop_nodes[0].text = target_cfop

        # 3) Paliativo desconto vProd=vItem
        if corrigir_descontos and vitem_nodes and vprod_nodes:
            vi_text = (vitem_nodes[0].text or "").strip()
            vp_text = (vprod_nodes[0].text or "").strip()
            vi = to_float(vi_text)
            vp = to_float(vp_text)
            if vp < vi:
                vprod_nodes[0].text = vi_text

    return etree.tostring(new_tree, encoding="utf-8", xml_declaration=True)


# ====================================================
# PDF – Erros
# ====================================================

def generate_pdf(df: pd.DataFrame, logo_path: str = LOGO_PATH) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    x = 40
    y = h - 60

    try:
        c.drawImage(logo_path, x, h - 100, width=90, preserveAspectRatio=True, mask='auto')
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x + 100, h - 60, "Relatório de Erros – NFCom")
        y = h - 120
    except Exception:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, h - 60, "Relatório de Erros – NFCom")
        y = h - 100

    c.setFont("Helvetica", 10)

    for _, row in df.iterrows():
        bloco = [
            f"Arquivo: {row.get('arquivo','')}",
            f"Regra: {row.get('regra_id','')} - {row.get('descricao_regra','')}",
            f"XPath: {row.get('campo_xpath','')}",
            f"Valor: {row.get('valor_encontrado','')}",
            f"Erro: {row.get('mensagem_erro','')}",
            f"Sugestão: {row.get('sugestao_correcao','')}",
            "-" * 120
        ]

        for linha in bloco:
            wrap = simpleSplit(linha, "Helvetica", 10, w - 80)
            for l in wrap:
                if y < 70:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = h - 60
                c.drawString(x, y, l)
                y -= 14

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        x, 40,
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
    )

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_pdf_cancelamento(qtd_ativos: int, qtd_cancelados: int, logo_path: str = LOGO_PATH) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    x = 40
    y = h - 60

    try:
        c.drawImage(logo_path, x, h - 100, width=90, preserveAspectRatio=True, mask='auto')
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x + 110, h - 60, "Resumo de Cancelamentos – NFCom")
        y = h - 130
    except Exception:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, h - 60, "Resumo de Cancelamentos – NFCom")
        y = h - 100

    c.setFont("Helvetica", 12)
    c.drawString(x, y, f"Quantidade de XML ativos (mantidos): {qtd_ativos}")
    y -= 20
    c.drawString(x, y, f"Quantidade de XML cancelados (excluídos): {qtd_cancelados}")
    y -= 30

    total = qtd_ativos + qtd_cancelados
    if total > 0:
        pct_ativo = (qtd_ativos / total) * 100
        pct_cancel = (qtd_cancelados / total) * 100
        c.drawString(x, y, f"Proporção de ativos: {pct_ativo:.2f}%")
        y -= 20
        c.drawString(x, y, f"Proporção de cancelados: {pct_cancel:.2f}%")
        y -= 20

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        x, 40,
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
    )

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ====================================================
# Excel – múltiplas abas
# ====================================================

def generate_excel_report(
    df_erros: pd.DataFrame | None,
    df_consolidado: pd.DataFrame | None,
    df_item_cclass: pd.DataFrame | None,
    df_cclass: pd.DataFrame | None,
    df_detalhe: pd.DataFrame | None,
    df_status_xml: pd.DataFrame | None,
    df_resumo: pd.DataFrame | None,
    df_class_sug: pd.DataFrame | None,
) -> bytes:
    output = io.BytesIO()

    def format_br_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(num_to_br)
        return df

    df_item_cclass_br = format_br_cols(df_item_cclass, ["total_vServ"])
    df_cclass_br = format_br_cols(df_cclass, ["total_vServ", "participacao_%"])
    if df_detalhe is not None and not df_detalhe.empty:
        numeric_cols = [
            "vItem", "vProd", "vDesc", "vOutros", "vServ",
            "vBCICMS", "pICMS", "vICMS", "pPIS", "vPIS", "pCOFINS", "vCOFINS"
        ]
        df_detalhe_br = format_br_cols(df_detalhe, numeric_cols)
    else:
        df_detalhe_br = df_detalhe

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df_resumo is not None and not df_resumo.empty:
            df_resumo.to_excel(writer, sheet_name="Resumo", index=False)

        if df_erros is not None and not df_erros.empty:
            df_erros.to_excel(writer, sheet_name="Erros Detalhados", index=False)

        if df_consolidado is not None and not df_consolidado.empty:
            df_consolidado.to_excel(writer, sheet_name="Erros Consolidados", index=False)

        if df_item_cclass_br is not None and not df_item_cclass_br.empty:
            df_item_cclass_br.to_excel(writer, sheet_name="Faturamento Item", index=False)

        if df_cclass_br is not None and not df_cclass_br.empty:
            df_cclass_br.to_excel(writer, sheet_name="Faturamento CClass", index=False)

        if df_detalhe_br is not None and not df_detalhe_br.empty:
            df_detalhe_br.to_excel(writer, sheet_name="Detalhamento Itens", index=False)

        if df_class_sug is not None and not df_class_sug.empty:
            df_class_sug.to_excel(writer, sheet_name="SCM_SVA_Sugerido", index=False)

        if df_status_xml is not None and not df_status_xml.empty:
            df_status_xml.to_excel(writer, sheet_name="Status XMLs", index=False)

    output.seek(0)
    return output.getvalue()


# ====================================================
# Processamento de um único XML (bytes)
# ====================================================

def process_single_xml_bytes(
    xml_bytes: bytes,
    logical_name: str,
    rules,
    cclass_cfg,
    corrigir_descontos: bool,
    usar_class_inteligente: bool,
):
    erros_total = []

    tree = parse_xml(xml_bytes)
    modelo = get_nf_model(tree)
    if modelo and modelo != "62":
        raise ValueError(f"Modelo {modelo} diferente de 62 (NFCom).")

    original_bytes = etree.tostring(tree, encoding="utf-8", xml_declaration=True)
    chave = extract_chave_acesso(tree)

    erros_total.extend(validate_with_rules_yaml(tree, rules, logical_name))
    erros_total.extend(validate_custom_rules(tree, logical_name, cclass_cfg))

    itens_detalhe = extract_item_details(tree, logical_name, cclass_cfg)

    corrigido_bytes = generate_corrected_xml(
        tree,
        cclass_cfg,
        corrigir_descontos=corrigir_descontos,
        usar_class_inteligente=usar_class_inteligente,
    )

    foi_corrigido = corrigido_bytes != original_bytes
    return erros_total, itens_detalhe, corrigido_bytes, foi_corrigido, chave


# ====================================================
# Estado (para não “sumir” ao baixar)
# ====================================================

def init_state():
    st.session_state.setdefault("has_results", False)
    st.session_state.setdefault("results", {})
    st.session_state.setdefault("downloads", {})


def clear_results():
    st.session_state["has_results"] = False
    st.session_state["results"] = {}
    st.session_state["downloads"] = {}


# ====================================================
# Processa lote e salva no session_state
# ====================================================

def run_processing(uploaded_files, cancel_keys, rules, cclass_cfg, consolidar, corrigir_descontos, usar_class_inteligente):
    erros_total: List[Dict[str, Any]] = []
    erros_invalidos: List[Dict[str, Any]] = []
    itens_detalhe: List[Dict[str, Any]] = []
    xml_resultados: List[Dict[str, Any]] = []
    canceled_xmls: List[Dict[str, Any]] = []

    for f in uploaded_files:
        nome = f.name
        try:
            content = f.read()

            if nome.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        for info in zf.infolist():
                            if not info.filename.lower().endswith(".xml"):
                                continue

                            base_name = info.filename.replace("\\", "/").replace("/", "_")
                            try:
                                xml_bytes = zf.read(info)

                                is_canc, chave_evt = detect_cancelamento_event_bytes(xml_bytes)
                                if is_canc:
                                    canceled_xmls.append({"base_name": base_name, "chave": chave_evt, "tipo": "evento_cancelamento"})
                                    continue

                                logical_name = f"{nome}::{info.filename}"

                                er, det, corr_bytes, foi_corrigido, chave = process_single_xml_bytes(
                                    xml_bytes, logical_name, rules, cclass_cfg,
                                    corrigir_descontos, usar_class_inteligente
                                )

                                if cancel_keys and chave and chave in cancel_keys:
                                    canceled_xmls.append({"base_name": base_name, "chave": chave, "tipo": "lista_canceladas"})
                                    continue

                                erros_total.extend(er)
                                itens_detalhe.extend(det)
                                xml_resultados.append({
                                    "base_name": base_name,
                                    "conteudo": corr_bytes,
                                    "corrigido": foi_corrigido,
                                    "chave": chave
                                })

                            except ValueError as e_xml:
                                erros_invalidos.append({"arquivo": f"{nome}::{info.filename}", "erro": str(e_xml)})

                except zipfile.BadZipFile:
                    erros_invalidos.append({"arquivo": nome, "erro": "Arquivo ZIP inválido ou corrompido."})

            else:
                base_name = nome

                is_canc, chave_evt = detect_cancelamento_event_bytes(content)
                if is_canc:
                    canceled_xmls.append({"base_name": base_name, "chave": chave_evt, "tipo": "evento_cancelamento"})
                    continue

                logical_name = nome
                er, det, corr_bytes, foi_corrigido, chave = process_single_xml_bytes(
                    content, logical_name, rules, cclass_cfg,
                    corrigir_descontos, usar_class_inteligente
                )

                if cancel_keys and chave and chave in cancel_keys:
                    canceled_xmls.append({"base_name": base_name, "chave": chave, "tipo": "lista_canceladas"})
                    continue

                erros_total.extend(er)
                itens_detalhe.extend(det)
                xml_resultados.append({
                    "base_name": base_name,
                    "conteudo": corr_bytes,
                    "corrigido": foi_corrigido,
                    "chave": chave
                })

        except ValueError as e:
            erros_invalidos.append({"arquivo": nome, "erro": str(e)})

    df_itens = pd.DataFrame(itens_detalhe) if itens_detalhe else pd.DataFrame()
    df_erros = pd.DataFrame(erros_total) if erros_total else pd.DataFrame()

    # Propostas divergentes (só sugestões fortes)
    if not df_itens.empty:
        divergentes = df_itens[
            (df_itens["sugestao_forte"] == True) &
            (df_itens["class_final_sugerida"] != df_itens["class_cclass"])
        ].copy()
    else:
        divergentes = pd.DataFrame()

    # Faturamento
    if not df_itens.empty:
        df_item_cclass = (
            df_itens.groupby(["cClass", "descricao"])
            .agg(qtd_itens=("arquivo", "count"), total_vServ=("vServ", "sum"))
            .reset_index()
        )
        df_cclass = (
            df_itens.groupby("cClass")
            .agg(qtd_itens=("arquivo", "count"), total_vServ=("vServ", "sum"))
            .reset_index()
        )
        total_geral_vserv = float(df_cclass["total_vServ"].sum())
        if total_geral_vserv > 0:
            df_cclass["participacao_%"] = (df_cclass["total_vServ"] / total_geral_vserv) * 100
    else:
        df_item_cclass = pd.DataFrame()
        df_cclass = pd.DataFrame()
        total_geral_vserv = 0.0

    # Status XMLs
    rows_status = []
    for x in xml_resultados:
        status = "corrigido" if x["corrigido"] else "sem_correcao"
        rows_status.append({"arquivo_base": x["base_name"], "chave": x.get("chave", ""), "status": status})
    for cxml in canceled_xmls:
        rows_status.append({
            "arquivo_base": cxml["base_name"],
            "chave": cxml.get("chave", ""),
            "status": cxml.get("tipo", "cancelado")
        })
    df_status_xml = pd.DataFrame(rows_status) if rows_status else pd.DataFrame()

    # Resumo
    df_resumo = pd.DataFrame([
        {"Métrica": "XMLs válidos processados (ativos + cancelados)", "Valor": len(xml_resultados) + len(canceled_xmls)},
        {"Métrica": "XMLs ativos (mantidos)", "Valor": len(xml_resultados)},
        {"Métrica": "XMLs cancelados (eventos + lista)", "Valor": len(canceled_xmls)},
        {"Métrica": "Total de erros/alertas encontrados (apenas ativos)", "Valor": len(df_erros)},
        {"Métrica": "Total de itens de faturamento (apenas ativos)", "Valor": len(df_itens)},
        {"Métrica": "Total faturado (vServ, apenas ativos)", "Valor": num_to_br(total_geral_vserv)},
    ])

    # Consolidado erros
    df_consolidado = None
    if (not df_erros.empty) and consolidar:
        df_consolidado = (
            df_erros.groupby(["regra_id", "descricao_regra", "mensagem_erro", "sugestao_correcao"])
            .agg(qtd=("arquivo", "count"), arquivos=("arquivo", lambda x: ", ".join(sorted(set(x)))))
            .reset_index()
        )

    # ZIP ativos (gera 1x)
    zip_bytes = b""
    if xml_resultados:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for x in xml_resultados:
                base = x["base_name"]
                base_sem_ext = base[:-4] if base.lower().endswith(".xml") else base
                out_name = f"{base_sem_ext}_corrigido.xml" if x["corrigido"] else f"{base_sem_ext}_sem_correcao.xml"
                z.writestr(out_name, x["conteudo"])
        buf.seek(0)
        zip_bytes = buf.getvalue()

    # PDFs/Excel (gera 1x)
    pdf_erros_bytes = b""
    if not df_erros.empty:
        pdf_erros_bytes = generate_pdf(df_erros)

    pdf_cancel_bytes = b""
    if canceled_xmls:
        pdf_cancel_bytes = generate_pdf_cancelamento(len(xml_resultados), len(canceled_xmls))

    excel_bytes = generate_excel_report(
        df_erros=df_erros if not df_erros.empty else None,
        df_consolidado=df_consolidado,
        df_item_cclass=df_item_cclass if not df_item_cclass.empty else None,
        df_cclass=df_cclass if not df_cclass.empty else None,
        df_detalhe=df_itens if not df_itens.empty else None,
        df_status_xml=df_status_xml if not df_status_xml.empty else None,
        df_resumo=df_resumo if not df_resumo.empty else None,
        df_class_sug=divergentes if not divergentes.empty else None,
    )

    return {
        "erros_invalidos": erros_invalidos,
        "xml_resultados": xml_resultados,
        "canceled_xmls": canceled_xmls,
        "df_itens": df_itens,
        "df_erros": df_erros,
        "df_consolidado": df_consolidado,
        "df_item_cclass": df_item_cclass,
        "df_cclass": df_cclass,
        "df_status_xml": df_status_xml,
        "df_resumo": df_resumo,
        "divergentes": divergentes,
        "downloads": {
            "zip_ativos": zip_bytes,
            "pdf_erros": pdf_erros_bytes,
            "pdf_cancel": pdf_cancel_bytes,
            "excel": excel_bytes,
        }
    }


# ====================================================
# UI
# ====================================================

def main():
    init_state()

    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image(LOGO_PATH)
        except Exception:
            st.write("")
    with col2:
        st.markdown(
            "<h2>Validador NFCom Modelo 62</h2>"
            "<p>Contare – Contabilidade especializada em Provedores de Internet</p>",
            unsafe_allow_html=True
        )

    st.write("Valide, corrija e gere relatórios completos (Excel/PDF/ZIP) de NFCom (modelo 62).")

    rules = load_rules()
    cclass_cfg = load_cclass_config()

    st.sidebar.header("Opções")
    consolidar = st.sidebar.checkbox("Consolidar erros iguais", value=True)
    corrigir_descontos = st.sidebar.checkbox("Corrigir descontos (vProd = vItem quando vProd < vItem)", value=False)
    usar_class_inteligente = st.sidebar.checkbox(
        "Aplicar classificação inteligente SCM/SVA nas correções (usar descrição do item)",
        value=True,
        help=(
            "Agora inclui detecção forte de STREAMING/TV como SVA para remover CFOP, mesmo quando o cClass estiver errado."
        )
    )

    st.sidebar.markdown("---")
    cancel_file = st.sidebar.file_uploader(
        "Relação de NFCom canceladas (CSV/TXT com chaves de acesso)",
        type=["csv", "txt"],
        accept_multiple_files=False
    )
    cancel_keys = set()
    if cancel_file is not None:
        raw = cancel_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin1", errors="ignore")
        cancel_keys = set(re.findall(r"\d{44}", text))

    st.sidebar.markdown("---")
    if st.sidebar.button("Limpar resultados (reset)", type="secondary"):
        clear_results()
        st.rerun()

    with st.form("form_upload"):
        uploaded = st.file_uploader(
            "Selecione arquivos XML ou ZIP contendo XMLs",
            type=["xml", "zip"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("Validar arquivos", type="primary")

    if submitted:
        if not uploaded:
            st.warning("Selecione ao menos 1 arquivo.")
        else:
            with st.spinner("Processando arquivos..."):
                payload = run_processing(
                    uploaded_files=uploaded,
                    cancel_keys=cancel_keys,
                    rules=rules,
                    cclass_cfg=cclass_cfg,
                    consolidar=consolidar,
                    corrigir_descontos=corrigir_descontos,
                    usar_class_inteligente=usar_class_inteligente,
                )
                st.session_state["results"] = payload
                st.session_state["downloads"] = payload["downloads"]
                st.session_state["has_results"] = True
            st.success("Processamento concluído.")
            st.rerun()

    if not st.session_state["has_results"]:
        st.info("Envie arquivos e clique em **Validar arquivos**. Depois disso, você pode baixar relatórios sem perder a tela.")
        st.markdown(
            "<hr><p style='text-align:center;font-size:12px;'>"
            "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
            "</p>",
            unsafe_allow_html=True
        )
        return

    r = st.session_state["results"]
    dls = st.session_state["downloads"]

    if r.get("erros_invalidos"):
        st.subheader("Arquivos não processados (inválidos ou modelo diferente de 62)")
        st.dataframe(pd.DataFrame(r["erros_invalidos"]), use_container_width=True)

    if r.get("xml_resultados") and dls.get("zip_ativos"):
        st.download_button(
            "Baixar XMLs (corrigidos e sem correção) – apenas ATIVOS",
            data=dls["zip_ativos"],
            file_name="xml_nfcom_ativos_processados.zip",
            mime="application/zip",
            key="download_zip_xmls_persist"
        )

    df_itens = r["df_itens"]
    df_erros = r["df_erros"]
    df_consolidado = r["df_consolidado"]
    df_item_cclass = r["df_item_cclass"]
    df_cclass = r["df_cclass"]
    df_status_xml = r["df_status_xml"]
    df_resumo = r["df_resumo"]
    divergentes = r["divergentes"]
    canceled_xmls = r["canceled_xmls"]

    if not divergentes.empty:
        st.subheader("Propostas de classificação SCM/SVA (sugestões fortes)")
        st.info(
            "Inclui STREAMING/TV como SVA forte para remover CFOP mesmo quando o cClass estiver invertido."
        )
        st.dataframe(
            divergentes[[
                "arquivo", "descricao", "cClass", "CFOP",
                "class_desc", "conf_desc", "class_cclass", "class_final_sugerida",
                "motivo_classificacao"
            ]],
            use_container_width=True
        )
    else:
        st.info("Nenhuma sugestão forte de reclassificação SCM/SVA foi identificada.")

    if not df_erros.empty:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Erros Detalhados",
            "Erros Consolidados",
            "Faturamento por CClass",
            "Dashboard"
        ])

        with tab1:
            st.subheader("Erros por arquivo (apenas NF ativas)")
            st.dataframe(df_erros, use_container_width=True)
            st.download_button(
                "Baixar CSV detalhado",
                data=df_erros.to_csv(index=False),
                file_name="erros_detalhado.csv",
                key="download_erros_detalhado_persist"
            )
            if dls.get("pdf_erros"):
                st.download_button(
                    "Baixar PDF de erros",
                    data=dls["pdf_erros"],
                    file_name="erros_nfcom.pdf",
                    mime="application/pdf",
                    key="download_erros_pdf_persist"
                )

        with tab2:
            if df_consolidado is not None and not df_consolidado.empty:
                st.dataframe(df_consolidado, use_container_width=True)
                st.download_button(
                    "Baixar CSV consolidado",
                    data=df_consolidado.to_csv(index=False),
                    file_name="erros_consolidados.csv",
                    key="download_erros_consolidado_persist"
                )
            else:
                st.info("Marque 'Consolidar erros iguais' na barra lateral para agrupar ou não há erros.")

        with tab3:
            st.subheader("Relatório de faturamento por CClass e Item (aba)")
            if not df_itens.empty:
                st.markdown("### Totais por Item (cClass + descrição)")
                st.dataframe(df_item_cclass, use_container_width=True)
                st.download_button(
                    "Baixar CSV – Totais por item",
                    data=df_item_cclass.to_csv(index=False),
                    file_name="faturamento_por_item_cclass.csv",
                    key="download_item_cclass_tab_persist"
                )

                st.markdown("### Totais por CClass")
                st.dataframe(df_cclass, use_container_width=True)
                st.download_button(
                    "Baixar CSV – Totais por CClass",
                    data=df_cclass.to_csv(index=False),
                    file_name="faturamento_por_cclass.csv",
                    key="download_cclass_tab_persist"
                )
            else:
                st.info("Nenhum item de faturamento encontrado.")

        with tab4:
            st.subheader("Dashboard de erros (apenas NF ativas)")
            total_erros = len(df_erros)
            total_arquivos = df_erros["arquivo"].nunique()
            total_regras = df_erros["regra_id"].nunique()

            colm1, colm2, colm3 = st.columns(3)
            colm1.metric("Total de erros/alertas", total_erros)
            colm2.metric("Arquivos afetados", total_arquivos)
            colm3.metric("Regras diferentes acionadas", total_regras)

            st.markdown("### Erros por regra")
            st.bar_chart(df_erros.groupby("regra_id").size())

            st.markdown("### Erros por arquivo")
            st.bar_chart(df_erros.groupby("arquivo").size())

            if "nivel" in df_erros.columns:
                st.markdown("### Erros por nível")
                st.bar_chart(df_erros.groupby("nivel").size())

    st.subheader("Relatório de faturamento por CClass e Item (sempre disponível – apenas NF ativas)")
    if not df_itens.empty:
        st.markdown("### Totais por Item (cClass + descrição)")
        st.dataframe(df_item_cclass, use_container_width=True)
        st.markdown("### Totais por CClass")
        st.dataframe(df_cclass, use_container_width=True)
        st.markdown("### Detalhamento completo dos itens (para conferência)")
        st.dataframe(df_itens, use_container_width=True)
    else:
        st.info("Nenhum item de faturamento encontrado nos XML válidos (ativos).")

    if canceled_xmls:
        qtd_ativos = len(r["xml_resultados"])
        qtd_cancelados = len(canceled_xmls)

        st.subheader("Resumo de cancelamentos aplicados")
        st.write(f"XML ATIVOS (mantidos em ZIP/relatórios): **{qtd_ativos}**")
        st.write(f"XML CANCELADOS (eventos ou listados): **{qtd_cancelados}**")
        st.dataframe(pd.DataFrame(canceled_xmls), use_container_width=True)

        if dls.get("pdf_cancel"):
            st.download_button(
                "Baixar PDF – Resumo de cancelamentos",
                data=dls["pdf_cancel"],
                file_name="resumo_cancelamentos_nfcom.pdf",
                mime="application/pdf",
                key="download_pdf_cancelamento_persist"
            )

    if dls.get("excel"):
        st.download_button(
            "Baixar Relatório Excel Completo",
            data=dls["excel"],
            file_name="relatorio_nfcom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_relatorio_persist"
        )

    st.markdown(
        "<hr><p style='text-align:center;font-size:12px;'>"
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

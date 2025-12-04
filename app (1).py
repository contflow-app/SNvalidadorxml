import io
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yaml
from lxml import etree
import re
import zipfile

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# Caminho do logo da Contare no repositório
LOGO_PATH = "Logo-Contare-ISP-1.png"


# ====================================================
# Namespace NFCom
# ====================================================

def get_ns(tree: etree._ElementTree) -> Dict[str, str]:
    """
    Retorna o namespace padrão do NFCom mapeado como prefixo 'n'.
    """
    root = tree.getroot()
    default_ns = root.nsmap.get(None)
    return {"n": default_ns} if default_ns else {}


# ====================================================
# Configuração
# ====================================================

st.set_page_config(page_title="Validador NFCom 62 - Contare", layout="wide")


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
    """
    Converte número para string no formato brasileiro (1.234,56).
    Se não for número, devolve como string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        x = float(value)
        s = f"{x:,.2f}"  # 1,234.56
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(value)


# ====================================================
# Palavras-chave SCM / SVA e classificação inteligente
# ====================================================

SCM_KEYWORDS = [
    "fibra", "fibra optica", "fibra óptica",
    "banda larga", "internet", "link", "link dedicado",
    "ftth", "plano", "velocidade", "scm",
    "dados", "conexão", "conexao", "wifi", "wi-fi",
    "provedor", "acesso", "link ip", "link rede"
]

SVA_KEYWORDS = [
    "antivirus", "anti-virus", "anti vírus", "antivírus",
    "e-mail", "email", "correio eletrônico", "correio eletronico",
    "suporte premium", "ip fixo", "voip", "telefonia",
    "serviço adicional", "servicos adicionais", "serviços adicionais",
    "sva"
]


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("á", "a").replace("ã", "a").replace("â", "a")
    s = s.replace("é", "e").replace("ê", "e")
    s = s.replace("í", "i")
    s = s.replace("ó", "o").replace("õ", "o").replace("ô", "o")
    s = s.replace("ú", "u")
    s = s.replace("ç", "c")
    return s


def classify_item_scm_sva(xprod: str, cclass: str, cclass_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifica item como SCM ou SVA usando:
      - descrição xProd (palavras-chave)
      - cClass informado (listas config: scm_cclasses, sva_cclasses)

    Retorna dict com:
      - class_por_descricao: "SCM" / "SVA" / "INDEFINIDO"
      - class_por_cclass: "SCM" / "SVA" / "INDEFINIDO"
      - class_final_sugerida: "SCM" / "SVA" / "INDEFINIDO" / "AMBIGUO"
      - motivo: texto explicando lógica
    """
    desc_norm = normalize_text(xprod or "")
    cclass = (cclass or "").strip()

    sva_cclasses = set(cclass_cfg.get("sva_cclasses", []) or [])
    scm_cclasses = set(cclass_cfg.get("scm_cclasses", []) or [])

    # 1) Classificação pela descrição
    desc_is_scm = any(k in desc_norm for k in SCM_KEYWORDS)
    desc_is_sva = any(k in desc_norm for k in SVA_KEYWORDS)

    if desc_is_scm and not desc_is_sva:
        class_desc = "SCM"
    elif desc_is_sva and not desc_is_scm:
        class_desc = "SVA"
    elif desc_is_scm and desc_is_sva:
        class_desc = "AMBIGUO"
    else:
        class_desc = "INDEFINIDO"

    # 2) Classificação pelo cClass
    if cclass in scm_cclasses:
        class_cclass = "SCM"
    elif cclass in sva_cclasses:
        class_cclass = "SVA"
    else:
        class_cclass = "INDEFINIDO"

    # 3) Classificação final sugerida
    motivo = []

    # Se descrição é clara, priorizamos descrição
    if class_desc in ("SCM", "SVA"):
        class_final = class_desc
        motivo.append(f"Descrição sugere {class_desc}.")
        if class_cclass not in ("INDEFINIDO", class_desc):
            motivo.append(f"cClass indica {class_cclass}, divergente da descrição.")
    elif class_desc == "AMBIGUO":
        class_final = "AMBIGUO"
        motivo.append("Descrição ambígua (indica SCM e SVA). Revisão manual recomendada.")
        if class_cclass != "INDEFINIDO":
            motivo.append(f"cClass atual indica {class_cclass}.")
    else:
        # Sem indicação forte pela descrição → usar cClass
        if class_cclass in ("SCM", "SVA"):
            class_final = class_cclass
            motivo.append(f"Sem indicação clara na descrição. Usando cClass ({class_cclass}).")
        else:
            class_final = "INDEFINIDO"
            motivo.append("Sem indicação clara na descrição nem no cClass. Revisão manual recomendada.")

    return {
        "class_por_descricao": class_desc,
        "class_por_cclass": class_cclass,
        "class_final_sugerida": class_final,
        "motivo": " ".join(motivo)
    }


# ====================================================
# XML helpers
# ====================================================

def parse_xml(file_bytes: bytes) -> etree._ElementTree:
    try:
        parser = etree.XMLParser(remove_blank_text=True)
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
    """
    Tenta extrair a chave de acesso (44 dígitos) da NFCom.
    Primeiro busca em infNFCom/@Id; se não achar, procura 44 dígitos em todo o XML.
    """
    root = tree.getroot()
    ns = get_ns(tree)

    # Tenta caminho padrão NFCom
    if ns:
        ids = root.xpath(".//n:infNFCom/@Id", namespaces=ns)
    else:
        ids = root.xpath(".//infNFCom/@Id")

    cand = ids[0] if ids else ""
    m = re.search(r"\d{44}", cand)
    if m:
        return m.group(0)

    # Fallback: busca 44 dígitos em todo o XML
    xml_str = etree.tostring(root, encoding="unicode")
    m2 = re.search(r"\d{44}", xml_str)
    return m2.group(0) if m2 else ""


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
                "nivel": rule.get("nivel", "erro"),
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
    """
    Continua apenas gerando erro quando CFOP incompatível com PF/PJ não contribuinte
    (não corrige aqui – a correção automática é feita em generate_corrected_xml).
    """
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

    if uf_emit == uf_dest:
        cfops_ok = ["5307"]
    else:
        cfops_ok = ["6307"]

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
    """
    Regras de SVA com CFOP – agora apenas como apoio (a lógica final de remoção
    será feita em generate_corrected_xml, considerando classificação inteligente).
    """
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
# Extração DETALHADA de itens (faturamento + conferência)
# ====================================================

def extract_item_details(tree, file_name, cclass_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrai dados detalhados por item:
      - Identificação: arquivo, cClass, xProd, CFOP, qFaturada, uMed
      - Valores: vItem, vProd, vDesc, vOutros, vServ
      - Impostos: vBCICMS, pICMS, vICMS, pPIS, vPIS, pCOFINS, vCOFINS
      - Classificação inteligente SCM/SVA (não altera XML).
    """
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

        vserv = vprod  # NFCom, consideramos vProd como valor de serviço

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
            "class_cclass": class_info["class_por_cclass"],
            "class_final_sugerida": class_info["class_final_sugerida"],
            "motivo_classificacao": class_info["motivo"],
        })

    return itens


# ====================================================
# Correção dos XMLs (CFOP de SVA + Paliativo vProd = vItem)
# com classificação inteligente SCM/SVA opcional
# ====================================================

def generate_corrected_xml(tree, cclass_cfg, corrigir_descontos: bool, usar_class_inteligente: bool):
    root = tree.getroot()
    copy_root = etree.fromstring(etree.tostring(root))
    new_tree = etree.ElementTree(copy_root)
    ns = get_ns(new_tree)
    sva_cclasses = set(cclass_cfg.get("sva_cclasses", []) or [])
    scm_cclasses = set(cclass_cfg.get("scm_cclasses", []) or [])

    # Info para CFOP automático PF/PJ não contrib.
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

    if uf_emit and uf_dest and uf_emit == uf_dest:
        cfop_expected_pf = "5307"
    elif uf_emit and uf_dest:
        cfop_expected_pf = "6307"
    else:
        cfop_expected_pf = None

    dets = copy_root.xpath(".//n:det", namespaces=ns) if ns else copy_root.xpath(".//det")

    for det in dets:
        # Leitura de campos
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

        # Classificação para correção
        if usar_class_inteligente:
            class_info = classify_item_scm_sva(xprod_text, cclass_text, cclass_cfg)
            class_final = class_info["class_final_sugerida"]
        else:
            if cclass_text in sva_cclasses:
                class_final = "SVA"
            elif cclass_text in scm_cclasses:
                class_final = "SCM"
            else:
                class_final = "INDEFINIDO"

        # 1) Remoção de CFOP de SVA (apenas se classificado como SVA)
        if class_final == "SVA" and cfop_nodes:
            for node in cfop_nodes:
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)
            cfop_nodes = []
            cfop_text = ""

        # 2) Ajuste de CFOP para SCM PF/PJ não contribuinte (5307/6307)
        if class_final == "SCM" and dest_pf_nao_contrib and cfop_expected_pf:
            if not cfop_text or cfop_text != cfop_expected_pf:
                # Criar ou ajustar nós CFOP
                if not cfop_nodes:
                    # criar nodo CFOP
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
                        cfop_elem.text = cfop_expected_pf
                else:
                    cfop_nodes[0].text = cfop_expected_pf

        # 3) Correção vProd = vItem se desconto
        if corrigir_descontos:
            if vitem_nodes and vprod_nodes:
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

def generate_pdf(df: pd.DataFrame, logo_path: str = LOGO_PATH):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    x = 40
    y = h - 60

    # Logo
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
            f"Arquivo: {row['arquivo']}",
            f"Regra: {row['regra_id']} - {row['descricao_regra']}",
            f"XPath: {row['campo_xpath']}",
            f"Valor: {row['valor_encontrado']}",
            f"Erro: {row['mensagem_erro']}",
            f"Sugestão: {row['sugestao_correcao']}",
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
        x,
        40,
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
    )

    c.save()
    buffer.seek(0)
    return buffer


# ====================================================
# PDF – Resumo de cancelamentos
# ====================================================

def generate_pdf_cancelamento(qtd_ativos: int, qtd_cancelados: int, logo_path: str = LOGO_PATH):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    x = 40
    y = h - 60

    # Cabeçalho com logo
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
        x,
        40,
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
    )

    c.save()
    buffer.seek(0)
    return buffer


# ====================================================
# Excel (XLSX) – múltiplas abas
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
) -> io.BytesIO:
    """
    Gera um arquivo Excel com múltiplas abas:
      - Resumo
      - Erros detalhados
      - Erros consolidados
      - Faturamento por item
      - Faturamento por CClass
      - Detalhamento de itens
      - Propostas SCM/SVA
      - Status XMLs
    Com números formatados em estilo BR nas abas de faturamento/detalhamento.
    """
    output = io.BytesIO()

    # Prepara cópias para formatação BR
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
    return output


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
    """
    Processa um único XML (em bytes) e retorna:
    - lista de erros
    - lista de itens detalhados para faturamento/conferência
    - bytes do XML corrigido
    - flag se foi de fato corrigido (True/False)
    - chave de acesso (44 dígitos), se encontrada
    """
    erros_total = []

    tree = parse_xml(xml_bytes)
    modelo = get_nf_model(tree)
    if modelo and modelo != "62":
        raise ValueError(f"Modelo {modelo} diferente de 62 (NFCom).")

    # Serialização "normalizada" do original para comparação
    original_bytes = etree.tostring(tree, encoding="utf-8", xml_declaration=True)

    # Extrai chave de acesso
    chave = extract_chave_acesso(tree)

    # Regras YAML
    erros_total.extend(validate_with_rules_yaml(tree, rules, logical_name))
    # Regras customizadas
    erros_total.extend(validate_custom_rules(tree, logical_name, cclass_cfg))
    # Itens detalhados + classificação inteligente
    itens_detalhe = extract_item_details(tree, logical_name, cclass_cfg)
    # XML corrigido (aplicando ou não classificação inteligente)
    corrigido_bytes = generate_corrected_xml(
        tree,
        cclass_cfg,
        corrigir_descontos=corrigir_descontos,
        usar_class_inteligente=usar_class_inteligente
    )

    # Verifica se de fato houve alteração estrutural
    foi_corrigido = corrigido_bytes != original_bytes

    return erros_total, itens_detalhe, corrigido_bytes, foi_corrigido, chave


# ====================================================
# Interface Streamlit
# ====================================================

def main():
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
    corrigir_descontos = st.sidebar.checkbox(
        "Corrigir descontos (vProd = vItem quando vProd < vItem)",
        value=False
    )
    usar_class_inteligente = st.sidebar.checkbox(
        "Aplicar classificação inteligente SCM/SVA nas correções (usar descrição do item)",
        value=False,
        help=(
            "Quando marcado, a remoção de CFOP e o ajuste de CFOP (5307/6307 para PF/PJ não contribuinte) "
            "serão baseados na classificação sugerida (SCM/SVA) pela descrição do item. "
            "Recomenda-se rodar primeiro sem marcar, revisar as sugestões na tela, e depois habilitar."
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
        # Captura todos os blocos de 44 dígitos como chaves de acesso
        cancel_keys = set(re.findall(r"\d{44}", text))

    uploaded = st.file_uploader(
        "Selecione arquivos XML ou ZIP contendo XMLs",
        type=["xml", "zip"],
        accept_multiple_files=True
    )

    if uploaded and st.button("Validar arquivos"):
        erros_total: List[Dict[str, Any]] = []
        erros_invalidos: List[Dict[str, Any]] = []
        itens_detalhe: List[Dict[str, Any]] = []
        xml_resultados: List[Dict[str, Any]] = []   # XMLs ATIVOS (vão para ZIP/relatórios)
        canceled_xmls: List[Dict[str, Any]] = []    # XMLs CANCELADOS (filtrados)

        for f in uploaded:
            nome = f.name
            try:
                content = f.read()

                # Se for ZIP, processa XMLs internos
                if nome.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(content)) as zf:
                            for info in zf.infolist():
                                if info.filename.lower().endswith(".xml"):
                                    try:
                                        xml_bytes = zf.read(info)
                                        logical_name = f"{nome}::{info.filename}"

                                        # Flatten do caminho interno: pasta1/arquivo.xml → pasta1_arquivo.xml
                                        base_name = info.filename.replace("\\", "/").replace("/", "_")

                                        er, det, corr_bytes, foi_corrigido, chave = process_single_xml_bytes(
                                            xml_bytes,
                                            logical_name,
                                            rules,
                                            cclass_cfg,
                                            corrigir_descontos,
                                            usar_class_inteligente,
                                        )

                                        # Se houver relação de canceladas e a chave estiver nela → filtra
                                        if cancel_keys and chave and chave in cancel_keys:
                                            canceled_xmls.append({
                                                "base_name": base_name,
                                                "chave": chave
                                            })
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
                                        erros_invalidos.append({
                                            "arquivo": f"{nome}::{info.filename}",
                                            "erro": str(e_xml)
                                        })
                    except zipfile.BadZipFile:
                        erros_invalidos.append({
                            "arquivo": nome,
                            "erro": "Arquivo ZIP inválido ou corrompido."
                        })
                else:
                    # XML individual
                    logical_name = nome
                    base_name = nome
                    er, det, corr_bytes, foi_corrigido, chave = process_single_xml_bytes(
                        content,
                        logical_name,
                        rules,
                        cclass_cfg,
                        corrigir_descontos,
                        usar_class_inteligente,
                    )

                    if cancel_keys and chave and chave in cancel_keys:
                        canceled_xmls.append({
                            "base_name": base_name,
                            "chave": chave
                        })
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
                erros_invalidos.append({
                    "arquivo": nome,
                    "erro": str(e)
                })

        # Arquivos inválidos
        if erros_invalidos:
            st.subheader("Arquivos não processados (inválidos ou modelo diferente de 62)")
            st.dataframe(pd.DataFrame(erros_invalidos), use_container_width=True)

        # ZIP com TODOS os XMLs válidos ATIVOS (corrigidos e sem correção)
        if xml_resultados:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for x in xml_resultados:
                    base = x["base_name"]
                    if base.lower().endswith(".xml"):
                        base_sem_ext = base[:-4]
                    else:
                        base_sem_ext = base

                    if x["corrigido"]:
                        out_name = f"{base_sem_ext}_corrigido.xml"
                    else:
                        out_name = f"{base_sem_ext}_sem_correcao.xml"

                    z.writestr(out_name, x["conteudo"])
            buf.seek(0)
            st.download_button(
                "Baixar XMLs (corrigidos e sem correção) – apenas ATIVOS",
                data=buf,
                file_name="xml_nfcom_ativos_processados.zip",
                mime="application/zip",
                key="download_zip_xmls"
            )

        # Se não houver itens detalhados nem erros
        if not erros_total and not itens_detalhe:
            st.warning("Nenhum XML NFCom válido foi encontrado nos arquivos enviados (ativos).")
            return

        # Monta DataFrames globais de itens
        df_itens = pd.DataFrame(itens_detalhe) if itens_detalhe else pd.DataFrame()

        # Propostas de reclassificação SCM/SVA (para revisão)
        if not df_itens.empty:
            df_class_sug = df_itens.copy()
            # Regra simples: divergências ou ambiguidades
            divergentes = df_class_sug[
                (df_class_sug["class_final_sugerida"] != "INDEFINIDO")
                & (df_class_sug["class_final_sugerida"] != df_class_sug["class_cclass"])
            ]
        else:
            df_class_sug = pd.DataFrame()
            divergentes = pd.DataFrame()

        if not divergentes.empty:
            st.subheader("Propostas de classificação SCM/SVA (para conferência)")
            st.info(
                "A tabela abaixo mostra itens em que a descrição sugere uma classificação diferente do cClass. "
                "Use isto para revisar antes de habilitar a opção de classificação inteligente na barra lateral."
            )
            st.dataframe(
                divergentes[
                    [
                        "arquivo", "descricao", "cClass", "CFOP",
                        "class_cclass", "class_final_sugerida", "motivo_classificacao"
                    ]
                ],
                use_container_width=True
            )
        else:
            st.info(
                "Nenhuma divergência relevante de classificação SCM/SVA encontrada entre descrição e cClass, "
                "ou nenhum item processado."
            )

        if not df_itens.empty:
            # Totais por item (cClass + descrição)
            df_item_cclass = (
                df_itens
                .groupby(["cClass", "descricao"])
                .agg(
                    qtd_itens=("arquivo", "count"),
                    total_vServ=("vServ", "sum")
                )
                .reset_index()
            )

            # Totais por CClass
            df_cclass = (
                df_itens
                .groupby("cClass")
                .agg(
                    qtd_itens=("arquivo", "count"),
                    total_vServ=("vServ", "sum")
                )
                .reset_index()
            )
            total_geral_vserv = df_cclass["total_vServ"].sum()
            if total_geral_vserv > 0:
                df_cclass["participacao_%"] = (df_cclass["total_vServ"] / total_geral_vserv) * 100
        else:
            df_item_cclass = pd.DataFrame()
            df_cclass = pd.DataFrame()
            total_geral_vserv = 0.0

        # df_status_xml (inclui ativos e cancelados)
        rows_status = []
        for x in xml_resultados:
            base = x["base_name"]
            status = "corrigido" if x["corrigido"] else "sem_correcao"
            rows_status.append({"arquivo_base": base, "chave": x.get("chave", ""), "status": status})
        for cxml in canceled_xmls:
            rows_status.append({
                "arquivo_base": cxml["base_name"],
                "chave": cxml.get("chave", ""),
                "status": "cancelado"
            })
        df_status_xml = pd.DataFrame(rows_status) if rows_status else pd.DataFrame()

        # Resumo
        resumo_rows = []
        resumo_rows.append({
            "Métrica": "XMLs válidos processados (ativos + cancelados)",
            "Valor": len(xml_resultados) + len(canceled_xmls)
        })
        resumo_rows.append({
            "Métrica": "XMLs ativos (mantidos)",
            "Valor": len(xml_resultados)
        })
        resumo_rows.append({
            "Métrica": "XMLs cancelados (excluídos das validações/faturamento)",
            "Valor": len(canceled_xmls)
        })
        resumo_rows.append({
            "Métrica": "Total de erros/alertas encontrados (apenas ativos)",
            "Valor": len(erros_total)
        })
        resumo_rows.append({
            "Métrica": "Total de itens de faturamento (apenas ativos)",
            "Valor": len(df_itens)
        })
        resumo_rows.append({
            "Métrica": "Total faturado (vServ, apenas ativos)",
            "Valor": num_to_br(total_geral_vserv)
        })
        df_resumo = pd.DataFrame(resumo_rows)

        # ====================================================
        # Quando houver erros → abas completas
        # ====================================================
        if erros_total:
            df_erros = pd.DataFrame(erros_total)

            if consolidar:
                df_consolidado = (
                    df_erros
                    .groupby(["regra_id", "descricao_regra", "mensagem_erro", "sugestao_correcao"])
                    .agg(
                        qtd=("arquivo", "count"),
                        arquivos=("arquivo", lambda x: ", ".join(sorted(set(x))))
                    )
                    .reset_index()
                )
            else:
                df_consolidado = None

            tab1, tab2, tab3, tab4 = st.tabs([
                "Erros Detalhados",
                "Erros Consolidados",
                "Faturamento por CClass",
                "Dashboard"
            ])

            # --------- Tab 1: Erros Detalhados ---------
            with tab1:
                st.subheader("Erros por arquivo (apenas NF ativas)")
                st.dataframe(df_erros, use_container_width=True)

                st.download_button(
                    "Baixar CSV detalhado",
                    df_erros.to_csv(index=False),
                    "erros_detalhado.csv",
                    key="download_erros_detalhado"
                )

                pdf = generate_pdf(df_erros)
                st.download_button(
                    "Baixar PDF de erros",
                    pdf,
                    "erros_nfcom.pdf",
                    mime="application/pdf",
                    key="download_erros_pdf"
                )

            # --------- Tab 2: Erros Consolidados ---------
            with tab2:
                if df_consolidado is not None and not df_consolidado.empty:
                    st.dataframe(df_consolidado, use_container_width=True)
                    st.download_button(
                        "Baixar CSV consolidado",
                        df_consolidado.to_csv(index=False),
                        "erros_consolidados.csv",
                        key="download_erros_consolidado"
                    )
                else:
                    st.info("Marque 'Consolidar erros iguais' na barra lateral para agrupar ou não há erros.")

            # --------- Tab 3: Faturamento (quando há erros) ---------
            with tab3:
                st.subheader("Relatório de faturamento por CClass e Item (aba)")
                if not df_itens.empty:
                    st.markdown("### Totais por Item (cClass + descrição)")
                    st.dataframe(df_item_cclass, use_container_width=True)

                    st.download_button(
                        "Baixar CSV – Totais por item",
                        df_item_cclass.to_csv(index=False),
                        "faturamento_por_item_cclass.csv",
                        key="download_item_cclass_tab"
                    )

                    st.markdown("### Totais por CClass")
                    st.dataframe(df_cclass, use_container_width=True)

                    st.download_button(
                        "Baixar CSV – Totais por CClass",
                        df_cclass.to_csv(index=False),
                        "faturamento_por_cclass.csv",
                        key="download_cclass_tab"
                    )
                else:
                    st.info("Nenhum item de faturamento encontrado.")

            # --------- Tab 4: Dashboard ---------
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
                por_regra = df_erros.groupby("regra_id").size()
                st.bar_chart(por_regra)

                st.markdown("### Erros por arquivo")
                por_arquivo = df_erros.groupby("arquivo").size()
                st.bar_chart(por_arquivo)

                if "nivel" in df_erros.columns:
                    st.markdown("### Erros por nível")
                    por_nivel = df_erros.groupby("nivel").size()
                    st.bar_chart(por_nivel)
        else:
            df_erros = pd.DataFrame()
            df_consolidado = None

        # ====================================================
        # Faturamento SEMPRE visível (com ou sem erros)
        # ====================================================
        st.subheader("Relatório de faturamento por CClass e Item (sempre disponível – apenas NF ativas)")
        if not df_itens.empty:
            st.markdown("### Totais por Item (cClass + descrição)")
            st.dataframe(df_item_cclass, use_container_width=True)

            st.download_button(
                "Baixar CSV – Totais por item",
                df_item_cclass.to_csv(index=False),
                "faturamento_por_item_cclass.csv",
                key="download_item_cclass_global"
            )

            st.markdown("### Totais por CClass")
            st.dataframe(df_cclass, use_container_width=True)

            st.download_button(
                "Baixar CSV – Totais por CClass",
                df_cclass.to_csv(index=False),
                "faturamento_por_cclass.csv",
                key="download_cclass_global"
            )

            st.markdown("### Detalhamento completo dos itens (para conferência)")
            st.dataframe(df_itens, use_container_width=True)
        else:
            st.info("Nenhum item de faturamento encontrado nos XML válidos (ativos).")

        # ====================================================
        # Resumo de cancelamentos (se fornecida lista)
        # ====================================================
        if cancel_keys:
            qtd_ativos = len(xml_resultados)
            qtd_cancelados = len(canceled_xmls)

            st.subheader("Resumo de cancelamentos aplicados")
            st.write(f"XML ATIVOS (mantidos em ZIP/relatórios): **{qtd_ativos}**")
            st.write(f"XML CANCELADOS (excluídos de ZIP/relatórios): **{qtd_cancelados}**")

            if canceled_xmls:
                st.markdown("XML cancelados identificados:")
                df_cancel = pd.DataFrame(canceled_xmls)
                st.dataframe(df_cancel, use_container_width=True)
            else:
                st.info("Nenhuma NFCom dos arquivos enviados constava na relação de canceladas.")

            pdf_cancel = generate_pdf_cancelamento(qtd_ativos, qtd_cancelados)
            st.download_button(
                "Baixar PDF – Resumo de cancelamentos",
                data=pdf_cancel,
                file_name="resumo_cancelamentos_nfcom.pdf",
                mime="application/pdf",
                key="download_pdf_cancelamento"
            )

        # ====================================================
        # Excel completo
        # ====================================================
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

        st.download_button(
            "Baixar Relatório Excel Completo",
            data=excel_bytes,
            file_name="relatorio_nfcom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_relatorio"
        )

    st.markdown(
        "<hr><p style='text-align:center;font-size:12px;'>"
        "Desenvolvido por Raul Martins – Contare Contabilidade especializada em Provedores de Internet"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

# ingest.py

import os
import re
import json
import unicodedata
from datetime import date
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from dateutil import parser as date_parser
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ← Nuevos imports para fallback LLM
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# ─── Entorno ─────────────────────────────────────────────
load_dotenv(find_dotenv())
DATA_DIR   = os.getenv("DATA_DIR", "data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

# ─── Regiones ────────────────────────────────────────────
REGION_COUNTRIES = {
    'Norteamérica': ['Estados Unidos','Canadá','México'],
    'Centroamérica': ['Belice','Costa Rica','El Salvador','Guatemala','Honduras','Nicaragua','Panamá'],
    'Caribe':        ['Cuba','República Dominicana','Puerto Rico','Haití','Jamaica'],
    'América Latina':['México','Guatemala','Belice','El Salvador','Honduras','Nicaragua','Costa Rica','Panamá','Cuba','República Dominicana','Puerto Rico','Haití','Jamaica','Colombia','Venezuela','Ecuador','Perú','Bolivia','Paraguay','Chile','Argentina','Uruguay','Brasil'],
    'América del Sur':['Argentina','Bolivia','Brasil','Chile','Colombia','Ecuador','Guyana','Paraguay','Perú','Surinam','Uruguay','Venezuela'],
    'Europa':        ['España','Francia','Alemania','Italia','Reino Unido','Portugal','Bélgica','Países Bajos','Suiza','Austria','Suecia','Noruega','Dinamarca','Finlandia','Polonia','Rumanía','Hungría','República Checa','Grecia','Irlanda','Rusia','Ucrania','Bulgaria','Croacia','Serbia'],
}
REGION_COUNTRIES_INV = {
    country: region
    for region, countries in REGION_COUNTRIES.items()
    for country in countries
}

# ─── Normalización de países ───────────────────────────────────
COUNTRY_ALIASES = {
    "usa": "Estados Unidos", "us": "Estados Unidos",
    "ee uu": "Estados Unidos", "ee. uu.": "Estados Unidos",
    "united states": "Estados Unidos", "united states of america": "Estados Unidos",
    "spain": "España", "espana": "España", "españa": "España",
    "canada": "Canadá", "mexico": "México", "brazil": "Brasil", "colombia": "Colombia",
    "estados unidos": "Estados Unidos", "estado unidos": "Estados Unidos",
}
def normalize_country(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = s.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').strip()
    return COUNTRY_ALIASES.get(s, raw.strip().title())

# Variantes comunes de España
for alt in ['esp','sp','espana','españa.','españa']:
    REGION_COUNTRIES_INV[alt.title().strip('.')] = 'Europa'

# ─── Utilidades de texto ─────────────────────────────────
def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ─── Fallback LLM para parsear fechas en calls ─────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
def parse_calls_llm(text: str) -> tuple[str|None, str|None]:
    if not OPENAI_KEY or not text.strip():
        return None, None
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-4o-mini", temperature=0)
    prompt = (
        f"Extrae fecha de inicio y fin de esta descripción:\n\n"
        f"\"{text.strip()}\"\n\n"
        "Responde EXACTAMENTE en JSON: {\"start\":\"YYYY-MM-DD\",\"end\":\"YYYY-MM-DD\"}. "
        "Si solo hay una fecha, ponla en ambos. Si no hay ninguna, null."
    )
    resp = llm([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(resp.content)
        return parsed.get("start"), parsed.get("end")
    except Exception:
        return None, None

# ─── Lectura/normalización de encabezados ─────────────────
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas para header simple o doble.
    - Doble: usa la segunda fila si no es 'Unnamed'; si no, usa la primera.
    - Simple: normaliza directamente.
    """
    trans = str.maketrans('áéíóúÁÉÍÓÚñÑ','aeiouaeiounn')
    cols: list[str] = []
    if isinstance(df.columns, pd.MultiIndex):
        for lvl0, lvl1 in df.columns:
            name = lvl1 if pd.notna(lvl1) and not str(lvl1).lower().startswith("unnamed") else lvl0
            s = str(name).strip().lower().translate(trans)
            s = re.sub(r'[^0-9a-z_]+','_',s).strip('_')
            cols.append(s)
    else:
        for col in df.columns:
            s = str(col).strip().lower().translate(trans)
            s = re.sub(r'[^0-9a-z_]+','_',s).strip('_')
            cols.append(s)
    df.columns = cols
    return df

def make_unique(cols: list[str]) -> list[str]:
    cnt,out = {},[]
    for c in cols:
        cnt[c] = cnt.get(c,0)+1
        out.append(c if cnt[c]==1 else f"{c}_{cnt[c]}")
    return out

def split_iso_range(raw: str) -> tuple[str|None,str|None]:
    if not raw or raw.strip().upper()=="TBD":
        return None,None
    parts = re.split(r'\s*[/]\s*',raw.strip())
    if len(parts)==2:
        return parts[0],parts[1]
    return parts[0],parts[0]

def read_excel_safely(path: str) -> pd.DataFrame:
    """
    Lee un Excel intentando primero header doble y luego simple.
    Lanza FileNotFoundError si no existe.
    """
    # FileNotFoundError debe propagarse (lo manejamos en el caller)
    try:
        df = pd.read_excel(path, header=[0,1], dtype=str, engine="openpyxl")
    except Exception:
        # cae a header simple
        df = pd.read_excel(path, header=0, dtype=str, engine="openpyxl")
    return df

# ─── Campos temáticos por tabla ──────────────────────────
TOPIC_FIELDS: dict[str, list[str]] = {
    "events": [
        "tema_de_enfoque", "tema", "temas", "tema_principal", "focus", "enfoque"
    ],
    "calls": [
        "prioridades", "prioridad", "objetivo", "objetivos", "temas_prioritarios"
    ],
    "donors": [
        "principales_temas_de_interes", "temas_de_interes", "areas_de_interes",
        "lineas_de_apoyo", "lineas_de_aporte", "interes_tematico"
    ],
}

def extract_topic_from_row(row_dict: dict, table: str) -> str:
    """
    Construye un texto temático normalizado a partir de los campos relevantes.
    Busca por nombre exacto y por 'contains' en los keys ya normalizados.
    """
    text_parts: list[str] = []
    keys = TOPIC_FIELDS.get(table, [])
    if not keys:
        return ""
    # row_dict ya viene con headers normalizados
    for k, v in row_dict.items():
        kn = norm_text(k)
        if any(kn == t or t in kn for t in keys):
            if isinstance(v, str) and v.strip():
                text_parts.append(v)
    return norm_text(" ".join(text_parts)) if text_parts else ""

# ─── Carga e indexación ──────────────────────────────────
def load_excels_to_docs(data_dir: str) -> list[Document]:
    docs: list[Document] = []
    title_map = {
        "donors": "organizacion",
        "events": "evento",
        "calls":  "nombre_de_la_convocatoria"
    }

    for table in ("donors","calls","events"):
        path = os.path.join(data_dir, f"{table}.xlsx")
        if not os.path.exists(path):
            print(f"⚠ Aviso: no se encontró `{table}.xlsx` en {data_dir}. Se continúa sin esta tabla.")
            continue

        print(f"📥 Leyendo `{table}.xlsx`…")
        try:
            df = read_excel_safely(path)
        except FileNotFoundError:
            print(f"❌ No se pudo abrir `{table}.xlsx`. Se omite esta tabla.")
            continue

        df = normalize_headers(df)
        df.columns = make_unique(df.columns.tolist())
        # Elimina filas/columnas completamente vacías
        df = df.dropna(how="all",axis=0).dropna(how="all",axis=1)

        if df.empty:
            print(f"⚠ Aviso: `{table}.xlsx` no contiene filas útiles tras limpieza.")
            continue

        for idx, row in df.iterrows():
            md: dict = {}
            # Limpieza de valores NaN para JSON
            row_dict_raw = row.to_dict()
            row_clean = {k: (None if pd.isna(v) else v) for k, v in row_dict_raw.items()}
            row_dict = row_clean  # alias semántico

            # título canónico
            raw_title = row_dict.get(title_map[table], "")
            md["titulo"] = (str(raw_title).strip() if raw_title else None) or None

            # Geografía
            pais_raw = str(row_dict.get("pais","") or "")
            pais      = normalize_country(pais_raw)
            ciudad    = str(row_dict.get("ciudad","") or "").strip().title()
            prov      = str(row_dict.get("estado_provincia","") or "").strip().title()
            region    = REGION_COUNTRIES_INV.get(pais)

            if table=="events":
                raw = str(row_dict.get("fecha","") or "").replace('"','').replace("'",'').replace('\n',' ').strip()
                fi_str,ff_str = split_iso_range(raw)
                if fi_str:
                    try:
                        fi = date_parser.isoparse(fi_str).date()
                        md["fecha_inicio"] = fi
                        md["mes_inicio"]   = fi.month
                    except Exception:
                        pass
                if ff_str:
                    try:
                        md["fecha_fin"] = date_parser.isoparse(ff_str).date()
                    except Exception:
                        pass
                if md.get("fecha_inicio") and not md.get("fecha_fin"):
                    md["fecha_fin"] = md["fecha_inicio"]

                estado = str(row_dict.get("evento_pasado_pendiente_falta_info","") or "").strip()
                if not md.get("fecha_inicio"):
                    estado = "Falta info"

                md.update({
                    "pais": pais,
                    "ciudad": ciudad,
                    "estado_evento": estado,
                    "region": region,
                })

            elif table=="calls":
                raw = str(row_dict.get("fecha_de_convocatoria","") or "").replace('"','').replace("'",'').replace('\n',' ').strip()
                # ISO
                dates_iso = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw)
                if dates_iso:
                    try:
                        md["fecha_conv_inicio"] = date_parser.isoparse(dates_iso[0]).date()
                    except Exception:
                        pass
                    if len(dates_iso)>1:
                        try:
                            md["fecha_conv_fin"] = date_parser.isoparse(dates_iso[-1]).date()
                        except Exception:
                            pass
                    else:
                        md["fecha_conv_fin"] = md.get("fecha_conv_inicio")
                else:
                    # libre español
                    meses_esp = r"(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)[a-z]*"
                    parts = re.findall(rf"(\d{{1,2}}\s+{meses_esp}\s+20\d{{2}})", raw, flags=re.IGNORECASE)
                    if parts:
                        try:
                            dt = date_parser.parse(parts[0], dayfirst=True).date()
                            md["fecha_conv_inicio"] = dt
                        except Exception:
                            pass
                        if len(parts)>1:
                            try:
                                dt2 = date_parser.parse(parts[1], dayfirst=True).date()
                                md["fecha_conv_fin"] = dt2
                            except Exception:
                                pass
                        else:
                            md["fecha_conv_fin"] = md.get("fecha_conv_inicio")
                    else:
                        # fallback LLM
                        start_s, end_s = parse_calls_llm(raw)
                        if start_s:
                            try:
                                md["fecha_conv_inicio"] = date_parser.isoparse(start_s).date()
                            except Exception:
                                pass
                        if end_s:
                            try:
                                md["fecha_conv_fin"] = date_parser.isoparse(end_s).date()
                            except Exception:
                                pass

                # mes de cierre
                if md.get("fecha_conv_fin"):
                    md["mes_cierre"] = md["fecha_conv_fin"].month

                lo = raw.lower()
                if lo.startswith("rolling"):
                    estado="Rolling"
                elif "cerrada" in lo:
                    estado="Cerrada"
                else:
                    estado=str(row_dict.get("estado","") or "Abierta").strip().title()

                if md.get("fecha_conv_inicio") and not md.get("fecha_conv_fin"):
                    md["fecha_conv_fin"] = md["fecha_conv_inicio"]
                if md.get("fecha_conv_fin") and not md.get("fecha_conv_inicio"):
                    md["fecha_conv_inicio"] = md["fecha_conv_fin"]

                md.update({
                    "estado_convocatoria": estado,
                    "pais": pais,
                    "ciudad": None,
                    "estado_provincia": None,
                    "region": region,
                })

            else:  # donors
                md.update({
                    "pais": pais,
                    "ciudad": ciudad,
                    "estado_provincia": prov,
                    "region": region,
                })

            # Tema/objetivo precalculado
            md["topic_text"] = extract_topic_from_row(row_dict, table)

            # Metadatos comunes
            md["source_table"] = table
            md["row_index"]    = idx

            # Serialización del contenido sin NaN
            page_json = json.dumps(row_clean, ensure_ascii=False)

            docs.append(Document(
                page_content=page_json,
                metadata=md
            ))

    print(f"✅ Total documentos indexados: {len(docs)}")
    return docs

def build_index():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró OPENAI_API_KEY.")
    docs = load_excels_to_docs(DATA_DIR)
    if not docs:
        raise RuntimeError("No hay documentos para indexar.")
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(docs[:BATCH_SIZE], embedding=embedder)
    for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        vectordb.add_documents(docs[i:i+BATCH_SIZE])
    print(f"✅ Índice completo con {len(docs)} documentos.")
    return vectordb

if __name__=="__main__":
    build_index()

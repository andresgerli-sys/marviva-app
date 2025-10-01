# app.py

import os
import re
import json
import unicodedata
from datetime import date
from dateutil import parser as date_parser
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
from calendar import monthrange

from ingest import (
    build_index, load_excels_to_docs,
    REGION_COUNTRIES, REGION_COUNTRIES_INV,
    COUNTRY_ALIASES, normalize_country
)
from typing import Any, Dict, List, Tuple, Optional, Set
from langchain.schema import BaseRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
# Utilidades de texto
# ─────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def norm_text(s: str) -> str:
    """minúsculas, sin acentos, sólo letras/números/espacios compactados."""
    if s is None:
        return ""
    s = strip_accents(str(s).lower())
    # unifica guiones/dashes raros a espacio
    s = s.replace('–', ' ').replace('—', ' ').replace('-', ' ')
    s = re.sub(r'[^a-z0-9\s]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ─────────────────────────────────────────────────────────
# Retrievers
# ─────────────────────────────────────────────────────────

class RetrieverWrapper(BaseRetriever):
    """(legacy) contenedor para un retriever base (FAISS)."""
    base_retriever: Any
    class Config:
        arbitrary_types_allowed = True
    def _get_relevant_documents(self, query: str):
        return self.base_retriever.get_relevant_documents(query)
    async def _aget_relevant_documents(self, query: str):
        return await self.base_retriever.aget_relevant_documents(query)

class ListRetriever(BaseRetriever):
    """
    Devuelve exactamente la lista de documentos ya filtrados
    (evita depender de filtros de FAISS que no entienden rangos).
    """
    docs: List
    class Config:
        arbitrary_types_allowed = True
    def _get_relevant_documents(self, query: str):
        return self.docs
    async def _aget_relevant_documents(self, query: str):
        return self.docs

# ─────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────

TODAY = date.today().isoformat()
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "today", "location"],
    template=(
        "Hoy es {today}. Eres el asistente experto de MarViva, especializado en Donantes, Convocatorias y Eventos.\n"
        "RESPONDE A LAS PREGUNTAS USANDO SÓLO LOS DATOS PROVENIENTES DE LOS EXCELS YA CARGADOS.\n"
        "- Si el usuario pide el “contenido” de un evento, convocatoria o donante, "
        "devuelve **todas** las columnas de la fila correspondiente.\n"
        "- Para listados simples, sigue usando `fecha_inicio`/`fecha_fin`/`mes_inicio` en Eventos, "
        "`fecha_conv_inicio`/`fecha_conv_fin`/`mes_cierre` en Convocatorias, y los campos de Donantes.\n"
        "- Siempre filtra por `pais`, `ciudad`, `estado_provincia`, `source_table`.\n"
        "❌ No inventes datos ni confíes en NER; usa sólo la metadata y el contenido real.\n\n"
        "**FORMATO DE SALIDA**\n"
        "1. Si es un “contenido”, muestra cada campo: “- <columna>: <valor>”.\n"
        "2. Si es un listado, numerado, con línea en blanco entre ítems y fechas “DD de MMMM de YYYY”.\n\n"
        "Contexto (docs filtrados):\n"
        "{context}\n\n"
        "Ubicación seleccionada: {location}\n\n"
        "Pregunta: {question}"
    ),
)


# ─────────────────────────────────────────────────────────
# Firma de datos → invalidación de caché del índice
# ─────────────────────────────────────────────────────────

def data_signature(data_dir: str) -> str:
    names = ("donors.xlsx", "calls.xlsx", "events.xlsx")
    parts = []
    for n in names:
        p = os.path.join(data_dir, n)
        try:
            stt = os.stat(p)
            parts.append(f"{n}:{stt.st_mtime_ns}:{stt.st_size}")
        except FileNotFoundError:
            parts.append(f"{n}:0:0")
    return "|".join(parts)

@st.cache_resource(show_spinner=False)
def get_index(_sig: str):
    return build_index()

# ─────────────────────────────────────────────────────────
# Fechas en español + solapamiento
# ─────────────────────────────────────────────────────────

SPANISH_MONTHS = ["", "enero","febrero","marzo","abril","mayo","junio",
                  "julio","agosto","septiembre","octubre","noviembre","diciembre"]

def fmt_es(d: Optional[date]) -> str:
    if not d:
        return "—"
    return f"{d.day:02d} de {SPANISH_MONTHS[d.month]} de {d.year}"

def overlaps(a0: Optional[date], a1: Optional[date],
             b0: Optional[date], b1: Optional[date]) -> bool:
    A0 = a0 or date.min
    A1 = a1 or date.max
    B0 = b0 or date.min
    B1 = b1 or date.max
    return not (A1 < B0 or B1 < A0)

# ─────────────────────────────────────────────────────────
# JSON tolerante a NaN
# ─────────────────────────────────────────────────────────

def safe_json_loads(s: str) -> Dict:
    if not s:
        return {}
    try:
        return json.loads(s, parse_constant=lambda x: None)
    except Exception:
        try:
            s2 = re.sub(r'\bNaN\b', 'null', s)
            s2 = re.sub(r'\bnan\b', 'null', s2, flags=re.I)
            s2 = re.sub(r'\bInfinity\b', 'null', s2)
            s2 = re.sub(r'-\s*Infinity', 'null', s2)
            return json.loads(s2)
        except Exception:
            return {}

# ─────────────────────────────────────────────────────────
# TAXONOMÍAS (Eventos, Donantes, Calls) — jerárquicas
# ─────────────────────────────────────────────────────────

def _kw(*args: str) -> List[str]:
    # normaliza palabras clave a formato sin acentos/espacios extra
    return [norm_text(x) for x in args if x]

# Macros + subtemas para EVENTOS
TAXONOMY_EVENTS: Dict[str, Dict] = {
    # Macros
    "ev_ocean_gov": {
        "label": "Océanos y gobernanza",
        "keywords": _kw("gobernanza","derecho del mar","bbnj","abnj","alta mar",
                        "politicas oceanicas","politica oceanica","tratados","acuerdos","high seas"),
        "parents": []
    },
    "ev_climate_env": {
        "label": "Cambio climático y medio ambiente",
        "keywords": _kw("clima","mitigacion","adaptacion","resiliencia costera",
                        "energia renovable marina","eolica marina","offshore wind",
                        "soluciones basadas en la naturaleza","nbs","descarbonizacion",
                        "acidificacion","oae"),
        "parents": []
    },
    "ev_blue_econ_fin": {
        "label": "Economía azul y finanzas sostenibles",
        "keywords": _kw("economia azul","blue economy","finanzas azules","blue finance",
                        "bonos azules","blue bonds","esg","impacto","blended finance",
                        "economia circular","inversion"),
        "parents": []
    },
    "ev_society_philanthropy": {
        "label": "Sociedad, filantropía y gobernanza comunitaria",
        "keywords": _kw("filantropia","rsc","csr","participacion comunitaria","justicia social",
                        "genero","equidad de genero","inclusion","gesi","comunidades"),
        "parents": []
    },
    "ev_science_innovation": {
        "label": "Ciencia, innovación y tecnología",
        "keywords": _kw("ciencia oceanica","ocean science","big data","monitoreo","robotica",
                        "robotica marina","sensores","teledeteccion","satélite","satelital",
                        "modelado","ia","inteligencia artificial","innovacion","datos"),
        "parents": []
    },
    "ev_ecosystems_biodiversity": {
        "label": "Ecosistemas y biodiversidad",
        "keywords": _kw("biodiversidad","especies","tortugas","mamelos marinos","mamiferos marinos",
                        "ballenas","delfines","tiburones","elasmobranquios","arrecifes",
                        "coral","corales","manglares","pastos marinos","invasoras","invasion biologica"),
        "parents": []
    },
    "ev_sustainable_dev": {
        "label": "Desarrollo sostenible y sociedad",
        "keywords": _kw("ods","objetivos de desarrollo sostenible","seguridad alimentaria",
                        "turismo sostenible","ciudades costeras","medios de vida","livelihoods"),
        "parents": []
    },
    # Subtemas
    "ev_conservation_protection": {
        "label": "Conservación marina y áreas protegidas",
        "keywords": _kw("conservacion","area marina protegida","areas marinas protegidas","amp","mpa",
                        "oecm","restauracion","servicios ecosistemicos","reserva marina","protected area"),
        "parents": ["ev_ecosystems_biodiversity","ev_ocean_gov"]
    },
    "ev_fisheries_aquaculture": {
        "label": "Pesca sostenible y acuicultura responsable",
        "keywords": _kw("pesca","pesquerias","acuicultura","iuu","sobrepesca","bycatch",
                        "trazabilidad","mcs","monitoreo control vigilancia"),
        "parents": ["ev_ocean_gov","ev_sustainable_dev"]
    },
    "ev_pollution": {
        "label": "Contaminación marina y soluciones",
        "keywords": _kw("contaminacion","plasticos","plásticos","microplasticos","basura marina",
                        "residuos","derrame","oil spill","quimicos","quimicos"),
        "parents": ["ev_climate_env","ev_ocean_gov"]
    },
}

# DONANTES
TAXONOMY_DONORS: Dict[str, Dict] = {
    "don_conservation_biodiversity": {
        "label": "Conservación marina y biodiversidad",
        "keywords": _kw("biodiversidad","conservacion","especies vulnerables","habitats criticos",
                        "areas protegidas","amp","mpa","oecm"),
        "parents": []
    },
    "don_protected_restoration": {
        "label": "Áreas protegidas y restauración de ecosistemas",
        "keywords": _kw("arrecifes","corales","manglares","humedales","restauracion","restauración"),
        "parents": ["don_conservation_biodiversity"]
    },
    "don_fisheries_management": {
        "label": "Pesquerías sostenibles y manejo pesquero",
        "keywords": _kw("pesca","pesquerias","manejo pesquero","iuu","trazabilidad","mcs"),
        "parents": []
    },
    "don_climate_mitigation_adaptation": {
        "label": "Cambio climático, mitigación y adaptación",
        "keywords": _kw("mitigacion","adaptacion","justicia climatica","transicion justa","resiliencia"),
        "parents": []
    },
    "don_blue_carbon": {
        "label": "Carbono azul y ecosistemas costeros",
        "keywords": _kw("carbono azul","manglares","pastos marinos","humedales","mrv"),
        "parents": ["don_climate_mitigation_adaptation"]
    },
    "don_plastics_circular": {
        "label": "Contaminación plástica y economía circular",
        "keywords": _kw("plasticos","plásticos","economia circular","residuos","reciclaje"),
        "parents": []
    },
    "don_freshwater_wash": {
        "label": "Agua dulce, cuencas y WASH",
        "keywords": _kw("agua dulce","cuencas","wash","saneamiento","agua y saneamiento"),
        "parents": []
    },
    "don_clean_energy": {
        "label": "Energía limpia y descarbonización",
        "keywords": _kw("energia limpia","descarbonizacion","offshore wind","renovables","shipping"),
        "parents": []
    },
    "don_blue_economy_livelihoods": {
        "label": "Economía azul y medios de vida responsables",
        "keywords": _kw("economia azul","blue economy","medios de vida","pymes costeras","cadenas de valor"),
        "parents": []
    },
    "don_climate_finance_blue_finance": {
        "label": "Finanzas climáticas y blue finance",
        "keywords": _kw("finanzas climaticas","blue finance","bonos azules","blended finance","metricas"),
        "parents": []
    },
    "don_governance_policy_transparency": {
        "label": "Gobernanza, políticas y transparencia",
        "keywords": _kw("gobernanza","politicas","marco legal","cumplimiento","enforcement","advocacy"),
        "parents": []
    },
    "don_social_justice_gender": {
        "label": "Justicia social, DDHH e igualdad de género",
        "keywords": _kw("justicia social","derechos humanos","genero","equidad de genero","inclusion","gesi"),
        "parents": []
    },
    "don_community_development": {
        "label": "Desarrollo comunitario y servicios sociales",
        "keywords": _kw("salud","educacion","vivienda","seguridad alimentaria","desarrollo comunitario"),
        "parents": []
    },
    "don_environmental_education": {
        "label": "Educación ambiental y sensibilización",
        "keywords": _kw("educacion ambiental","sensibilizacion","storytelling","ciencia ciudadana","steam"),
        "parents": []
    },
    "don_science_data_innovation": {
        "label": "Ciencia, datos e innovación",
        "keywords": _kw("datos satelitales","teledeteccion","modelado","tecnologias limpias","innovacion"),
        "parents": []
    },
    "don_coastal_mgmt_infrastructure": {
        "label": "Gestión costera e infraestructura sostenible",
        "keywords": _kw("ordenamiento marino espacial","msp","nbs","urbanismo costero","infraestructura"),
        "parents": []
    },
    "don_agriculture_food_systems": {
        "label": "Agricultura sostenible y sistemas alimentarios",
        "keywords": _kw("agroecologia","agroecología","sistemas alimentarios","seguridad alimentaria"),
        "parents": []
    },
    "don_capacity_building": {
        "label": "Fortalecimiento institucional y capacidades",
        "keywords": _kw("fortalecimiento institucional","capacitacion","formacion","capacidades"),
        "parents": []
    },
    "don_supply_chain_transparency": {
        "label": "Transparencia en cadenas de suministro",
        "keywords": _kw("cadenas de suministro","trazabilidad","seafood","cacao","mineria","minería"),
        "parents": []
    },
    "don_ocean_culture": {
        "label": "Exploración y cultura oceánica",
        "keywords": _kw("patrimonio cultural","arte y oceano","cultura oceanica"),
        "parents": []
    },
}

# CALLS (ejes + flags)
TAXONOMY_CALLS: Dict[str, Dict] = {
    "call_ocean_protection_mgmt": {
        "label": "Protección y gestión del océano",
        "keywords": _kw("amp","oecm","areas marinas protegidas","gestion marina","gestion del oceano",
                        "pesquerias","pesca","iuu","habitats criticos","arrecifes","manglares",
                        "pastos marinos","acuicultura","contaminacion","plasticos","residuos"),
        "parents": []
    },
    "call_biodiversity_conservation": {
        "label": "Conservación de biodiversidad y especies",
        "keywords": _kw("especies amenazadas","elasmobranquios","depredadores apice","tortugas",
                        "mamiferos marinos","restauracion ecologica","arrecifes","manglares","humedales",
                        "investigacion aplicada","conservacion aplicada"),
        "parents": []
    },
    "call_climate_resilience_blue_carbon": {
        "label": "Clima, resiliencia y carbono azul",
        "keywords": _kw("mitigacion","transicion energetica","reduccion de gei","cdr","oae",
                        "adaptacion","infraestructura verde","nbs","drr","gestion del riesgo",
                        "carbono azul","mrv","proyectos listos para inversion"),
        "parents": []
    },
    "call_circular_economy_water_wash": {
        "label": "Economía circular, agua y saneamiento",
        "keywords": _kw("economia circular de plasticos","gestion de residuos","contaminacion",
                        "agua","cuencas","wash","saneamiento"),
        "parents": []
    },
    "call_blue_economy_livelihoods": {
        "label": "Economía azul y medios de vida",
        "keywords": _kw("medios de vida costeros","pymes","turismo","diversificacion","innovacion marina",
                        "blue tech","finanzas azules","finanzas climaticas"),
        "parents": []
    },
    "call_gov_policy_transparency": {
        "label": "Gobernanza, políticas y transparencia",
        "keywords": _kw("gobernanza oceanica","bbnj","alta mar","mcs","control y vigilancia",
                        "cumplimiento","fortalecimiento institucional"),
        "parents": []
    },
    "call_science_data_knowledge": {
        "label": "Ciencia, datos y conocimiento",
        "keywords": _kw("datasets","gfw","teledeteccion","trazabilidad","redes regionales",
                        "ciencia abierta","educacion tecnica","formacion tecnica"),
        "parents": []
    },
    "call_communities_equity_human_dev": {
        "label": "Comunidades, equidad y desarrollo humano",
        "keywords": _kw("pobreza","inclusion economica","genero","gesi","derechos humanos",
                        "pueblos indigenas","salud","nutricion","bienestar","educacion",
                        "cultura oceanica","sensibilizacion"),
        "parents": []
    },
    "call_agriculture_territory_sea": {
        "label": "Agricultura sostenible y enfoque territorio‑mar",
        "keywords": _kw("agroecologia","seguridad alimentaria","manejo integrado tierra mar",
                        "desertificacion","desertificación"),
        "parents": []
    },
    # Flags/transversales
    "flag_marine_focus": {
        "label": "Enfoque marino obligatorio",
        "keywords": _kw("enfoque marino","marino obligatorio"),
        "parents": []
    },
    "flag_poverty_gesi": {
        "label": "Reducción de pobreza e inclusión GESI",
        "keywords": _kw("gesi","inclusion","pobreza","equidad de genero","genero"),
        "parents": []
    },
    "flag_local_leadership": {
        "label": "Liderazgo local",
        "keywords": _kw("liderazgo local","local leadership"),
        "parents": []
    },
    "flag_capacity_building": {
        "label": "Fortalecimiento de capacidades",
        "keywords": _kw("fortalecimiento de capacidades","capacitacion","formacion"),
        "parents": []
    },
    "flag_innovation_scalability": {
        "label": "Innovación y escalabilidad",
        "keywords": _kw("innovacion","escalabilidad","escalar","piloto a escala"),
        "parents": []
    },
    "flag_community_participation": {
        "label": "Participación comunitaria",
        "keywords": _kw("participacion comunitaria","participacion de la comunidad"),
        "parents": []
    },
    "flag_partnerships": {
        "label": "Alianzas (público‑privadas, locales, regionales)",
        "keywords": _kw("alianzas","partnerships","publico privado","publico‑privadas"),
        "parents": []
    },
    "flag_exclusions_no_overhead": {
        "label": "Exclusiones: no financian gasto general/capital/endowments",
        "keywords": _kw("no financian gasto general","no financian capital","no endowments"),
        "parents": []
    },
}

# Índice por tipo
TAXONOMIES: Dict[str, Dict[str, Dict]] = {
    "events": TAXONOMY_EVENTS,
    "donors": TAXONOMY_DONORS,
    "calls":  TAXONOMY_CALLS,
}

def taxonomy_keywords(doc_type: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    tax = TAXONOMIES.get(doc_type, {})
    for tag_id, cfg in tax.items():
        kws = set(cfg.get("keywords", []))
        out[tag_id] = kws
    return out

def taxonomy_parents(doc_type: str) -> Dict[str, List[str]]:
    tax = TAXONOMIES.get(doc_type, {})
    return {tag_id: cfg.get("parents", []) for tag_id, cfg in tax.items()}

# ─────────────────────────────────────────────────────────
# Etiquetado de documentos (multietiqueta)
# ─────────────────────────────────────────────────────────

def infer_doc_topics(doc, doc_type: str,
                     kw_index: Dict[str, Set[str]],
                     parents: Dict[str, List[str]],
                     cache: Dict) -> Set[str]:
    key = (doc.metadata.get("source_table"), doc.metadata.get("row_index"))
    if key in cache:
        return cache[key]

    # Texto base: metadata.topic_text → fallback a campos JSON
    topic_text = norm_text(doc.metadata.get("topic_text") or "")
    if not topic_text:
        data = safe_json_loads(doc.page_content)
        candidates = []
        for k, v in data.items():
            kn = norm_text(k)
            if doc_type == "events" and ("tema" in kn or "enfoque" in kn):
                if isinstance(v, str):
                    candidates.append(v)
            elif doc_type == "calls" and ("priori" in kn or "objetiv" in kn or "tema" in kn):
                if isinstance(v, str):
                    candidates.append(v)
            elif doc_type == "donors" and ("tema" in kn or "interes" in kn or "interés" in kn):
                if isinstance(v, str):
                    candidates.append(v)
        topic_text = norm_text(" ".join(candidates))

    if not topic_text:
        data = safe_json_loads(doc.page_content)
        title = norm_text(doc.metadata.get("titulo") or "")
        desc = norm_text(str(data.get("descripcion", "") or data.get("descripción", "")))
        topic_text = (title + " " + desc).strip()

    tags: Set[str] = set()
    for tag_id, kws in kw_index.items():
        if any(kw and kw in topic_text for kw in kws):
            tags.add(tag_id)

    # Propagar a padres
    added = True
    while added:
        added = False
        for t in list(tags):
            for p in parents.get(t, []):
                if p not in tags:
                    tags.add(p); added = True

    # Garantizar al menos una
    if not tags:
        tags = {f"{doc_type}:other"}

    cache[key] = tags
    return tags

def map_query_to_topic_tags(clean_q: str, doc_type: str,
                            kw_index: Dict[str, Set[str]],
                            parents: Dict[str, List[str]]) -> Set[str]:
    qn = norm_text(clean_q)
    tags: Set[str] = set()
    for tag_id, kws in kw_index.items():
        if any(kw and kw in qn for kw in kws):
            tags.add(tag_id)
    # Propaga a padres
    added = True
    while added:
        added = False
        for t in list(tags):
            for p in parents.get(t, []):
                if p not in tags:
                    tags.add(p); added = True
    return tags

# ─────────────────────────────────────────────────────────
# Helpers para mostrar etiquetas (Temas / Criterios)
# ─────────────────────────────────────────────────────────

def tag_labels_for_doc(doc_type: str, tag_ids: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Devuelve (topic_labels, flag_labels) en texto legible.
    - Para calls, los tag_id que empiezan con 'flag_' van a flag_labels.
    - Si hay '...:other', lo mostramos como 'Otros'.
    """
    tax = TAXONOMIES.get(doc_type, {})
    topics: Set[str] = set()
    flags: Set[str] = set()
    for tid in tag_ids:
        if tid in tax:
            label = tax[tid].get("label", tid)
        elif tid.endswith(":other"):
            label = "Otros"
        else:
            label = tid.replace('_',' ')
        if doc_type == "calls" and tid.startswith("flag_"):
            flags.add(label)
        else:
            topics.add(label)
    return sorted(topics), sorted(flags)

# ─────────────────────────────────────────────────────────
# Detección de intención “contenido” y búsqueda por título
# ─────────────────────────────────────────────────────────

CONTENT_TRIGGERS = [
    "contenido", "contenido completo", "toda la informacion", "toda la información",
    "todo sobre", "muestra todo", "detalle", "detalles", "todas las columnas", "todos los campos"
]

def is_content_intent(q: str) -> bool:
    nq = norm_text(q)
    return any(t in nq for t in CONTENT_TRIGGERS)

def extract_quoted_name(q: str) -> Optional[str]:
    m = re.search(r'["“”]([^"“”]{3,})["“”]', q)
    if m:
        return m.group(1).strip().strip('] ').strip()
    m = re.search(r"[']([^']{3,})[']", q)
    if m:
        return m.group(1).strip().strip('] ').strip()
    m = re.search(r"\[([^\]]{3,})\]", q)
    if m:
        return m.group(1).strip()
    return None

def trigram_similarity(a: str, b: str) -> float:
    a = norm_text(a); b = norm_text(b)
    if not a or not b:
        return 0.0
    A = {a[i:i+3] for i in range(len(a)-2)} if len(a) >= 3 else {a}
    B = {b[i:i+3] for i in range(len(b)-2)} if len(b) >= 3 else {b}
    return len(A & B) / max(1, len(A | B))

def token_jaccard(a: str, b: str) -> float:
    a = norm_text(a); b = norm_text(b)
    A = set(a.split()); B = set(b.split())
    return len(A & B) / max(1, len(A | B))

def title_similarity(a: str, b: str) -> float:
    return 0.6 * trigram_similarity(a, b) + 0.4 * token_jaccard(a, b)

def get_doc_title(doc) -> str:
    md_title = doc.metadata.get('titulo')
    if md_title:
        return str(md_title)
    data = safe_json_loads(doc.page_content)
    for k in ("evento", "organizacion", "organización", "nombre_de_la_convocatoria", "convocatoria"):
        if k in data and data[k]:
            return str(data[k])
    return ""

def find_best_doc_by_title(pool_docs: List, name: str) -> Optional[Any]:
    if not pool_docs:
        return None
    target = norm_text(name)
    exact_matches = [d for d in pool_docs if norm_text(get_doc_title(d)) == target]
    if exact_matches:
        exact_matches.sort(key=lambda d: d.metadata.get('row_index', 10**9))
        return exact_matches[0]
    scored = []
    for d in pool_docs:
        t = get_doc_title(d)
        if not t:
            continue
        s = title_similarity(t, name)
        scored.append((s, d))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_doc = scored[0]
    return best_doc if best_score >= 0.62 else None

# ─────────────────────────────────────────────────────────
# Parseo de filtros de la pregunta
# ─────────────────────────────────────────────────────────

def parse_filters(
    q: str, sd, ed, region: str, cc: str
) -> Tuple[Dict, str, List[int], Tuple[Optional[date],Optional[date]], str]:
    ql = strip_accents(q.lower())
    clean_q = re.sub(r'[^\w\s]', '', ql)

    # tipo
    if 'convocatori' in ql:
        t = 'calls'
    elif 'donant' in ql:
        t = 'donors'
    else:
        t = 'events'

    # región / país
    country_from_cc = normalize_country(cc) if cc.strip() else None
    region_from_q = next((
        reg for reg in REGION_COUNTRIES
        if strip_accents(reg.lower()) in clean_q
    ), None)
    country_from_q = None
    for alias, canon in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", clean_q):
            country_from_q = canon; break
    if not country_from_q:
        for ctry in REGION_COUNTRIES_INV:
            if re.search(rf"\b{re.escape(ctry.lower())}\b", clean_q):
                country_from_q = ctry; break

    if region_from_q:
        loc = region_from_q
    elif country_from_cc:
        loc = country_from_cc
    elif country_from_q:
        loc = country_from_q
    elif region != 'Global':
        loc = region
    else:
        loc = None

    # años y meses
    meses_map = {
        'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,
        'julio':7,'agosto':8,'septiembre':9,'setiembre':9,'octubre':10,'noviembre':11,'diciembre':12
    }
    Y = int(re.search(r"\b(20\d{2})\b", clean_q).group(1)) if re.search(r"\b(20\d{2})\b", clean_q) else None

    # extrae meses correctamente
    meses = [n for m,n in meses_map.items() if m in clean_q]

    # rango libre calls (permite solapamiento después)
    call_start, call_end = None, None
    iso = re.findall(r"(20\d{2}-\d{2}-\d{2})", clean_q)
    if len(iso) >= 2:
        call_start = date_parser.isoparse(iso[0]).date()
        call_end   = date_parser.isoparse(iso[1]).date()
    else:
        esp = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        parts = re.findall(rf"(\d{{1,2}}\s+{esp}\s+20\d{{2}})", clean_q, flags=re.IGNORECASE)
        if len(parts) >= 2:
            try:
                call_start = date_parser.parse(parts[0], dayfirst=True).date()
                call_end   = date_parser.parse(parts[1], dayfirst=True).date()
            except:
                pass

    # filtros iniciales
    f: Dict = {'source_table': t}
    if t == 'events' and Y:
        f['fecha_inicio'] = {'$gte': date(Y,1,1), '$lte': date(Y,12,31)}
    if t == 'calls':
        if Y:
            f['fecha_conv_inicio'] = {'$gte': date(Y,1,1), '$lte': date(Y,12,31)}
        if 'rolling' in clean_q:
            f['estado_convocatoria'] = 'Rolling'
        elif 'cerrad' in clean_q:
            f['estado_convocatoria'] = 'Cerrada'
        else:
            f['estado_convocatoria'] = 'Abierta'

    if loc:
        if loc in REGION_COUNTRIES:
            f['pais'] = {'$in': REGION_COUNTRIES[loc]}
        else:
            f['pais'] = loc

    return f, loc, meses, (call_start, call_end), clean_q

# ─────────────────────────────────────────────────────────
# Formateo para "contenido" (con etiquetas)
# ─────────────────────────────────────────────────────────

def render_full_content(doc) -> str:
    data = safe_json_loads(doc.page_content)
    lines = []
    # Título
    titulo = doc.metadata.get('titulo')
    if titulo:
        lines.append(f"**{titulo}**")

    # Etiquetas (Temas / Criterios)
    doc_type = doc.metadata.get("source_table")
    kw_index = taxonomy_keywords(doc_type)
    parents  = taxonomy_parents(doc_type)
    topics   = infer_doc_topics(doc, doc_type, kw_index, parents, cache={})
    topic_labels, flag_labels = tag_labels_for_doc(doc_type, topics)

    if topic_labels:
        lines.append(f"**Temas:** {', '.join(topic_labels)}")
    if doc_type == "calls" and flag_labels:
        lines.append(f"**Criterios:** {', '.join(flag_labels)}")

    if titulo or topic_labels or flag_labels:
        lines.append("")

    # Todas las columnas
    if not data:
        lines.append("No hay datos de columnas para esta fila.")
    else:
        for k, v in data.items():
            val = "—" if v is None or (isinstance(v, str) and not v.strip()) else str(v)
            lines.append(f"- {k}: {val}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────
# Aplicación de filtros (incluye taxonomía)
# ─────────────────────────────────────────────────────────

def apply_filters(
    docs_list, f: Dict, t: str, meses: List[int],
    call_range: Tuple[Optional[date],Optional[date]], clean_q: str,
    ui_range: Tuple[Optional[date], Optional[date]] = (None, None)
) -> List:
    def match_meta(d):
        for k,v in f.items():
            val = d.metadata.get(k)
            if val is None: return False
            if isinstance(v, dict):
                if '$gte' in v and val < v['$gte']: return False
                if '$lte' in v and val > v['$lte']: return False
                if '$in'  in v and val not in v['$in']: return False
            elif val != v:
                return False
        return True

    docs = [d for d in docs_list if match_meta(d)]

    # Filtro por meses en events
    if t=='events' and meses:
        docs = [d for d in docs if (
            (d.metadata.get('fecha_inicio') and d.metadata['fecha_inicio'].month in meses) or
            (d.metadata.get('fecha_fin')    and d.metadata['fecha_fin'].month    in meses)
        )]

    # Filtro Desde/Hasta (solapamiento)
    if t=='events' and (ui_range[0] or ui_range[1]):
        s, e = ui_range
        docs = [d for d in docs if overlaps(
            d.metadata.get('fecha_inicio'), d.metadata.get('fecha_fin'), s, e
        )]

    # Filtros de calls por solapamiento
    if t=='calls' and (call_range[0] or call_range[1]):
        s, e = call_range
        docs = [d for d in docs if overlaps(
            d.metadata.get('fecha_conv_inicio'),
            d.metadata.get('fecha_conv_fin'), s, e
        )]
    if t=='calls' and (ui_range[0] or ui_range[1]):
        s, e = ui_range
        docs = [d for d in docs if overlaps(
            d.metadata.get('fecha_conv_inicio'),
            d.metadata.get('fecha_conv_fin'), s, e
        )]

    # ====== Filtro TEMÁTICO con taxonomía ======
    kw_index  = taxonomy_keywords(t)
    parents   = taxonomy_parents(t)
    req_tags  = map_query_to_topic_tags(clean_q, t, kw_index, parents)
    if req_tags:
        topics_cache: Dict = {}
        docs = [
            d for d in docs
            if infer_doc_topics(d, t, kw_index, parents, topics_cache) & req_tags
        ]

    # Quitar duplicados + ordenar (fix: titulo puede ser None)
    seen, unique = set(), []
    for d in docs:
        key = (d.metadata.get('source_table'), d.metadata.get('row_index'))
        if key not in seen:
            seen.add(key); unique.append(d)
    docs = unique
    if t=='events':
        docs.sort(key=lambda d: d.metadata.get('fecha_inicio') or date.max)
    elif t=='calls':
        docs.sort(key=lambda d: d.metadata.get('fecha_conv_inicio') or date.max)
    else:
        docs.sort(key=lambda d: (d.metadata.get('titulo') or '').lower())

    return docs

# ─────────────────────────────────────────────────────────
# Formateo de salida (con etiquetas)
# ─────────────────────────────────────────────────────────

def format_output(
    docs, t: str, clean_q: str, loc: str,
    vectordb, api_key: str, f: Dict, q: str, history: str
) -> str:
    if not docs:
        scope = f"en {loc}" if loc else "globalmente"
        return (f"**0 resultados** {scope} con los filtros actuales. "
                f"Prueba a ajustar fechas, ubicación o el término temático.")

    out = ''
    # conteos
    if t=='events' and re.search(r"cu[áa]nt(?:o|os)\s+eventos", clean_q):
        return f"**Total de eventos:** {len(docs)}"
    if t=='calls' and re.search(r"cu[áa]nt(?:a|as)\s+convocatori", clean_q):
        return f"**Total de convocatorias:** {len(docs)}"

    # Preparar taxonomía para mostrar etiquetas por doc
    kw_index = taxonomy_keywords(t)
    parents  = taxonomy_parents(t)
    topics_cache: Dict = {}

    # listados EVENTS
    if t=='events' and re.search(r"\b(?:dame|muestrame|cuales?)\b", clean_q):
        out += f"**Listado de eventos en {loc or 'Global'}:**\n\n"
        for d in docs:
            md = d.metadata
            if not md.get('fecha_inicio'):
                continue
            fi = fmt_es(md.get('fecha_inicio'))
            ff = fmt_es(md.get('fecha_fin'))
            tags = infer_doc_topics(d, t, kw_index, parents, topics_cache)
            topic_labels, _ = tag_labels_for_doc(t, tags)
            temas_line = f"  Temas: {', '.join(topic_labels)}  \n" if topic_labels else ""

            out += (
                f"- **{md.get('titulo','(sin título)')}**  \n"
                f"  Fechas: {fi} – {ff}  \n"
                f"  Ciudad: {md.get('ciudad','')} — País: {md.get('pais','—')}  \n"
                f"{temas_line}"
                f"  Tabla: events — Chunk: {md['row_index']}\n\n"
            )
        return out

    # listados CALLS
    if t=='calls' and re.search(r"\b(?:dame|muestrame|cuales?)\b", clean_q):
        limit = int(re.search(r"\b(\d+)\b", clean_q).group(1)) if re.search(r"\b(\d+)\b", clean_q) else len(docs)
        out += f"**Listado de convocatorias en {loc or 'Global'} (máx. {limit}):**\n\n"
        for d in docs[:limit]:
            md = d.metadata
            ci_date = md.get('fecha_conv_inicio') or md.get('fecha_conv_fin')
            cf_date = md.get('fecha_conv_fin') or ci_date
            if not ci_date:
                continue
            ci = fmt_es(ci_date)
            cf = fmt_es(cf_date)
            tags = infer_doc_topics(d, t, kw_index, parents, topics_cache)
            topic_labels, flag_labels = tag_labels_for_doc(t, tags)
            temas_line = f"  Temas: {', '.join(topic_labels)}  \n" if topic_labels else ""
            flags_line = f"  Criterios: {', '.join(flag_labels)}  \n" if flag_labels else ""

            out += (
                f"- **{md.get('titulo','(sin título)')}**  \n"
                f"  Convocatoria: {ci} – {cf}  \n"
                f"  País: {md.get('pais','—')} — Estado: {md.get('estado_convocatoria','—')}  \n"
                f"{temas_line}"
                f"{flags_line}"
                f"  Tabla: calls — Chunk: {md['row_index']}\n\n"
            )
        return out

    # listados DONORS
    if t=='donors' and re.search(r"\b(?:dame|muestrame|cuales?)\b", clean_q):
        out += f"**Listado de donantes en {loc or 'Global'}:**\n\n"
        for d in docs:
            md = d.metadata
            data = safe_json_loads(d.page_content)
            tags = infer_doc_topics(d, t, kw_index, parents, topics_cache)
            topic_labels, _ = tag_labels_for_doc(t, tags)
            temas_line = f"  Temas: {', '.join(topic_labels)}  \n" if topic_labels else ""

            out += (
                f"- **{md.get('titulo','(sin título)')}**  \n"
                f"  Tipo: {data.get('tipo_de_donante','—')}  \n"
                f"  Misión: {data.get('mision','—')}  \n"
                f"  Interés geográfico: {data.get('interes_geografico','—')}  \n"
                f"  Rango de financiamiento: {data.get('rango_de_financiamiento','—')}  \n"
                f"{temas_line}"
                f"  Tabla: donors — Chunk: {md['row_index']}\n\n"
            )
        return out

    # Fallback LLM: solo con los docs YA FILTRADOS (ListRetriever)
    retr = ListRetriever(docs=docs[:20])
    llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4o-mini', temperature=0)
    prompt = QA_PROMPT.partial(
        history=history,
        today=TODAY,
        location=loc or 'Global'
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever=retr,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt},
    )
    res = qa({'query': q})
    return res['result']

# ─────────────────────────────────────────────────────────
# Main / UI
# ─────────────────────────────────────────────────────────

def canonical_target_name(filename: str) -> Optional[str]:
    fn = filename.lower()
    if 'donor' in fn or 'donant' in fn or 'donante' in fn:
        return 'donors.xlsx'
    if 'call' in fn or 'convocatoria' in fn or 'proposal' in fn or 'propuesta' in fn:
        return 'calls.xlsx'
    if 'event' in fn or 'evento' in fn:
        return 'events.xlsx'
    return None

def main():
    load_dotenv(find_dotenv())
    api_key  = os.getenv('OPENAI_API_KEY')
    DATA_DIR = os.getenv('DATA_DIR','data')
    if not api_key:
        st.error('❌ No se encontró OPENAI_API_KEY.')
        return
    os.makedirs(DATA_DIR, exist_ok=True)

    st.sidebar.header('🔄 Actualizar datos')
    files = st.sidebar.file_uploader(
        'Sube donors.xlsx, calls.xlsx, events.xlsx (en cualquier nombre: detectamos y renombramos)',
        type='xlsx', accept_multiple_files=True
    )
    if st.sidebar.button('Re-indexar datos'):
        if not files or len(files) < 1:
            st.sidebar.error('Sube al menos los archivos necesarios.')
        else:
            saved = set()
            for f in files:
                target = canonical_target_name(f.name) or f.name
                with open(os.path.join(DATA_DIR, target), 'wb') as o:
                    o.write(f.getbuffer())
                saved.add(target)
                st.sidebar.write(f"📄 `{f.name}` → guardado como **{target}**")
            missing = {'donors.xlsx','calls.xlsx','events.xlsx'} - saved
            if missing:
                st.sidebar.warning(f"⚠ No se subieron: {', '.join(sorted(missing))}. "
                                   "Si faltan, el índice puede quedar incompleto.")
            try:
                sig = data_signature(DATA_DIR)
                st.session_state['vectordb']  = get_index(sig)
                st.session_state['docs_list'] = load_excels_to_docs(DATA_DIR)
                st.sidebar.success('✅ Datos re-indexados')
            except Exception as ex:
                st.sidebar.error(f"❌ Error al indexar: {ex}")

    if 'vectordb' not in st.session_state:
        st.info('Por favor sube y re-indexa los Excel.')
        return

    vectordb  = st.session_state['vectordb']
    docs_list = st.session_state['docs_list']

    st.sidebar.header('📅 Filtros (opcional)')
    sd     = st.sidebar.date_input('Desde', value=None)
    ed     = st.sidebar.date_input('Hasta', value=None)
    region = st.sidebar.selectbox('Región:', ['Global']+list(REGION_COUNTRIES.keys()))
    cc     = st.sidebar.text_input('País específico:')

    st.title('📚 MarViva AI Assistant')

    if 'history' not in st.session_state:
        st.session_state.history = []

    with st.form('chat_form', clear_on_submit=True):
        user_q   = st.text_input('Escribe tu pregunta:')
        submitted = st.form_submit_button('Enviar')
    if submitted and user_q:
        # historial
        hist_str = "\n".join(
            f"Usuario: {turn['q']}\nAI: {turn['a']}"
            for turn in reversed(st.session_state.history)
        ) or "— inicio de conversación —"

        f, loc, meses, call_range, clean_q = parse_filters(user_q, sd, ed, region, cc)
        ui_s = sd if isinstance(sd, date) else None
        ui_e = ed if isinstance(ed, date) else None

        # 1) Modo "contenido" por nombre entrecomillado
        answer_done = False
        if is_content_intent(user_q):
            name = extract_quoted_name(user_q)
            if name:
                pool = [d for d in docs_list if d.metadata.get('source_table') == f['source_table']]
                best = find_best_doc_by_title(pool, name)
                if best:
                    ans = render_full_content(best)
                    st.session_state.history.insert(0, {'q': user_q, 'a': ans})
                    answer_done = True

        if not answer_done:
            # 2) Flujo normal con filtros + taxonomía
            docs = apply_filters(
                docs_list, f, f['source_table'], meses, call_range, clean_q,
                ui_range=(ui_s, ui_e)
            )
            ans = format_output(
                docs,
                f['source_table'],
                clean_q,
                loc,
                vectordb,
                api_key,
                f,
                user_q,
                hist_str
            )
            st.session_state.history.insert(0, {'q': user_q, 'a': ans})

    # historial
    for turn in st.session_state.history:
        st.markdown(f"**Tú:** {turn['q']}")
        st.markdown(f"**AI:**  {turn['a']}")
        st.write('---')

if __name__ == '__main__':
    main()

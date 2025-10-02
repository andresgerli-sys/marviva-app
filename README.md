# marviva-app

# MarViva AI Assistant

**Búsqueda experta en Donantes, Convocatorias y Eventos (desde Excel)**

---

## 📑 Índice

1. [Descripción general](#descripción-general)  
2. [Cómo funciona (en pocas palabras)](#cómo-funciona-en-pocas-palabras)  
3. [Los 3 Excel que utiliza](#los-3-excel-que-utiliza)  
4. [Temas (taxonomía) y cómo consultarlos](#temas-taxonomía-y-cómo-consultarlos)  
5. [Filtros de fecha y región/país](#filtros-de-fecha-y-regiónpaís)  
6. [Dos modos de respuesta](#dos-modos-de-respuesta)  
7. [Uso paso a paso](#uso-paso-a-paso)  
   - [A) En la nube (Streamlit Community Cloud)](#a-en-la-nube-streamlit-community-cloud)  
   - [B) En tu computador (ejecución local)](#b-en-tu-computador-ejecución-local)  
8. [Ejemplos de consultas](#ejemplos-de-consultas)  
9. [Estructura del proyecto](#estructura-del-proyecto)  
10. [Variables de entorno](#variables-de-entorno)  
11. [requirements.txt sugerido](#requirementstxt-sugerido)  
12. [Solución de problemas](#solución-de-problemas)  
13. [Privacidad y alcance](#privacidad-y-alcance)  
14. [Licencia / Créditos](#licencia--créditos)  

---

## 📖 Descripción general

**MarViva AI Assistant** es una aplicación **Streamlit** que permite consultar, filtrar y explicar la información que MarViva mantiene en tres archivos Excel: **Donantes, Convocatorias y Eventos**.  
El asistente **no inventa datos**: responde exclusivamente en base a lo que esté cargado en esos Excel.

### Qué aporta:
- Carga y re-indexación de los Excel con un clic.  
- Búsqueda por tema gracias a una **taxonomía jerárquica y multietiqueta** (Economía azul, Gobernanza, Biodiversidad, Plásticos, Carbono azul, etc.).  
- Filtrado por **país/región** y **rango de fechas** (con lógica de solapamiento).  
- **Modo contenido**: recuperación de todas las columnas de un registro específico.  
- **Trazabilidad**: cada ítem listado incluye `Tabla: <source> — Chunk: <n>` para ubicar la fila original.  
- Fechas en **formato español**: `DD de MMMM de YYYY`.  
- Lectura robusta de Excel (doble encabezado, normalización de columnas, NaN evitados).  

---

## ⚙️ Cómo funciona (en pocas palabras)

1. Cargas los 3 Excel desde la barra lateral y pulsas **Re-indexar datos**.  
2. La app prepara un índice de búsqueda y normaliza fechas/ubicaciones.  
3. Al escribir preguntas en español, el sistema:  
   - Detecta si buscas un **contenido completo** (nombre entre comillas) o un **listado**.  
   - Aplica filtros de fecha y geografía.  
   - Etiqueta cada entrada por temas y responde con base en ellas.  
   - El **LLM solo ve los documentos ya filtrados**, evitando información fuera de los Excel.  

---

## 📊 Los 3 Excel que utiliza

### 1. `donors.xlsx` (Donantes)  
- Columnas típicas: País, Ciudad, Estado/Provincia, misión, tipo, intereses.  
- Temas: leídos de columnas como *Principales temas de interés*.  

### 2. `calls.xlsx` (Convocatorias)  
- Fechas: `fecha_conv_inicio` / `fecha_conv_fin`.  
- Estado: **Abierta, Cerrada o Rolling**.  
- Temas: *Prioridades, Objetivos u otras columnas*.  
- Flags: *Liderazgo local, GESI, Innovación, etc.*  

### 3. `events.xlsx` (Eventos)  
- Fechas: `fecha_inicio` / `fecha_fin` (o una sola).  
- Temas: de la columna *Tema de enfoque*.  

🔹 **Nota**: Los archivos pueden tener nombres distintos, pero la app los mapea internamente a `events.xlsx`, `calls.xlsx` y `donors.xlsx`.  

---

## 🗂️ Temas (taxonomía) y cómo consultarlos

- Taxonomía jerárquica por tipo (**Eventos, Convocatorias, Donantes**).  
- Cada entrada puede tener **una o varias etiquetas**.  
- Consultas posibles:  

---

## 📅 Filtros de fecha y región/país

- **Desde/Hasta** (con lógica de solapamiento).
- **Una sola fecha** también es válida.
- **Región**: Norteamérica, Centroamérica, Caribe, Europa, etc.
- **País específico**: tiene prioridad sobre región. Incluye alias (EE. UU. / US / USA, España/Espana, etc.).

---

## 📝 Dos modos de respuesta

### Modo **contenido** (todas las columnas de una fila)
**Ejemplo:**

---

## 🚀 Uso paso a paso

### A) En la nube (Streamlit Community Cloud)

1. **Crea un repo en GitHub** y sube estos archivos: `app.py`, `ingest.py`, `requirements.txt`, `.gitignore`.
2. **Despliega en Streamlit Cloud**:
   - Selecciona tu repo y branch.
   - Define **Main file path** = `app.py`.
   - En **Secrets**, añade:
     ```env
     OPENAI_API_KEY="TU_API_KEY"
     DATA_DIR="data"
     BATCH_SIZE="50"
     ```
3. **Carga los Excel** (`donors.xlsx`, `calls.xlsx`, `events.xlsx`) y pulsa **Re-indexar datos**.

> ⚠️ **Nota**: el almacenamiento en Community Cloud es efímero; si el servicio se reinicia, vuelve a subir los Excel y re-indexa.

---

### B) En tu computador (ejecución local)

- **Requisitos**: Python **3.10+** (recomendado 3.11).
- **Crea y activa** un entorno virtual:
  ```bash
  python -m venv .venv
  # macOS / Linux
  source .venv/bin/activate
  # Windows
  .venv\Scripts\activate

---

## 🧩 Instala dependencias

```bash
pip install -r requirements.txt
```

---

## 🗝️ Crea `.env` en la raíz

```env
OPENAI_API_KEY=TU_API_KEY
DATA_DIR=data
BATCH_SIZE=50
```

---

## ▶️ Ejecuta la app

```bash
streamlit run app.py
```

---

## 💡 Ejemplos de consultas

### Contenido exacto
- "UNESCO – Programa de Participación 2024-2025"
- "One Ocean Science Congress"
- "Tinker Foundation"

### Listados
- Eventos sobre biodiversidad en Centroamérica en 2025.
- 5 convocatorias de economía azul en México.
- Donantes interesados en gobernanza en Colombia.

### Conteos
- ¿Cuántos eventos hay?
- ¿Cuántas convocatorias en 2024?

---

## 📂 Estructura del proyecto

```text
.
├─ app.py        # Interfaz Streamlit
├─ ingest.py     # Lectura de Excel, indexación FAISS
├─ requirements.txt
├─ .gitignore
└─ data/         # Carpeta temporal para Excel
```

---

## 🔑 Variables de entorno

| Variable         | Obligatoria | Descripción                                         |
|------------------|-------------|-----------------------------------------------------|
| `OPENAI_API_KEY` | ✅          | API Key de OpenAI.                                  |
| `DATA_DIR`       | ✅          | Carpeta donde se guardan los Excel.                 |
| `BATCH_SIZE`     | Opcional    | Tamaño de lote al indexar (default: 50).            |

---

## 📦 `requirements.txt` sugerido

```txt
streamlit>=1.31
python-dotenv>=1.0
pandas>=2.0
openpyxl>=3.1
python-dateutil>=2.8
numpy>=1.26
tiktoken>=0.5

# LangChain + OpenAI
langchain>=0.2
langchain-community>=0.2
langchain-openai>=0.1
openai>=1.30.0

# Vector store
faiss-cpu>=1.7
```

⚠️ Para algunos entornos en la nube:

```txt
faiss-cpu==1.8.0.post1
```

---

## 🛠️ Solución de problemas

- **Falta `OPENAI_API_KEY`** → Define la variable en `.env` (local) o en **Secrets** (Cloud).
- **`ImportError: Could not import openai`** → Añade `openai` y `langchain-openai` a `requirements.txt`, guarda y reinstala.
- **Los Excel no persisten en Cloud** → Es normal por el almacenamiento efímero. Vuelve a subirlos y pulsa **Re-indexar datos**.
- **Resultados “fuera de tema”** → Amplía sinónimos en la taxonomía o revisa si la columna temática del Excel está vacía/ambigua.
- **No encuentra un nombre exacto** → Usa comillas y el título tal como aparece en el Excel (hay similitud aproximada, pero las comillas ayudan).

---

## 🔒 Privacidad y alcance

- La app no consulta fuentes externas para el contenido; responde exclusivamente con lo que está en los Excel cargados.
- `OPENAI_API_KEY` se utiliza para **embeddings** y para que el LLM **formatee y resuma** solo los documentos ya filtrados (evitando alucinaciones).

---

## 📜 Licencia / Créditos

- Proyecto interno para **MarViva**.
- Si es necesario formalizar una licencia, añádela aquí (p. ej., **MIT**, **Apache-2.0**, **GPL-3.0**).


  



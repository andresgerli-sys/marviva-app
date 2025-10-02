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

  ```text
  "Eventos sobre plásticos en Caribe este año"
  "Convocatorias de economía azul en Centroamérica entre 2025-01-01 y 2025-06-30"
  "Donantes interesados en gobernanza marina en México"

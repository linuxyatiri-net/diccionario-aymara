# %% [markdown]
# # Digitalización del Diccionario Aymara-Español de Ludovico Bertonio
# 
# Este notebook procesa imágenes escaneadas del diccionario histórico de Bertonio (1612) y extrae las entradas léxicas usando IA.
# 

# %% [markdown]
# ## 1. Instalación de dependencias

# %%
!pip install -q PyMuPDF google-generativeai pandas tqdm
!apt-get install -y poppler-utils

# %%
# Verificar instalación
import fitz  # PyMuPDF
import google.generativeai as genai
import pandas as pd
from tqdm.notebook import tqdm
import json
import re
from google.colab import userdata, files
import os

print("✅ Librerías instaladas correctamente")

# %% [markdown]
# ## 2. Configuración de API de Gemini

# %%
# @title Configurar API Key de Gemini
# Obtener la API key de los secretos de Colab
try:
    api_key = userdata.get('GEMINI_API_KEY')
except:
    api_key = input("Por favor ingresa tu API key de Google AI Studio: ")

# Configurar Gemini
genai.configure(api_key=api_key)

# Crear modelo (usamos Gemini 1.5 Flash para mejor contexto)
model = genai.GenerativeModel('gemini-1.5-flash')

print("✅ Gemini AI configurado correctamente")

# %% [markdown]
# ## 3. Subir archivos PDF

# %%
# @title Subir archivo PDF del diccionario
from google.colab import files

uploaded = files.upload()
pdf_filename = next(iter(uploaded))
print(f"✅ PDF subido: {pdf_filename}")

# %% [markdown]
# ## 4. Funciones de procesamiento

# %%
def es_pagina_con_entradas(texto):
    """
    Detecta si la página contiene entradas léxicas del diccionario.
    """
    if not texto or len(texto.strip()) < 50:
        return False
    
    # Patrones que indican páginas introductorias
    patrones_intro = [
        'vocabvlario', 'dedicado', 'compuesto', 'impreso', 
        'por francisco', 'chucuito', '1612', 'provinvia'
    ]
    
    texto_lower = texto.lower()
    if any(patron in texto_lower for patron in patrones_intro):
        return False
    
    # Contar líneas que parecen entradas (texto antes y después de punto)
    lineas = texto.split('\n')
    count_entradas = 0
    
    for linea in lineas:
        linea_limpia = linea.strip()
        if (len(linea_limpia) > 10 and 
            '. ' in linea_limpia and 
            not linea_limpia.startswith(('&','*','#','//')) and
            not any(palabra in linea_limpia.lower() for palabra in ['página', 'pág', 'folio'])):
            count_entradas += 1
    
    return count_entradas > 3

# %%
def limpiar_texto_pagina(texto):
    """
    Limpieza básica del texto de la página
    """
    # Remover números de página comunes
    texto = re.sub(r'\b\d{1,3}\b', '', texto)
    # Remover múltiples espacios
    texto = re.sub(r'\s+', ' ', texto)
    # Remover encabezados/pies de página comunes
    lineas = texto.split('\n')
    lineas_limpias = []
    
    for linea in lineas:
        linea_limpia = linea.strip()
        if (len(linea_limpia) > 5 and 
            not linea_limpia.isdigit() and
            not any(palabra in linea_limpia.lower() for palabra in ['vocabvlario', 'bertonio', 'página'])):
            lineas_limpias.append(linea_limpia)
    
    return '\n'.join(lineas_limpias)

# %%
def procesar_pagina_con_ia(texto_pagina, numero_pagina):
    """
    Envía el texto de una página a Gemini AI para extraer entradas
    """
    prompt = f"""
    Eres un experto lingüista digitalizando el diccionario Aymara-Español de Ludovico Bertonio (1612).

    **INSTRUCCIONES ESTRICTAS:**
    1. Analiza el texto de la página {numero_pagina} de un diccionario escaneado.
    2. Extrae SOLAMENTE las entradas léxicas (palabra en español + traducción en aymara).
    3. CORRIGE errores de OCR en español pero PRESERVA el texto aymara exactamente como está.
    4. Las entradas típicamente siguen el formato: "Palabra_español. Palabra_aymara"
    5. Para entradas múltiples, separa con comas: "Palabra_español. Palabra_aymara1, palabra_aymara2"
    6. Devuelve ÚNICAMENTE un array JSON válido. Nada de texto antes o después.

    **FORMATO DE SALIDA:**
    [
      {{"espanol": "palabra corregida", "aymara": "texto aymara original"}},
      {{"espanol": "otra palabra", "aymara": "texto aymara"}}
    ]

    **EJEMPLOS:**
    - "Agaulas en la garganta. Cchaña haque, añanque" → 
      {{"espanol": "Agallas en la garganta", "aymara": "Cchaña haque, añanque"}}
    
    - "Agarrar hazrendo preja. Cchumi chapicha" → 
      {{"espanol": "Agarrar haciendo presa", "aymara": "Cchumi chapicha"}}

    **TEXTO DE LA PÁGINA:**
    ```
    {texto_pagina}
    ```
    """

    try:
        response = model.generate_content(prompt)
        respuesta_texto = response.text.strip()
        
        # Limpiar respuesta (remover markdown code blocks)
        respuesta_texto_limpia = re.sub(r'```json\s*|\s*```', '', respuesta_texto)
        
        # Parsear JSON
        entradas_extraidas = json.loads(respuesta_texto_limpia)
        
        # Asegurar formato correcto
        entradas_validas = []
        for entrada in entradas_extraidas:
            if ('espanol' in entrada and 'aymara' in entrada and 
                entrada['espanol'].strip() and entrada['aymara'].strip()):
                entradas_validas.append({
                    'espanol': entrada['espanol'].strip(),
                    'aymara': entrada['aymara'].strip(),
                    'pagina': numero_pagina
                })
        
        return entradas_validas
        
    except json.JSONDecodeError as e:
        print(f"❌ Error en página {numero_pagina}: JSON inválido")
        print(f"Respuesta de IA: {respuesta_texto}")
        return []
    except Exception as e:
        print(f"❌ Error inesperado en página {numero_pagina}: {str(e)}")
        return []

# %% [markdown]
# ## 5. Procesamiento del PDF

# %%
# @title Configurar procesamiento

# Rango de páginas a procesar (ajusta según necesites)
PAGINA_INICIO = 40  # Las primeras páginas son introductorias
PAGINA_FIN = 60     # Cambia a None para procesar todo

# %%
# Abrir PDF y procesar
doc = fitz.open(pdf_filename)
total_paginas = len(doc)

if PAGINA_FIN is None:
    PAGINA_FIN = total_paginas

print(f"📖 PDF tiene {total_paginas} páginas")
print(f"🔧 Procesando páginas {PAGINA_INICIO} a {PAGINA_FIN}")

# %%
# Procesar páginas
todas_entradas = []
paginas_procesadas = 0
paginas_con_errores = []

for num_pagina in tqdm(range(PAGINA_INICIO - 1, PAGINA_FIN), desc="Procesando páginas"):
    try:
        # Extraer texto
        pagina = doc.load_page(num_pagina)
        texto_crudo = pagina.get_text("text")
        
        # Saltar páginas introductorias
        if not es_pagina_con_entradas(texto_crudo):
            continue
        
        # Limpiar texto
        texto_limpio = limpiar_texto_pagina(texto_crudo)
        
        if not texto_limpio.strip():
            continue
        
        # Procesar con IA
        entradas_pagina = procesar_pagina_con_ia(texto_limpio, num_pagina + 1)
        
        if entradas_pagina:
            todas_entradas.extend(entradas_pagina)
            paginas_procesadas += 1
            
        # Pausa para no sobrecargar la API
        import time
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Error procesando página {num_pagina + 1}: {str(e)}")
        paginas_con_errores.append(num_pagina + 1)

doc.close()

print(f"\n✅ Procesamiento completado!")
print(f"📊 Páginas procesadas: {paginas_procesadas}")
print(f"📝 Entradas extraídas: {len(todas_entradas)}")
if paginas_con_errores:
    print(f"⚠️  Páginas con errores: {paginas_con_errores}")

# %% [markdown]
# ## 6. Guardar resultados

# %%
# Crear DataFrame
if todas_entradas:
    df = pd.DataFrame(todas_entradas)
    # Reordenar columnas
    df = df[['espanol', 'aymara', 'pagina']]
    # Ordenar por página
    df = df.sort_values('pagina')
    
    # Mostrar preview
    print("📋 Primeras 10 entradas:")
    display(df.head(10))
    
    # Guardar como CSV
    csv_filename = "diccionario_aymara_espanol_bertonio.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # Guardar como Excel
    excel_filename = "diccionario_aymara_espanol_bertonio.xlsx"
    df.to_excel(excel_filename, index=False)
    
    print(f"💾 Archivos guardados:")
    print(f"   - {csv_filename}")
    print(f"   - {excel_filename}")
    
    # Descargar archivos
    files.download(csv_filename)
    files.download(excel_filename)
else:
    print("❌ No se extrajeron entradas. Revisa el rango de páginas.")

# %% [markdown]
# ## 7. Análisis de resultados

# %%
if todas_entradas:
    print("📈 Estadísticas:")
    print(f"   - Total entradas: {len(df)}")
    print(f"   - Páginas procesadas: {df['pagina'].nunique()}")
    print(f"   - Promedio entradas por página: {len(df) / df['pagina'].nunique():.1f}")
    
    # Mostrar algunas entradas de ejemplo
    print("\n🔍 Ejemplos de entradas extraídas:")
    for i, row in df.sample(5).iterrows():
        print(f"   - {row['espanol']} → {row['aymara']} (pág. {row['pagina']})")

# %% [markdown]
# ## 8. Procesamiento adicional (opcional)

# %%
# @title Limpieza adicional de datos

if todas_entradas:
    # Eliminar duplicados
    df_limpio = df.drop_duplicates(subset=['espanol', 'aymara'])
    
    # Ordenar alfabéticamente por español
    df_limpio = df_limpio.sort_values('espanol')
    
    # Filtrar entradas muy cortas (probablemente errores)
    df_limpio = df_limpio[df_limpio['espanol'].str.len() > 3]
    df_limpio = df_limpio[df_limpio['aymara'].str.len() > 2]
    
    print(f"🧹 Entradas después de limpieza: {len(df_limpio)}")
    
    # Guardar versión limpia
    csv_limpio = "diccionario_aymara_espanol_bertonio_limpio.csv"
    df_limpio.to_csv(csv_limpio, index=False, encoding='utf-8-sig')
    files.download(csv_limpio)

# %% [markdown]
# ## 9. Procesar más páginas

# %%
# @title Continuar procesamiento desde última página

if todas_entradas:
    ultima_pagina = df['pagina'].max()
    print(f"➡️  Última página procesada: {ultima_pagina}")
    print(f"📖 Páginas restantes: {total_paginas - ultima_pagina}")

    continuar = input("¿Continuar procesamiento desde la última página? (s/n): ")
    if continuar.lower() == 's':
        # Aquí puedes ejecutar de nuevo el procesamiento desde ultima_pagina + 1
        print("Ejecuta de nuevo la celda de procesamiento cambiando PAGINA_INICIO")